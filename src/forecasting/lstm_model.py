"""
LSTM Forecaster for SocialProphet.

Deep Learning approach using LSTM networks for time series forecasting.

Note: With 292 training samples, LSTM performance may be limited.
Deep learning typically requires thousands of samples for optimal performance.
If Prophet/SARIMA outperform LSTM, this is expected and academically valid.

Architecture:
- 2 LSTM layers with 50 units each
- Dropout layers (0.2) for regularization
- Dense output layer

Weight in ensemble: 25%
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

# TensorFlow/Keras imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. LSTM functionality unavailable.")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class LSTMForecaster:
    """
    LSTM-based time series forecaster.

    Handles log-transformed engagement data with sequence windowing.

    Data Flow:
    1. Input: y (log scale) + features
    2. MinMax scale all features
    3. Create sequences (window_size timesteps)
    4. Train LSTM
    5. Predict -> inverse MinMax -> log scale -> expm1 -> original scale
    """

    # Architecture parameters from requirements
    DEFAULT_PARAMS = {
        'n_units': 50,
        'n_layers': 2,
        'window_size': 30,  # Look-back window
        'epochs': 100,
        'batch_size': 16,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'early_stopping_patience': 10,
        'validation_split': 0.1,
    }

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize LSTM forecaster.

        Args:
            params: Model parameters (see DEFAULT_PARAMS)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        self.history = None
        self.feature_columns = None
        self.n_features = None

    def _create_sequences(
        self,
        data: np.ndarray,
        window_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.

        Args:
            data: Scaled feature array (n_samples, n_features)
            window_size: Look-back window (30 days)

        Returns:
            X: (n_sequences, window_size, n_features)
            y: (n_sequences,) - target is first column
        """
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size:i])  # Previous window_size days
            y.append(data[i, 0])  # Target (first column = y)
        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM architecture.

        Architecture (per requirements):
        - 2 LSTM layers, 50 units each
        - Dropout for regularization
        - Dense output layer

        Args:
            input_shape: (window_size, n_features)

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # Input layer
            Input(shape=input_shape),

            # First LSTM layer
            LSTM(
                units=self.params['n_units'],
                return_sequences=True  # Return sequences for stacking
            ),
            Dropout(self.params['dropout']),

            # Second LSTM layer
            LSTM(
                units=self.params['n_units'],
                return_sequences=False
            ),
            Dropout(self.params['dropout']),

            # Dense layers
            Dense(25, activation='relu'),

            # Output layer
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )

        return model

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'y',
        feature_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Prepare feature matrix for LSTM.

        Args:
            df: DataFrame with features
            target_col: Target column name ('y' - log scale)
            feature_cols: Feature columns to include

        Returns:
            Feature array with target as first column
        """
        # Default features for LSTM
        if feature_cols is None:
            feature_cols = [
                'y',  # Target (log scale) - MUST be first
                'y_lag_1', 'y_lag_7', 'y_lag_14',
                'y_rolling_mean_7', 'y_rolling_std_7',
                'day_sin', 'day_cos',
                'is_weekend'
            ]

        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]

        # Ensure target is first
        if target_col in available_cols:
            available_cols.remove(target_col)
        available_cols = [target_col] + available_cols

        self.feature_columns = available_cols
        self.n_features = len(available_cols)

        print(f"  Using {self.n_features} features: {available_cols}")

        return df[available_cols].values

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        verbose: int = 1
    ) -> "LSTMForecaster":
        """
        Train LSTM model.

        Args:
            df: Training DataFrame with log-scale target 'y'
            feature_cols: Feature columns to use
            verbose: Training verbosity (0, 1, 2)

        Returns:
            self
        """
        print(f"\nTraining LSTM on {len(df)} observations...")
        print(f"  Window size: {self.params['window_size']}")
        print(f"  Epochs: {self.params['epochs']} (with early stopping)")

        # Prepare features
        features = self.prepare_features(df, feature_cols=feature_cols)

        # Scale features
        scaled_features = self.scaler.fit_transform(features)

        # Create sequences
        window_size = self.params['window_size']
        X, y = self._create_sequences(scaled_features, window_size)

        print(f"  Sequences created: {X.shape[0]} samples")
        print(f"  Input shape: {X.shape}")

        # Build model
        self.model = self._build_model(input_shape=(X.shape[1], X.shape[2]))

        if verbose > 0:
            self.model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.params['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            )
        ]

        # Train
        self.history = self.model.fit(
            X, y,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_split=self.params['validation_split'],
            callbacks=callbacks,
            verbose=verbose
        )

        self.is_fitted = True
        print("\nLSTM model fitted successfully!")
        print(f"  Final training loss: {self.history.history['loss'][-1]:.6f}")
        print(f"  Final validation loss: {self.history.history['val_loss'][-1]:.6f}")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for given data.

        Args:
            df: DataFrame with features

        Returns:
            Predictions in LOG SCALE
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare features
        features = df[self.feature_columns].values

        # Scale features
        scaled_features = self.scaler.transform(features)

        # Create sequences
        window_size = self.params['window_size']
        X, _ = self._create_sequences(scaled_features, window_size)

        # Predict
        predictions_scaled = self.model.predict(X, verbose=0)

        # Inverse scale (only target column)
        predictions_log = self._inverse_scale_predictions(predictions_scaled.flatten())

        return predictions_log

    def _inverse_scale_predictions(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Convert scaled predictions back to log scale.

        Args:
            y_scaled: Scaled predictions from model

        Returns:
            Log-scale predictions
        """
        # Create dummy array with y in first column
        dummy = np.zeros((len(y_scaled), self.n_features))
        dummy[:, 0] = y_scaled
        unscaled = self.scaler.inverse_transform(dummy)
        return unscaled[:, 0]  # Return only target column (log scale)

    def predict_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate predictions for test set.

        Requires combining train and test to create proper sequences.

        Args:
            train_df: Training data
            test_df: Test data

        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Combine train and test for sequence creation
        combined_df = pd.concat([train_df, test_df], ignore_index=True)

        # Prepare features
        features = combined_df[self.feature_columns].values
        scaled_features = self.scaler.transform(features)

        # Create sequences
        window_size = self.params['window_size']
        X, _ = self._create_sequences(scaled_features, window_size)

        # Predict all
        predictions_scaled = self.model.predict(X, verbose=0)
        predictions_log = self._inverse_scale_predictions(predictions_scaled.flatten())

        # Extract test predictions (last len(test_df) predictions)
        test_start_idx = len(train_df) - window_size
        test_predictions_log = predictions_log[test_start_idx:]

        # Align with test data length
        test_predictions_log = test_predictions_log[:len(test_df)]

        # Build result DataFrame
        result = pd.DataFrame({
            'yhat': test_predictions_log,
            'yhat_original': self.inverse_transform(test_predictions_log)
        })

        if 'y' in test_df.columns:
            result['y_actual'] = test_df['y'].values[:len(result)]
            result['y_actual_original'] = self.inverse_transform(
                test_df['y'].values[:len(result)]
            )

        if 'y_raw' in test_df.columns:
            result['y_raw'] = test_df['y_raw'].values[:len(result)]

        return result

    def inverse_transform(self, y_log: np.ndarray) -> np.ndarray:
        """Convert log-scale to original scale."""
        return np.expm1(y_log)

    def evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Evaluate model on test set.

        Args:
            train_df: Training data (needed for sequence context)
            test_df: Test data

        Returns:
            dict with evaluation metrics (on original scale)
        """
        predictions = self.predict_test(train_df, test_df)

        y_pred = predictions['yhat_original'].values

        # Get actual values
        if 'y_raw' in test_df.columns:
            y_true = test_df['y_raw'].values[:len(y_pred)]
        else:
            y_true = self.inverse_transform(test_df['y'].values[:len(y_pred)])

        # Calculate metrics
        mask = y_true != 0  # Avoid division by zero
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        # R² calculation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        results = {
            'model': 'LSTM',
            'n_test': len(y_pred),
            'params': self.params,
            'metrics_original_scale': {
                'mape': float(mape),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
            },
            'predictions': predictions
        }

        print(f"\nLSTM Evaluation Results (Original Scale):")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE: {mae:,.2f}")
        print(f"  R²: {r2:.4f}")

        # Note about dataset size
        if mape > 15 or r2 < 0.7:
            print(f"\n  Note: LSTM performance may be limited due to small dataset size")
            print(f"  ({len(train_df)} samples is borderline for deep learning)")

        return results

    def get_training_history(self) -> pd.DataFrame:
        """Return training loss history."""
        if self.history is None:
            raise ValueError("No training history. Call fit() first.")

        return pd.DataFrame(self.history.history)

    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save model, scaler, and parameters.

        Args:
            filepath: Output directory path

        Returns:
            Path to saved directory
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if filepath is None:
            Config.ensure_directories()
            filepath = Config.PREDICTIONS_DIR / "lstm_model"

        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        self.model.save(filepath / "model.keras")

        # Save scaler and metadata
        with open(filepath / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'params': self.params,
                'feature_columns': self.feature_columns,
                'n_features': self.n_features,
                'is_fitted': self.is_fitted
            }, f)

        print(f"LSTM model saved: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "LSTMForecaster":
        """Load saved model."""
        filepath = Path(filepath)

        # Load metadata
        with open(filepath / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        forecaster = cls(params=metadata['params'])
        forecaster.scaler = metadata['scaler']
        forecaster.feature_columns = metadata['feature_columns']
        forecaster.n_features = metadata['n_features']
        forecaster.is_fitted = metadata['is_fitted']

        # Load Keras model
        forecaster.model = load_model(filepath / "model.keras")

        print(f"LSTM model loaded from: {filepath}")
        return forecaster


def run_lstm_forecast():
    """Run LSTM forecasting on training data."""
    print("\n" + "#" * 60)
    print("# SocialProphet - LSTM Forecasting")
    print("#" * 60)

    # Check TensorFlow
    if not TENSORFLOW_AVAILABLE:
        print("ERROR: TensorFlow not available. Install with: pip install tensorflow")
        return None, None

    # Load data
    train_path = Config.PROCESSED_DATA_DIR / "train_data.csv"
    test_path = Config.PROCESSED_DATA_DIR / "test_data.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"\nTrain: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")

    # Initialize and fit
    forecaster = LSTMForecaster()
    forecaster.fit(train_df, verbose=1)

    # Evaluate on test set
    results = forecaster.evaluate(train_df, test_df)

    # Save model
    forecaster.save()

    print("\n" + "#" * 60)
    print("# LSTM Forecasting Complete!")
    print("#" * 60)

    return forecaster, results


if __name__ == "__main__":
    run_lstm_forecast()
