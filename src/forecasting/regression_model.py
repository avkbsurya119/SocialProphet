"""
Regression-based Forecaster for SocialProphet.

Uses sklearn gradient boosting with lag features for engagement prediction.
This approach often works better than pure time series models when:
- Data has high day-to-day variance
- Strong autocorrelation (today predicts tomorrow)
- Limited historical data (< 500 samples)

This model replaces SARIMA in the ensemble due to SARIMA's poor performance.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class RegressionForecaster:
    """
    Gradient Boosting regression forecaster with lag features.

    Uses recent history (lags) to predict next day's engagement.
    Works on ORIGINAL SCALE for better variance capture.
    """

    DEFAULT_PARAMS = {
        'alpha': 100,  # Ridge regularization
    }

    def __init__(self, params: Optional[Dict] = None, use_simple: bool = True):
        """
        Initialize regression forecaster.

        Args:
            params: Model parameters
            use_simple: Use simple Ridge model (recommended to avoid overfitting)
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = None
        self.y_mean = None
        self.y_std = None
        self.use_simple = use_simple

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix from DataFrame.

        Uses simple rolling mean features to avoid overfitting.
        """
        df = df.copy()

        # Use y_raw if available, otherwise convert y
        if 'y_raw' not in df.columns:
            df['y_raw'] = np.expm1(df['y'])

        if self.use_simple:
            # Simple features: rolling means only (avoid overfitting)
            feature_names = []

            # Rolling means (shifted by 1 to avoid lookahead)
            for window in [7, 14, 21]:
                col = f'rolling_mean_{window}'
                df[col] = df['y_raw'].rolling(window, min_periods=1).mean().shift(1)
                feature_names.append(col)

            # Day-of-week encoding (one feature, not one-hot)
            if 'day_of_week' in df.columns:
                feature_names.append('day_of_week')

            # Fill NaN with mean
            for col in feature_names:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())

            self.feature_columns = feature_names
            return df[feature_names].values, feature_names
        else:
            # Complex features (more prone to overfitting)
            features = []
            feature_names = []

            # Lag features
            for lag in [1, 7, 14]:
                col_name = f'y_raw_lag_{lag}'
                df[col_name] = df['y_raw'].shift(lag)
                features.append(col_name)
                feature_names.append(col_name)

            # Rolling statistics
            for window in [7, 14]:
                mean_col = f'y_raw_rolling_mean_{window}'
                df[mean_col] = df['y_raw'].rolling(window, min_periods=1).mean().shift(1)
                features.append(mean_col)
                feature_names.append(mean_col)

            # Temporal features
            for f in ['day_of_week', 'is_weekend']:
                if f in df.columns:
                    features.append(f)
                    feature_names.append(f)

            df_clean = df[features].dropna()
            self.feature_columns = feature_names
            return df_clean.values, feature_names

    def fit(self, df: pd.DataFrame, verbose: int = 1) -> "RegressionForecaster":
        """
        Train regression model.

        Args:
            df: Training DataFrame with y_raw and features
            verbose: Verbosity level

        Returns:
            self
        """
        print(f"Training Regression model on {len(df)} observations...")

        # Prepare features
        X, self.feature_columns = self._prepare_features(df)

        # Get target (y_raw)
        y_col = 'y_raw' if 'y_raw' in df.columns else 'y'
        if y_col == 'y':
            y = np.expm1(df[y_col].values)
        else:
            y = df[y_col].values

        # Align target with features (drop first rows due to lags)
        n_dropped = len(df) - len(X)
        y = y[n_dropped:]

        self.y_mean = y.mean()
        self.y_std = y.std()

        print(f"  Features: {len(self.feature_columns)}")
        print(f"  y_raw range: [{y.min():.0f}, {y.max():.0f}]")
        print(f"  y_raw mean: {self.y_mean:.0f}, std: {self.y_std:.0f}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        if self.use_simple:
            self.model = Ridge(alpha=self.params.get('alpha', 100))
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                min_samples_split=10, min_samples_leaf=5
            )
        self.model.fit(X_scaled, y)

        self.is_fitted = True

        # In-sample R²
        y_pred = self.model.predict(X_scaled)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        print(f"  In-sample R²: {r2:.4f}")

        print("Regression model fitted successfully!")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            df: DataFrame with features

        Returns:
            Predictions in original scale
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X, _ = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        return np.clip(predictions, 0, None)

    def predict_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate predictions for test set.

        Combines train and test to create proper lag features.

        Args:
            train_df: Training data
            test_df: Test data

        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Combine train and test for proper lag calculation
        combined = pd.concat([train_df.tail(30), test_df], ignore_index=True)

        # Prepare features
        X, _ = self._prepare_features(combined)

        # We only need predictions for test portion
        n_test = len(test_df)
        X_test = X[-n_test:]

        X_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_scaled)
        predictions = np.clip(predictions, 0, None)

        # Build result DataFrame
        result = pd.DataFrame({
            'yhat_original': predictions,
            'yhat': np.log1p(predictions)  # Log scale for compatibility
        })

        if 'y' in test_df.columns:
            result['y_actual'] = test_df['y'].values[:len(result)]
            result['y_actual_original'] = np.expm1(test_df['y'].values[:len(result)])

        if 'y_raw' in test_df.columns:
            result['y_raw'] = test_df['y_raw'].values[:len(result)]

        return result

    def evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Evaluate model on test set.

        Args:
            train_df: Training data
            test_df: Test data

        Returns:
            dict with evaluation metrics
        """
        predictions = self.predict_test(train_df, test_df)

        y_pred = predictions['yhat_original'].values

        # Get actual values
        if 'y_raw' in test_df.columns:
            y_true = test_df['y_raw'].values[:len(y_pred)]
        else:
            y_true = np.expm1(test_df['y'].values[:len(y_pred)])

        # Calculate metrics
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        results = {
            'model': 'Regression',
            'n_test': len(y_pred),
            'metrics_original_scale': {
                'mape': float(mape),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
            },
            'predictions': predictions
        }

        print(f"\nRegression Evaluation Results (Original Scale):")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE: {mae:,.2f}")
        print(f"  R²: {r2:.4f}")

        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if filepath is None:
            Config.ensure_directories()
            filepath = Config.PREDICTIONS_DIR / "regression_model.pkl"

        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'params': self.params,
                'feature_columns': self.feature_columns,
                'y_mean': self.y_mean,
                'y_std': self.y_std,
                'is_fitted': self.is_fitted
            }, f)

        print(f"Regression model saved: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "RegressionForecaster":
        """Load model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        forecaster = cls(params=data['params'])
        forecaster.model = data['model']
        forecaster.scaler = data['scaler']
        forecaster.feature_columns = data['feature_columns']
        forecaster.y_mean = data['y_mean']
        forecaster.y_std = data['y_std']
        forecaster.is_fitted = data['is_fitted']

        print(f"Regression model loaded from: {filepath}")
        return forecaster


def run_regression_forecast():
    """Run regression forecasting."""
    print("\n" + "#" * 60)
    print("# SocialProphet - Regression Forecasting")
    print("#" * 60)

    # Load data
    train_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "test_data.csv")

    print(f"\nTrain: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")

    # Initialize and fit
    forecaster = RegressionForecaster()
    forecaster.fit(train_df)

    # Evaluate
    results = forecaster.evaluate(train_df, test_df)

    # Feature importance
    print("\nTop 5 features:")
    importance = forecaster.get_feature_importance()
    print(importance.head())

    # Save
    forecaster.save()

    print("\n" + "#" * 60)
    print("# Regression Forecasting Complete!")
    print("#" * 60)

    return forecaster, results


if __name__ == "__main__":
    run_regression_forecast()
