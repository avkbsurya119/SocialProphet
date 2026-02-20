"""
Ensemble Forecaster for SocialProphet.

Combines Prophet, SARIMA, and LSTM predictions using weighted averaging.

Default Weights:
- Prophet: 40% (best for trend + seasonality)
- SARIMA: 35% (statistical rigor, short-term)
- LSTM: 25% (non-linear patterns)

The ensemble approach provides more robust predictions than any single model.
"""

import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config

from .prophet_model import ProphetForecaster
from .sarima_model import SARIMAForecaster
from .lstm_model import LSTMForecaster, TENSORFLOW_AVAILABLE


class EnsembleForecaster:
    """
    Ensemble of multiple forecasting models.

    Combines predictions from Prophet, SARIMA, and LSTM using weighted averaging.
    Handles missing models gracefully by adjusting weights.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        models: Optional[Dict] = None
    ):
        """
        Initialize ensemble forecaster.

        Args:
            weights: Model weights (default from Config.ENSEMBLE_WEIGHTS)
                - prophet: 0.4
                - sarima: 0.35
                - lstm: 0.25
            models: Pre-fitted models dict
        """
        default_weights = Config.ENSEMBLE_WEIGHTS.copy()
        self.weights = weights if weights is not None else default_weights
        self.models = models if models is not None else {}
        self.predictions = {}
        self.is_fitted = False
        self.available_models = []

    def add_model(
        self,
        name: str,
        model,
        weight: Optional[float] = None
    ) -> None:
        """
        Add a model to the ensemble.

        Args:
            name: Model identifier ('prophet', 'sarima', 'lstm')
            model: Fitted model instance
            weight: Override default weight
        """
        self.models[name] = model
        if weight is not None:
            self.weights[name] = weight
        self.available_models.append(name)
        print(f"Added {name} model to ensemble (weight: {self.weights.get(name, 0)})")

    def fit_all(
        self,
        train_df: pd.DataFrame,
        train_prophet_df: Optional[pd.DataFrame] = None,
        fit_lstm: bool = True
    ) -> "EnsembleForecaster":
        """
        Fit all component models.

        Args:
            train_df: Full training data with features
            train_prophet_df: Prophet-format data (ds, y)
            fit_lstm: Whether to fit LSTM model
        """
        print("\n" + "=" * 60)
        print("FITTING ENSEMBLE MODELS")
        print("=" * 60)

        # Prepare Prophet data
        if train_prophet_df is None:
            train_prophet_df = train_df[['ds', 'y']].copy()
            train_prophet_df['ds'] = pd.to_datetime(train_prophet_df['ds'])

        # 1. Fit Prophet
        print("\n--- Prophet ---")
        try:
            prophet = ProphetForecaster()
            prophet.fit(train_prophet_df)
            self.add_model('prophet', prophet)
        except Exception as e:
            print(f"Prophet fitting failed: {e}")

        # 2. Fit SARIMA
        print("\n--- SARIMA ---")
        try:
            sarima = SARIMAForecaster(auto_select=True)
            sarima.fit(pd.Series(train_df['y'].values))
            self.add_model('sarima', sarima)
        except Exception as e:
            print(f"SARIMA fitting failed: {e}")

        # 3. Fit LSTM (optional)
        if fit_lstm and TENSORFLOW_AVAILABLE:
            print("\n--- LSTM ---")
            try:
                lstm = LSTMForecaster()
                lstm.fit(train_df, verbose=0)
                self.add_model('lstm', lstm)
            except Exception as e:
                print(f"LSTM fitting failed: {e}")
        elif not TENSORFLOW_AVAILABLE:
            print("\n--- LSTM ---")
            print("Skipping LSTM (TensorFlow not available)")

        # Normalize weights for available models
        self._normalize_weights()

        self.is_fitted = len(self.available_models) > 0
        print(f"\nEnsemble fitted with {len(self.available_models)} models")
        print(f"Normalized weights: {self.weights}")

        return self

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1 for available models."""
        available_weights = {
            name: self.weights.get(name, 0)
            for name in self.available_models
        }

        total_weight = sum(available_weights.values())

        if total_weight > 0:
            self.weights = {
                name: w / total_weight
                for name, w in available_weights.items()
            }

    def predict(
        self,
        test_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions.

        Args:
            test_df: Test data
            train_df: Training data (needed for LSTM context)

        Returns:
            DataFrame with individual and ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit_all() first.")

        print("\nGenerating ensemble predictions...")

        predictions = {}
        test_dates = pd.to_datetime(test_df['ds']) if 'ds' in test_df.columns else None

        # Prophet predictions
        if 'prophet' in self.models:
            prophet_preds = self.models['prophet'].predict_test(test_df)
            predictions['prophet'] = prophet_preds['yhat'].values
            predictions['prophet_original'] = prophet_preds['yhat_original'].values
            print(f"  Prophet predictions: {len(predictions['prophet'])} values")

        # SARIMA predictions
        if 'sarima' in self.models:
            sarima_preds = self.models['sarima'].predict(steps=len(test_df))
            predictions['sarima'] = sarima_preds['yhat'].values
            predictions['sarima_original'] = sarima_preds['yhat_original'].values
            print(f"  SARIMA predictions: {len(predictions['sarima'])} values")

        # LSTM predictions
        if 'lstm' in self.models and train_df is not None:
            lstm_preds = self.models['lstm'].predict_test(train_df, test_df)
            predictions['lstm'] = lstm_preds['yhat'].values
            predictions['lstm_original'] = lstm_preds['yhat_original'].values
            print(f"  LSTM predictions: {len(predictions['lstm'])} values")

        # Store predictions
        self.predictions = predictions

        # Compute weighted ensemble
        ensemble_log = self.weighted_average(
            {k: v for k, v in predictions.items() if not k.endswith('_original')}
        )
        ensemble_original = self.inverse_transform(ensemble_log)

        # Build result DataFrame
        result = pd.DataFrame()

        if test_dates is not None:
            result['ds'] = test_dates.values

        # Individual model predictions
        for name in self.available_models:
            if name in predictions:
                result[f'{name}_pred'] = predictions[name]
                result[f'{name}_pred_original'] = predictions[f'{name}_original']

        # Ensemble prediction
        result['ensemble_pred'] = ensemble_log
        result['ensemble_pred_original'] = ensemble_original

        # Actual values
        if 'y' in test_df.columns:
            result['y_actual'] = test_df['y'].values[:len(result)]
            result['y_actual_original'] = self.inverse_transform(
                test_df['y'].values[:len(result)]
            )

        if 'y_raw' in test_df.columns:
            result['y_raw'] = test_df['y_raw'].values[:len(result)]

        return result

    def weighted_average(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute weighted average of predictions.

        Args:
            predictions: Dict of model_name -> predictions array (log scale)

        Returns:
            Weighted average predictions (log scale)
        """
        # Find minimum length
        min_len = min(len(p) for p in predictions.values())

        # Initialize
        weighted_sum = np.zeros(min_len)
        total_weight = 0

        for name, preds in predictions.items():
            if name in self.weights:
                weight = self.weights[name]
                weighted_sum += weight * preds[:min_len]
                total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback: simple average
            return np.mean([p[:min_len] for p in predictions.values()], axis=0)

    def inverse_transform(self, y_log: np.ndarray) -> np.ndarray:
        """Convert log-scale to original scale."""
        return np.expm1(y_log)

    def evaluate(
        self,
        test_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Evaluate ensemble and individual models on test set.

        Args:
            test_df: Test data
            train_df: Training data (needed for LSTM)

        Returns:
            Dict with metrics for each model and ensemble
        """
        predictions_df = self.predict(test_df, train_df)

        # Get actual values (original scale)
        if 'y_raw' in test_df.columns:
            y_true = test_df['y_raw'].values[:len(predictions_df)]
        else:
            y_true = self.inverse_transform(test_df['y'].values[:len(predictions_df)])

        results = {
            'n_test': len(predictions_df),
            'models': {},
            'ensemble': {},
            'weights': self.weights.copy()
        }

        # Evaluate each model
        for name in self.available_models:
            col = f'{name}_pred_original'
            if col in predictions_df.columns:
                y_pred = predictions_df[col].values
                metrics = self._calculate_metrics(y_true, y_pred)
                results['models'][name] = metrics

        # Evaluate ensemble
        y_ensemble = predictions_df['ensemble_pred_original'].values
        results['ensemble'] = self._calculate_metrics(y_true, y_ensemble)

        # Print results
        self._print_evaluation_results(results)

        return results

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Calculate evaluation metrics."""
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'mape': float(mape),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }

    def _print_evaluation_results(self, results: Dict) -> None:
        """Print formatted evaluation results."""
        print("\n" + "=" * 60)
        print("ENSEMBLE EVALUATION RESULTS (Original Scale)")
        print("=" * 60)

        print(f"\nWeights: {results['weights']}")

        print("\n" + "-" * 60)
        print("Individual Model Performance:")
        print("-" * 60)
        print(f"{'Model':<12} {'MAPE':>10} {'RMSE':>12} {'R²':>10}")
        print("-" * 60)

        for name, metrics in results['models'].items():
            print(f"{name:<12} {metrics['mape']:>9.2f}% {metrics['rmse']:>11,.0f} {metrics['r2']:>10.4f}")

        print("-" * 60)
        print(f"{'ENSEMBLE':<12} {results['ensemble']['mape']:>9.2f}% {results['ensemble']['rmse']:>11,.0f} {results['ensemble']['r2']:>10.4f}")
        print("=" * 60)

        # Check thresholds
        print("\nThreshold Check:")
        print(f"  MAPE < 15%: {'PASS' if results['ensemble']['mape'] < 15 else 'FAIL'} ({results['ensemble']['mape']:.2f}%)")
        print(f"  R² > 0.70: {'PASS' if results['ensemble']['r2'] > 0.70 else 'FAIL'} ({results['ensemble']['r2']:.4f})")

    def optimize_weights(
        self,
        val_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        metric: str = 'mape'
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights on validation set.

        Uses grid search to find optimal weights.

        Args:
            val_df: Validation data
            train_df: Training data (for LSTM context)
            metric: Optimization metric ('mape', 'rmse')

        Returns:
            Optimized weights
        """
        print("\nOptimizing ensemble weights...")

        # Get individual predictions
        self.predict(val_df, train_df)

        # Get actual values
        if 'y_raw' in val_df.columns:
            y_true = val_df['y_raw'].values[:len(self.predictions.get('prophet', []))]
        else:
            y_true = self.inverse_transform(val_df['y'].values)

        best_score = float('inf') if metric in ['mape', 'rmse'] else float('-inf')
        best_weights = self.weights.copy()

        # Grid search
        model_names = self.available_models
        n_models = len(model_names)

        if n_models == 3:
            # 3-way grid search
            for w1 in np.arange(0.1, 0.8, 0.1):
                for w2 in np.arange(0.1, 0.9 - w1, 0.1):
                    w3 = 1.0 - w1 - w2
                    if w3 >= 0.1:
                        test_weights = {
                            model_names[0]: w1,
                            model_names[1]: w2,
                            model_names[2]: w3
                        }
                        score = self._evaluate_weights(test_weights, y_true, metric)

                        if (metric in ['mape', 'rmse'] and score < best_score) or \
                           (metric == 'r2' and score > best_score):
                            best_score = score
                            best_weights = test_weights.copy()

        elif n_models == 2:
            # 2-way grid search
            for w1 in np.arange(0.1, 1.0, 0.1):
                w2 = 1.0 - w1
                test_weights = {
                    model_names[0]: w1,
                    model_names[1]: w2
                }
                score = self._evaluate_weights(test_weights, y_true, metric)

                if (metric in ['mape', 'rmse'] and score < best_score) or \
                   (metric == 'r2' and score > best_score):
                    best_score = score
                    best_weights = test_weights.copy()

        self.weights = best_weights
        print(f"Optimized weights: {best_weights}")
        print(f"Best {metric.upper()}: {best_score:.4f}")

        return best_weights

    def _evaluate_weights(
        self,
        weights: Dict[str, float],
        y_true: np.ndarray,
        metric: str
    ) -> float:
        """Evaluate specific weight combination."""
        # Compute weighted average
        min_len = min(len(y_true), min(
            len(self.predictions.get(name, [])) for name in weights
        ))

        weighted_sum = np.zeros(min_len)
        for name, weight in weights.items():
            if name in self.predictions:
                weighted_sum += weight * self.predictions[name][:min_len]

        y_pred = self.inverse_transform(weighted_sum)
        y_true_trimmed = y_true[:min_len]

        if metric == 'mape':
            mask = y_true_trimmed != 0
            return np.mean(np.abs((y_true_trimmed[mask] - y_pred[mask]) / y_true_trimmed[mask])) * 100
        elif metric == 'rmse':
            return np.sqrt(np.mean((y_true_trimmed - y_pred) ** 2))
        elif metric == 'r2':
            ss_res = np.sum((y_true_trimmed - y_pred) ** 2)
            ss_tot = np.sum((y_true_trimmed - np.mean(y_true_trimmed)) ** 2)
            return 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_model_contributions(self) -> pd.DataFrame:
        """Analyze individual model contributions to ensemble."""
        contributions = pd.DataFrame({
            'model': list(self.weights.keys()),
            'weight': list(self.weights.values()),
            'weight_pct': [w * 100 for w in self.weights.values()]
        })
        return contributions

    def save(self, dirpath: Optional[Path] = None) -> Path:
        """
        Save all models and ensemble config.

        Args:
            dirpath: Output directory

        Returns:
            Path to saved directory
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit_all() first.")

        if dirpath is None:
            Config.ensure_directories()
            dirpath = Config.PREDICTIONS_DIR / "ensemble"

        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        # Save individual models
        for name, model in self.models.items():
            if name == 'prophet':
                model.save(dirpath / "prophet_model.pkl")
            elif name == 'sarima':
                model.save(dirpath / "sarima_model.pkl")
            elif name == 'lstm':
                model.save(dirpath / "lstm_model")

        # Save ensemble config
        with open(dirpath / "ensemble_config.json", 'w') as f:
            json.dump({
                'weights': self.weights,
                'available_models': self.available_models
            }, f, indent=2)

        print(f"Ensemble saved: {dirpath}")
        return dirpath

    @classmethod
    def load(cls, dirpath: Path) -> "EnsembleForecaster":
        """Load ensemble from directory."""
        dirpath = Path(dirpath)

        # Load config
        with open(dirpath / "ensemble_config.json") as f:
            config = json.load(f)

        ensemble = cls(weights=config['weights'])
        ensemble.available_models = config['available_models']

        # Load models
        if 'prophet' in config['available_models']:
            prophet_path = dirpath / "prophet_model.pkl"
            if prophet_path.exists():
                ensemble.models['prophet'] = ProphetForecaster.load(prophet_path)

        if 'sarima' in config['available_models']:
            sarima_path = dirpath / "sarima_model.pkl"
            if sarima_path.exists():
                ensemble.models['sarima'] = SARIMAForecaster.load(sarima_path)

        if 'lstm' in config['available_models']:
            lstm_path = dirpath / "lstm_model"
            if lstm_path.exists():
                ensemble.models['lstm'] = LSTMForecaster.load(lstm_path)

        ensemble.is_fitted = True
        print(f"Ensemble loaded from: {dirpath}")
        return ensemble


def run_ensemble_forecast():
    """Run complete ensemble forecasting."""
    print("\n" + "#" * 60)
    print("# SocialProphet - Ensemble Forecasting")
    print("#" * 60)

    # Load data
    train_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "test_data.csv")
    train_prophet = pd.read_csv(Config.PROCESSED_DATA_DIR / "train_prophet.csv")

    print(f"\nTrain: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")

    # Initialize and fit ensemble
    ensemble = EnsembleForecaster()
    ensemble.fit_all(train_df, train_prophet, fit_lstm=TENSORFLOW_AVAILABLE)

    # Evaluate
    results = ensemble.evaluate(test_df, train_df)

    # Save ensemble
    ensemble.save()

    # Save results
    results_path = Config.PROCESSED_DATA_DIR / "ensemble_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'weights': results['weights'],
            'ensemble_metrics': results['ensemble'],
            'model_metrics': results['models'],
            'evaluated_at': datetime.now().isoformat()
        }, f, indent=2)
    print(f"\nResults saved: {results_path}")

    print("\n" + "#" * 60)
    print("# Ensemble Forecasting Complete!")
    print("#" * 60)

    return ensemble, results


if __name__ == "__main__":
    run_ensemble_forecast()
