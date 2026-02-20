"""
Forecast Evaluation Metrics for SocialProphet.

CRITICAL: All metrics are computed on ORIGINAL SCALE (after inverse transform).
Never compute MAPE/RMSE on log-scale values!

Target Thresholds (for excellent grade):
- MAPE < 15%
- RMSE < 15% of mean
- R² > 0.70
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class ForecastMetrics:
    """
    Comprehensive forecasting metrics calculator.

    All metrics are computed on ORIGINAL scale values.
    Input log-scale values are automatically inverse-transformed.
    """

    # Target thresholds from requirements
    THRESHOLDS = {
        'mape': 15.0,       # < 15%
        'rmse_pct': 15.0,   # RMSE < 15% of mean
        'r2': 0.70,         # > 0.70
    }

    def __init__(self):
        """Initialize metrics calculator."""
        self.results = {}

    @staticmethod
    def inverse_transform_log(y_log: np.ndarray) -> np.ndarray:
        """
        Convert log-scale predictions to original scale.

        Args:
            y_log: Log-transformed values (from log1p)

        Returns:
            Original scale values (via expm1)
        """
        return np.expm1(y_log)

    def mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error.

        MAPE = (1/n) * sum(|y_true - y_pred| / y_true) * 100

        Args:
            y_true: Actual values (ORIGINAL SCALE)
            y_pred: Predicted values (ORIGINAL SCALE)

        Returns:
            MAPE percentage
        """
        # Handle zero values
        mask = y_true != 0
        if not mask.any():
            return float('inf')
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error.

        Args:
            y_true: Actual values (ORIGINAL SCALE)
            y_pred: Predicted values (ORIGINAL SCALE)

        Returns:
            RMSE value
        """
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def rmse_percentage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        RMSE as percentage of mean.

        Returns:
            RMSE / mean(y_true) * 100
        """
        rmse_val = self.rmse(y_true, y_pred)
        mean_val = np.mean(y_true)
        if mean_val == 0:
            return float('inf')
        return float((rmse_val / mean_val) * 100)

    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(mean_absolute_error(y_true, y_pred))

    def r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared (coefficient of determination)."""
        return float(r2_score(y_true, y_pred))

    def smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Symmetric Mean Absolute Percentage Error.

        More robust than MAPE for values near zero.
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not mask.any():
            return float('inf')
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(mean_squared_error(y_true, y_pred))

    def evaluate(
        self,
        y_true_log: np.ndarray,
        y_pred_log: np.ndarray,
        include_log_metrics: bool = False
    ) -> Dict:
        """
        Complete evaluation on original scale.

        Args:
            y_true_log: Actual values (LOG SCALE from data)
            y_pred_log: Predicted values (LOG SCALE from model)
            include_log_metrics: Also compute metrics on log scale

        Returns:
            Dict with all metrics and pass/fail status
        """
        # Inverse transform to original scale
        y_true = self.inverse_transform_log(y_true_log)
        y_pred = self.inverse_transform_log(y_pred_log)

        # Clip negative predictions (shouldn't happen but safety)
        y_pred = np.clip(y_pred, 0, None)

        results = {
            'n_samples': len(y_true),
            'y_true_mean': float(np.mean(y_true)),
            'y_true_std': float(np.std(y_true)),
            'y_pred_mean': float(np.mean(y_pred)),
            'y_pred_std': float(np.std(y_pred)),
            'metrics_original_scale': {
                'mape': self.mape(y_true, y_pred),
                'rmse': self.rmse(y_true, y_pred),
                'rmse_pct': self.rmse_percentage(y_true, y_pred),
                'mae': self.mae(y_true, y_pred),
                'mse': self.mse(y_true, y_pred),
                'r2': self.r2(y_true, y_pred),
                'smape': self.smape(y_true, y_pred),
            },
            'thresholds': self.THRESHOLDS.copy(),
            'pass_fail': {
                'mape': self.mape(y_true, y_pred) < self.THRESHOLDS['mape'],
                'rmse_pct': self.rmse_percentage(y_true, y_pred) < self.THRESHOLDS['rmse_pct'],
                'r2': self.r2(y_true, y_pred) > self.THRESHOLDS['r2'],
            }
        }

        results['all_passed'] = all(results['pass_fail'].values())

        if include_log_metrics:
            results['metrics_log_scale'] = {
                'mape': self.mape(y_true_log, y_pred_log),
                'rmse': self.rmse(y_true_log, y_pred_log),
                'mae': self.mae(y_true_log, y_pred_log),
                'r2': self.r2(y_true_log, y_pred_log),
            }

        self.results = results
        return results

    def evaluate_original_scale(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Evaluate with values already in original scale.

        Args:
            y_true: Actual values (ORIGINAL SCALE)
            y_pred: Predicted values (ORIGINAL SCALE)

        Returns:
            Dict with all metrics
        """
        # Clip negative predictions
        y_pred = np.clip(y_pred, 0, None)

        results = {
            'n_samples': len(y_true),
            'y_true_mean': float(np.mean(y_true)),
            'y_pred_mean': float(np.mean(y_pred)),
            'metrics_original_scale': {
                'mape': self.mape(y_true, y_pred),
                'rmse': self.rmse(y_true, y_pred),
                'rmse_pct': self.rmse_percentage(y_true, y_pred),
                'mae': self.mae(y_true, y_pred),
                'r2': self.r2(y_true, y_pred),
                'smape': self.smape(y_true, y_pred),
            },
            'thresholds': self.THRESHOLDS.copy(),
            'pass_fail': {
                'mape': self.mape(y_true, y_pred) < self.THRESHOLDS['mape'],
                'rmse_pct': self.rmse_percentage(y_true, y_pred) < self.THRESHOLDS['rmse_pct'],
                'r2': self.r2(y_true, y_pred) > self.THRESHOLDS['r2'],
            }
        }

        results['all_passed'] = all(results['pass_fail'].values())
        self.results = results
        return results

    def compare_models(
        self,
        y_true_log: np.ndarray,
        predictions: Dict[str, np.ndarray],
        model_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            y_true_log: Actual values (log scale)
            predictions: Dict of model_name -> predictions (log scale)
            model_names: Optional list to order results

        Returns:
            DataFrame with metrics comparison
        """
        if model_names is None:
            model_names = list(predictions.keys())

        y_true = self.inverse_transform_log(y_true_log)

        comparison = []
        for name in model_names:
            if name in predictions:
                y_pred = self.inverse_transform_log(predictions[name])
                y_pred = np.clip(y_pred, 0, None)

                comparison.append({
                    'model': name,
                    'mape': self.mape(y_true, y_pred),
                    'rmse': self.rmse(y_true, y_pred),
                    'rmse_pct': self.rmse_percentage(y_true, y_pred),
                    'mae': self.mae(y_true, y_pred),
                    'r2': self.r2(y_true, y_pred),
                    'smape': self.smape(y_true, y_pred),
                })

        df = pd.DataFrame(comparison)

        # Add pass/fail columns
        df['mape_pass'] = df['mape'] < self.THRESHOLDS['mape']
        df['rmse_pass'] = df['rmse_pct'] < self.THRESHOLDS['rmse_pct']
        df['r2_pass'] = df['r2'] > self.THRESHOLDS['r2']
        df['all_pass'] = df['mape_pass'] & df['rmse_pass'] & df['r2_pass']

        return df

    def generate_report(self) -> str:
        """Generate human-readable evaluation report."""
        if not self.results:
            return "No evaluation results available. Run evaluate() first."

        r = self.results
        m = r['metrics_original_scale']
        p = r['pass_fail']

        report = []
        report.append("=" * 60)
        report.append("FORECAST EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"\nSamples: {r['n_samples']}")
        report.append(f"Actual Mean: {r['y_true_mean']:,.2f}")
        report.append(f"Predicted Mean: {r['y_pred_mean']:,.2f}")

        report.append("\n" + "-" * 60)
        report.append("METRICS (Original Scale)")
        report.append("-" * 60)
        report.append(f"MAPE:    {m['mape']:>8.2f}%  {'[PASS]' if p['mape'] else '[FAIL]'} (target: <{self.THRESHOLDS['mape']}%)")
        report.append(f"RMSE:    {m['rmse']:>8,.2f}")
        report.append(f"RMSE %:  {m['rmse_pct']:>8.2f}%  {'[PASS]' if p['rmse_pct'] else '[FAIL]'} (target: <{self.THRESHOLDS['rmse_pct']}%)")
        report.append(f"MAE:     {m['mae']:>8,.2f}")
        report.append(f"R²:      {m['r2']:>8.4f}   {'[PASS]' if p['r2'] else '[FAIL]'} (target: >{self.THRESHOLDS['r2']})")
        report.append(f"SMAPE:   {m['smape']:>8.2f}%")

        report.append("\n" + "-" * 60)
        report.append(f"OVERALL: {'ALL THRESHOLDS PASSED' if r['all_passed'] else 'SOME THRESHOLDS FAILED'}")
        report.append("=" * 60)

        return "\n".join(report)

    def print_report(self) -> None:
        """Print evaluation report."""
        print(self.generate_report())

    def save_results(self, filepath: Optional[Path] = None) -> Path:
        """
        Save evaluation results to JSON.

        Args:
            filepath: Output path

        Returns:
            Path to saved file
        """
        if not self.results:
            raise ValueError("No results to save. Run evaluate() first.")

        if filepath is None:
            Config.ensure_directories()
            filepath = Config.PROCESSED_DATA_DIR / "evaluation_results.json"

        results_to_save = {
            **self.results,
            'saved_at': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)

        print(f"Evaluation results saved: {filepath}")
        return filepath


def run_evaluation():
    """Run evaluation on saved predictions."""
    print("\n" + "#" * 60)
    print("# SocialProphet - Forecast Evaluation")
    print("#" * 60)

    # Load test data
    test_path = Config.PROCESSED_DATA_DIR / "test_data.csv"
    test_df = pd.read_csv(test_path)

    print(f"\nTest data: {len(test_df)} rows")

    # Load ensemble results
    results_path = Config.PROCESSED_DATA_DIR / "ensemble_results.json"
    if results_path.exists():
        with open(results_path) as f:
            ensemble_results = json.load(f)

        print("\nEnsemble Results:")
        for metric, value in ensemble_results.get('ensemble_metrics', {}).items():
            print(f"  {metric}: {value:.4f}")
    else:
        print("\nNo ensemble results found. Run ensemble forecasting first.")

    print("\n" + "#" * 60)
    print("# Evaluation Complete!")
    print("#" * 60)


if __name__ == "__main__":
    run_evaluation()
