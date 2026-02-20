"""
Prophet Forecaster for SocialProphet.

Facebook Prophet implementation for time series forecasting.
Handles log-transformed data and provides inverse-transformed predictions.

Prophet excels at:
- Capturing trend changes
- Weekly/yearly seasonality
- Holiday effects
- Handling missing data

Weight in ensemble: 40%
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class ProphetForecaster:
    """
    Prophet-based time series forecaster.

    Handles log-transformed engagement data:
    - Input: y in log scale (log1p transformed)
    - Output: Predictions in both log and original scale
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize Prophet forecaster.

        Args:
            params: Prophet parameters (defaults from Config.PROPHET_PARAMS)
                - daily_seasonality: True
                - weekly_seasonality: True
                - yearly_seasonality: True
                - changepoint_prior_scale: 0.05
                - seasonality_prior_scale: 10.0
        """
        self.params = params if params is not None else Config.PROPHET_PARAMS.copy()
        self.model = None
        self.is_fitted = False
        self.train_df = None
        self.forecast = None

    def fit(self, df: pd.DataFrame) -> "ProphetForecaster":
        """
        Train Prophet model.

        Args:
            df: DataFrame with columns ['ds', 'y']
                - ds: datetime column
                - y: log-transformed target

        Returns:
            self
        """
        # Validate input
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must have 'ds' and 'y' columns")

        # Prepare data
        self.train_df = df[['ds', 'y']].copy()
        self.train_df['ds'] = pd.to_datetime(self.train_df['ds'])

        print(f"Training Prophet on {len(self.train_df)} observations...")
        print(f"  Date range: {self.train_df['ds'].min()} to {self.train_df['ds'].max()}")
        print(f"  y (log) range: [{self.train_df['y'].min():.2f}, {self.train_df['y'].max():.2f}]")

        # Initialize Prophet with parameters
        self.model = Prophet(
            daily_seasonality=self.params.get('daily_seasonality', True),
            weekly_seasonality=self.params.get('weekly_seasonality', True),
            yearly_seasonality=self.params.get('yearly_seasonality', False),  # Not enough data
            changepoint_prior_scale=self.params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=self.params.get('seasonality_prior_scale', 10.0),
        )

        # Fit model (suppress Stan output)
        import logging
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

        self.model.fit(self.train_df)
        self.is_fitted = True

        print("Prophet model fitted successfully!")
        return self

    def predict(
        self,
        periods: int = 30,
        include_history: bool = True,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        Generate forecasts.

        Args:
            periods: Number of future periods to forecast
            include_history: Include historical predictions
            freq: Frequency for future dates

        Returns:
            DataFrame with columns:
            - ds: date
            - yhat: prediction (log scale)
            - yhat_lower, yhat_upper: confidence interval (log scale)
            - yhat_original: prediction (original scale, via expm1)
            - yhat_lower_original, yhat_upper_original: CI (original scale)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history
        )

        # Generate predictions
        self.forecast = self.model.predict(future)

        # Add original scale predictions
        result = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        result['yhat_original'] = self.inverse_transform(result['yhat'].values)
        result['yhat_lower_original'] = self.inverse_transform(result['yhat_lower'].values)
        result['yhat_upper_original'] = self.inverse_transform(result['yhat_upper'].values)

        return result

    def predict_test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for test set dates.

        Args:
            test_df: Test DataFrame with 'ds' column

        Returns:
            DataFrame with predictions aligned to test dates
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare test dates
        test_dates = pd.DataFrame({'ds': pd.to_datetime(test_df['ds'])})

        # Generate predictions
        predictions = self.model.predict(test_dates)

        # Build result DataFrame
        result = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        result['yhat_original'] = self.inverse_transform(result['yhat'].values)
        result['yhat_lower_original'] = self.inverse_transform(result['yhat_lower'].values)
        result['yhat_upper_original'] = self.inverse_transform(result['yhat_upper'].values)

        # Add actual values if present
        if 'y' in test_df.columns:
            result['y_actual'] = test_df['y'].values
            result['y_actual_original'] = self.inverse_transform(test_df['y'].values)

        if 'y_raw' in test_df.columns:
            result['y_raw'] = test_df['y_raw'].values

        return result

    def inverse_transform(self, y_log: np.ndarray) -> np.ndarray:
        """
        Convert log-scale predictions to original scale.

        Args:
            y_log: Log-transformed values (from log1p)

        Returns:
            Original scale values via np.expm1()
        """
        return np.expm1(y_log)

    def get_components(self) -> pd.DataFrame:
        """
        Extract seasonality components (trend, weekly, yearly).

        Returns:
            DataFrame with component values
        """
        if self.forecast is None:
            raise ValueError("No forecast available. Call predict() first.")

        components = ['ds', 'trend']

        if 'weekly' in self.forecast.columns:
            components.append('weekly')
        if 'yearly' in self.forecast.columns:
            components.append('yearly')
        if 'daily' in self.forecast.columns:
            components.append('daily')

        return self.forecast[components].copy()

    def cross_validate(
        self,
        initial: str = '180 days',
        period: str = '30 days',
        horizon: str = '30 days'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform time series cross-validation.

        Args:
            initial: Initial training period
            period: Spacing between cutoff dates
            horizon: Forecast horizon

        Returns:
            Tuple of (cv_results, performance_metrics)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        print(f"Running cross-validation (initial={initial}, horizon={horizon})...")

        cv_results = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )

        metrics = performance_metrics(cv_results)

        print("Cross-validation complete!")
        print(f"  MAPE: {metrics['mape'].mean() * 100:.2f}%")
        print(f"  RMSE: {metrics['rmse'].mean():.4f}")

        return cv_results, metrics

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate model on test set.

        Args:
            test_df: Test DataFrame with 'ds' and 'y' columns

        Returns:
            dict with evaluation metrics (on original scale)
        """
        predictions = self.predict_test(test_df)

        y_true_log = test_df['y'].values
        y_pred_log = predictions['yhat'].values

        # Convert to original scale for metrics
        y_true = self.inverse_transform(y_true_log)
        y_pred = self.inverse_transform(y_pred_log)

        # Calculate metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        # R² calculation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        results = {
            'model': 'Prophet',
            'n_test': len(test_df),
            'metrics_original_scale': {
                'mape': float(mape),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
            },
            'predictions': predictions
        }

        print(f"\nProphet Evaluation Results (Original Scale):")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE: {mae:,.2f}")
        print(f"  R²: {r2:.4f}")

        return results

    def plot_forecast(self, figsize: Tuple[int, int] = (12, 6)):
        """Plot forecast with components."""
        if self.forecast is None:
            raise ValueError("No forecast available. Call predict() first.")

        fig1 = self.model.plot(self.forecast, figsize=figsize)
        fig2 = self.model.plot_components(self.forecast, figsize=figsize)

        return fig1, fig2

    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save fitted model to pickle.

        Args:
            filepath: Output path (default: data/predictions/prophet_model.pkl)

        Returns:
            Path to saved file
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if filepath is None:
            Config.ensure_directories()
            filepath = Config.PREDICTIONS_DIR / "prophet_model.pkl"

        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'params': self.params,
                'train_df': self.train_df,
                'is_fitted': self.is_fitted
            }, f)

        print(f"Prophet model saved: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "ProphetForecaster":
        """
        Load fitted model from pickle.

        Args:
            filepath: Path to saved model

        Returns:
            ProphetForecaster instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        forecaster = cls(params=data['params'])
        forecaster.model = data['model']
        forecaster.train_df = data['train_df']
        forecaster.is_fitted = data['is_fitted']

        print(f"Prophet model loaded from: {filepath}")
        return forecaster


def run_prophet_forecast():
    """Run Prophet forecasting on training data."""
    print("\n" + "#" * 60)
    print("# SocialProphet - Prophet Forecasting")
    print("#" * 60)

    # Load data
    train_path = Config.PROCESSED_DATA_DIR / "train_prophet.csv"
    test_path = Config.PROCESSED_DATA_DIR / "test_prophet.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"\nTrain: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")

    # Initialize and fit
    forecaster = ProphetForecaster()
    forecaster.fit(train_df)

    # Evaluate on test set
    results = forecaster.evaluate(test_df)

    # Save model
    forecaster.save()

    print("\n" + "#" * 60)
    print("# Prophet Forecasting Complete!")
    print("#" * 60)

    return forecaster, results


if __name__ == "__main__":
    run_prophet_forecast()
