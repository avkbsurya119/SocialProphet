"""
SARIMA Forecaster for SocialProphet.

Seasonal ARIMA implementation with auto_arima for parameter selection.
Uses pmdarima for automatic order selection.

SARIMA excels at:
- Capturing autoregressive patterns
- Seasonal patterns (weekly in our case)
- Statistical rigor with confidence intervals

Weight in ensemble: 35%
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from datetime import datetime

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class SARIMAForecaster:
    """
    SARIMA-based time series forecaster.

    Handles log-transformed engagement data with automatic order selection.
    """

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        auto_select: bool = True
    ):
        """
        Initialize SARIMA forecaster.

        Args:
            order: (p, d, q) - AR, differencing, MA orders
            seasonal_order: (P, D, Q, m) - Seasonal orders, m=7 for weekly
            auto_select: Use auto_arima to select optimal parameters
        """
        default_params = Config.SARIMA_PARAMS
        self.order = order if order is not None else default_params.get('order', (1, 1, 1))
        self.seasonal_order = seasonal_order if seasonal_order is not None else default_params.get('seasonal_order', (1, 1, 1, 7))
        self.auto_select = auto_select

        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.selected_order = None
        self.selected_seasonal_order = None
        self.train_series = None
        self.train_index = None

    def auto_select_order(
        self,
        series: pd.Series,
        seasonal: bool = True,
        m: int = 7
    ) -> Tuple[Tuple, Tuple]:
        """
        Use auto_arima to find optimal (p,d,q) and (P,D,Q,m).

        Args:
            series: Time series data (log scale)
            seasonal: Include seasonal component
            m: Seasonal period (7 for weekly)

        Returns:
            (order, seasonal_order)
        """
        print("Running auto_arima for optimal parameter selection...")
        print("  This may take a moment...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            auto_model = auto_arima(
                series,
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                d=None,  # auto-detect differencing
                seasonal=seasonal,
                m=m,
                start_P=0, max_P=2,
                start_Q=0, max_Q=2,
                D=None,  # auto-detect seasonal differencing
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                information_criterion='aic'
            )

        self.selected_order = auto_model.order
        self.selected_seasonal_order = auto_model.seasonal_order

        print(f"  Selected order: {self.selected_order}")
        print(f"  Selected seasonal order: {self.selected_seasonal_order}")
        print(f"  AIC: {auto_model.aic():.2f}")

        return self.selected_order, self.selected_seasonal_order

    def fit(
        self,
        series: pd.Series,
        exog: Optional[np.ndarray] = None
    ) -> "SARIMAForecaster":
        """
        Train SARIMA model.

        Args:
            series: Log-transformed time series
            exog: Exogenous variables (optional)

        Returns:
            self
        """
        # Store training data
        self.train_series = series.copy()
        self.train_index = series.index if hasattr(series, 'index') else None

        print(f"Training SARIMA on {len(series)} observations...")
        print(f"  y (log) range: [{series.min():.2f}, {series.max():.2f}]")

        # Auto-select parameters if enabled
        if self.auto_select:
            self.auto_select_order(series)
            order = self.selected_order
            seasonal_order = self.selected_seasonal_order
        else:
            order = self.order
            seasonal_order = self.seasonal_order
            self.selected_order = order
            self.selected_seasonal_order = seasonal_order

        print(f"  Using order: {order}")
        print(f"  Using seasonal order: {seasonal_order}")

        # Fit SARIMAX model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                exog=exog,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            self.fitted_model = self.model.fit(disp=False)

        self.is_fitted = True
        print("SARIMA model fitted successfully!")

        return self

    def predict(
        self,
        steps: int = 30,
        exog: Optional[np.ndarray] = None,
        return_conf_int: bool = True,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate forecasts.

        Args:
            steps: Number of periods ahead
            exog: Future exogenous variables
            return_conf_int: Include confidence intervals
            alpha: Significance level for CI (0.05 = 95% CI)

        Returns:
            DataFrame with:
            - yhat: prediction (log scale)
            - yhat_lower, yhat_upper: CI bounds (log scale)
            - yhat_original: prediction (original scale)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get forecast
        forecast = self.fitted_model.get_forecast(steps=steps, exog=exog)
        predictions = forecast.predicted_mean

        # Build result DataFrame
        result = pd.DataFrame({
            'yhat': predictions.values
        })

        if return_conf_int:
            conf_int = forecast.conf_int(alpha=alpha)
            result['yhat_lower'] = conf_int.iloc[:, 0].values
            result['yhat_upper'] = conf_int.iloc[:, 1].values

        # Add original scale predictions
        result['yhat_original'] = self.inverse_transform(result['yhat'].values)
        if return_conf_int:
            result['yhat_lower_original'] = self.inverse_transform(result['yhat_lower'].values)
            result['yhat_upper_original'] = self.inverse_transform(result['yhat_upper'].values)

        return result

    def predict_in_sample(self) -> pd.DataFrame:
        """
        Get in-sample (fitted) values.

        Returns:
            DataFrame with fitted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        fitted_values = self.fitted_model.fittedvalues

        result = pd.DataFrame({
            'yhat': fitted_values.values,
            'yhat_original': self.inverse_transform(fitted_values.values)
        })

        return result

    def predict_test(
        self,
        n_test: int,
        y_test: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Generate predictions for test period.

        Args:
            n_test: Number of test observations
            y_test: Actual test values (optional, for comparison)

        Returns:
            DataFrame with predictions
        """
        result = self.predict(steps=n_test)

        if y_test is not None:
            result['y_actual'] = y_test
            result['y_actual_original'] = self.inverse_transform(y_test)

        return result

    def inverse_transform(self, y_log: np.ndarray) -> np.ndarray:
        """Convert log-scale to original scale."""
        return np.expm1(y_log)

    def evaluate(
        self,
        y_test: np.ndarray,
        y_test_raw: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Evaluate model on test set.

        Args:
            y_test: Test values (log scale)
            y_test_raw: Test values (original scale, optional)

        Returns:
            dict with evaluation metrics
        """
        predictions = self.predict(steps=len(y_test))
        y_pred_log = predictions['yhat'].values

        # Convert to original scale for metrics
        y_true = self.inverse_transform(y_test)
        y_pred = predictions['yhat_original'].values

        # If raw values provided, use those instead
        if y_test_raw is not None:
            y_true = y_test_raw

        # Calculate metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        # R² calculation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        results = {
            'model': 'SARIMA',
            'order': self.selected_order,
            'seasonal_order': self.selected_seasonal_order,
            'n_test': len(y_test),
            'metrics_original_scale': {
                'mape': float(mape),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
            },
            'predictions': predictions
        }

        print(f"\nSARIMA Evaluation Results (Original Scale):")
        print(f"  Order: {self.selected_order}, Seasonal: {self.selected_seasonal_order}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE: {mae:,.2f}")
        print(f"  R²: {r2:.4f}")

        return results

    def get_diagnostics(self) -> Dict:
        """
        Return model diagnostics.

        Returns:
            dict with AIC, BIC, Ljung-Box test results
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        diagnostics = {
            'aic': float(self.fitted_model.aic),
            'bic': float(self.fitted_model.bic),
            'hqic': float(self.fitted_model.hqic),
            'order': self.selected_order,
            'seasonal_order': self.selected_seasonal_order,
        }

        # Ljung-Box test on residuals
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            residuals = self.fitted_model.resid
            lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
            diagnostics['ljung_box'] = {
                'statistic': float(lb_result['lb_stat'].values[0]),
                'p_value': float(lb_result['lb_pvalue'].values[0]),
                'significant_autocorrelation': lb_result['lb_pvalue'].values[0] < 0.05
            }
        except Exception as e:
            diagnostics['ljung_box'] = {'error': str(e)}

        return diagnostics

    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot residual diagnostics (ACF, PACF, QQ, histogram)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        import matplotlib.pyplot as plt

        fig = self.fitted_model.plot_diagnostics(figsize=figsize)
        plt.tight_layout()
        return fig

    def summary(self) -> str:
        """Get model summary."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.fitted_model.summary()

    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save fitted model.

        Args:
            filepath: Output path

        Returns:
            Path to saved file
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if filepath is None:
            Config.ensure_directories()
            filepath = Config.PREDICTIONS_DIR / "sarima_model.pkl"

        with open(filepath, 'wb') as f:
            pickle.dump({
                'fitted_model': self.fitted_model,
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'selected_order': self.selected_order,
                'selected_seasonal_order': self.selected_seasonal_order,
                'auto_select': self.auto_select,
                'train_series': self.train_series,
                'is_fitted': self.is_fitted
            }, f)

        print(f"SARIMA model saved: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "SARIMAForecaster":
        """Load fitted model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        forecaster = cls(
            order=data['order'],
            seasonal_order=data['seasonal_order'],
            auto_select=data['auto_select']
        )
        forecaster.fitted_model = data['fitted_model']
        forecaster.selected_order = data['selected_order']
        forecaster.selected_seasonal_order = data['selected_seasonal_order']
        forecaster.train_series = data['train_series']
        forecaster.is_fitted = data['is_fitted']

        print(f"SARIMA model loaded from: {filepath}")
        return forecaster


def run_sarima_forecast():
    """Run SARIMA forecasting on training data."""
    print("\n" + "#" * 60)
    print("# SocialProphet - SARIMA Forecasting")
    print("#" * 60)

    # Load data
    train_path = Config.PROCESSED_DATA_DIR / "train_data.csv"
    test_path = Config.PROCESSED_DATA_DIR / "test_data.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"\nTrain: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")

    # Get target series
    y_train = train_df['y'].values
    y_test = test_df['y'].values
    y_test_raw = test_df['y_raw'].values if 'y_raw' in test_df.columns else None

    # Initialize and fit
    forecaster = SARIMAForecaster(auto_select=True)
    forecaster.fit(pd.Series(y_train))

    # Print diagnostics
    print("\nModel Diagnostics:")
    diagnostics = forecaster.get_diagnostics()
    print(f"  AIC: {diagnostics['aic']:.2f}")
    print(f"  BIC: {diagnostics['bic']:.2f}")

    # Evaluate on test set
    results = forecaster.evaluate(y_test, y_test_raw)

    # Save model
    forecaster.save()

    print("\n" + "#" * 60)
    print("# SARIMA Forecasting Complete!")
    print("#" * 60)

    return forecaster, results


if __name__ == "__main__":
    run_sarima_forecast()
