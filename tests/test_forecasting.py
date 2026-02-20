"""
Unit tests for SocialProphet forecasting modules.

Tests cover:
- Stationarity analysis (ADF, KPSS)
- Prophet forecaster
- SARIMA forecaster
- LSTM forecaster
- Ensemble forecaster
- Evaluation metrics

Target: >80% coverage
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import warnings

# Suppress TensorFlow warnings for cleaner test output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forecasting.stationarity import StationarityAnalyzer
from src.forecasting.prophet_model import ProphetForecaster
from src.forecasting.sarima_model import SARIMAForecaster
from src.forecasting.lstm_model import LSTMForecaster
from src.forecasting.ensemble import EnsembleForecaster
from src.evaluation.metrics import ForecastMetrics


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_series():
    """Generate sample time series for testing."""
    np.random.seed(42)
    n = 100
    trend = np.linspace(10, 10.5, n)
    seasonality = 0.2 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 0.1, n)
    return pd.Series(trend + seasonality + noise)


@pytest.fixture
def sample_train_df():
    """Generate sample training DataFrame."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

    trend = np.linspace(10, 10.5, n)
    seasonality = 0.2 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 0.1, n)
    y = trend + seasonality + noise

    df = pd.DataFrame({
        'ds': dates,
        'y': y,
        'y_raw': np.expm1(y),
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
    })

    # Add lag features
    df['y_lag_1'] = df['y'].shift(1).fillna(df['y'].mean())
    df['y_lag_7'] = df['y'].shift(7).fillna(df['y'].mean())

    # Add rolling features
    df['y_rolling_mean_7'] = df['y'].rolling(7, min_periods=1).mean()
    df['y_rolling_std_7'] = df['y'].rolling(7, min_periods=1).std().fillna(0)

    return df


@pytest.fixture
def sample_test_df(sample_train_df):
    """Generate sample test DataFrame."""
    np.random.seed(43)
    n = 30
    last_date = sample_train_df['ds'].max()
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=n, freq='D')

    trend = np.linspace(10.5, 10.7, n)
    seasonality = 0.2 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 0.1, n)
    y = trend + seasonality + noise

    df = pd.DataFrame({
        'ds': dates,
        'y': y,
        'y_raw': np.expm1(y),
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
    })

    # Add lag features (using last values from train)
    df['y_lag_1'] = df['y'].shift(1).fillna(sample_train_df['y'].iloc[-1])
    df['y_lag_7'] = df['y'].shift(7).fillna(sample_train_df['y'].iloc[-7:].mean())

    # Add rolling features
    df['y_rolling_mean_7'] = df['y'].rolling(7, min_periods=1).mean()
    df['y_rolling_std_7'] = df['y'].rolling(7, min_periods=1).std().fillna(0)

    return df


@pytest.fixture
def sample_prophet_df():
    """Generate sample Prophet-format DataFrame."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

    trend = np.linspace(10, 10.5, n)
    seasonality = 0.2 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 0.1, n)
    y = trend + seasonality + noise

    return pd.DataFrame({'ds': dates, 'y': y})


# =============================================================================
# Test Stationarity Analyzer
# =============================================================================

class TestStationarityAnalyzer:
    """Tests for StationarityAnalyzer class."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = StationarityAnalyzer()
        assert analyzer.results is None

    def test_adf_test(self, sample_series):
        """Test ADF test execution."""
        analyzer = StationarityAnalyzer()
        result = analyzer.adf_test(sample_series)

        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'critical_values' in result
        assert 'is_stationary' in result
        assert isinstance(result['is_stationary'], bool)

    def test_kpss_test(self, sample_series):
        """Test KPSS test execution."""
        analyzer = StationarityAnalyzer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyzer.kpss_test(sample_series)

        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'critical_values' in result
        assert 'is_stationary' in result

    def test_analyze(self, sample_series):
        """Test full analysis."""
        analyzer = StationarityAnalyzer()
        result = analyzer.analyze(sample_series, name="test_series")

        assert 'series_name' in result
        assert 'n_observations' in result
        assert 'adf_test' in result
        assert 'kpss_test' in result
        assert 'recommendation' in result
        assert result['series_name'] == "test_series"
        assert result['n_observations'] == len(sample_series)

    def test_differencing_analysis(self, sample_series):
        """Test differencing analysis."""
        analyzer = StationarityAnalyzer()
        result = analyzer.differencing_analysis(sample_series, max_d=2)

        assert 'd=0' in result
        assert 'd=1' in result
        assert 'recommended_d' in result

    def test_save_report(self, sample_series):
        """Test report saving."""
        analyzer = StationarityAnalyzer()
        analyzer.analyze(sample_series)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "stationarity_report.json"
            saved_path = analyzer.save_report(filepath)

            assert saved_path.exists()
            assert saved_path.suffix == '.json'


# =============================================================================
# Test Prophet Forecaster
# =============================================================================

class TestProphetForecaster:
    """Tests for ProphetForecaster class."""

    def test_init(self):
        """Test forecaster initialization."""
        forecaster = ProphetForecaster()
        assert forecaster.model is None
        assert forecaster.is_fitted == False

    def test_fit(self, sample_prophet_df):
        """Test model fitting."""
        forecaster = ProphetForecaster()
        forecaster.fit(sample_prophet_df)

        assert forecaster.is_fitted == True
        assert forecaster.model is not None

    def test_predict(self, sample_prophet_df):
        """Test prediction."""
        forecaster = ProphetForecaster()
        forecaster.fit(sample_prophet_df)

        predictions = forecaster.predict(periods=10)

        assert len(predictions) > 0
        assert 'ds' in predictions.columns
        assert 'yhat' in predictions.columns

    def test_predict_test(self, sample_prophet_df):
        """Test prediction on test dates."""
        forecaster = ProphetForecaster()
        forecaster.fit(sample_prophet_df)

        # Create test dates
        test_dates = pd.date_range(
            start=sample_prophet_df['ds'].max() + timedelta(days=1),
            periods=10,
            freq='D'
        )
        test_df = pd.DataFrame({'ds': test_dates, 'y': np.random.randn(10) + 10})

        predictions = forecaster.predict_test(test_df)

        assert len(predictions) == len(test_df)
        assert 'yhat' in predictions.columns

    def test_inverse_transform(self):
        """Test inverse transform."""
        forecaster = ProphetForecaster()
        y_log = np.array([10.0, 10.5, 11.0])
        y_original = forecaster.inverse_transform(y_log)

        expected = np.expm1(y_log)
        np.testing.assert_array_almost_equal(y_original, expected)

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        forecaster = ProphetForecaster()

        with pytest.raises(ValueError, match="not fitted"):
            forecaster.predict()


# =============================================================================
# Test SARIMA Forecaster
# =============================================================================

class TestSARIMAForecaster:
    """Tests for SARIMAForecaster class."""

    def test_init(self):
        """Test forecaster initialization."""
        forecaster = SARIMAForecaster()
        assert forecaster.model is None
        assert forecaster.is_fitted == False

    def test_fit(self, sample_series):
        """Test model fitting."""
        forecaster = SARIMAForecaster(order=(1, 0, 1))
        forecaster.fit(sample_series)

        assert forecaster.is_fitted == True
        assert forecaster.model is not None

    def test_predict(self, sample_series):
        """Test prediction."""
        forecaster = SARIMAForecaster(order=(1, 0, 1))
        forecaster.fit(sample_series)

        predictions = forecaster.predict(steps=10)

        assert len(predictions) == 10
        assert 'forecast' in predictions.columns

    def test_predict_with_confidence(self, sample_series):
        """Test prediction with confidence intervals."""
        forecaster = SARIMAForecaster(order=(1, 0, 1))
        forecaster.fit(sample_series)

        predictions = forecaster.predict(steps=10, return_conf_int=True)

        assert 'lower' in predictions.columns
        assert 'upper' in predictions.columns

    def test_get_diagnostics(self, sample_series):
        """Test model diagnostics."""
        forecaster = SARIMAForecaster(order=(1, 0, 1))
        forecaster.fit(sample_series)

        diagnostics = forecaster.get_diagnostics()

        assert 'aic' in diagnostics
        assert 'bic' in diagnostics
        assert 'order' in diagnostics

    def test_auto_select_order(self, sample_series):
        """Test automatic order selection."""
        forecaster = SARIMAForecaster()
        order, seasonal_order = forecaster.auto_select_order(
            sample_series,
            seasonal=False,
            max_p=2,
            max_q=2
        )

        assert len(order) == 3
        assert all(isinstance(x, (int, np.integer)) for x in order)

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        forecaster = SARIMAForecaster()

        with pytest.raises(ValueError, match="not fitted"):
            forecaster.predict()


# =============================================================================
# Test LSTM Forecaster
# =============================================================================

class TestLSTMForecaster:
    """Tests for LSTMForecaster class."""

    def test_init(self):
        """Test forecaster initialization."""
        forecaster = LSTMForecaster()
        assert forecaster.model is None
        assert forecaster.is_fitted == False

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        forecaster = LSTMForecaster(
            n_units=32,
            n_layers=1,
            window_size=14
        )
        assert forecaster.params['n_units'] == 32
        assert forecaster.params['n_layers'] == 1
        assert forecaster.params['window_size'] == 14

    def test_create_sequences(self, sample_train_df):
        """Test sequence creation."""
        forecaster = LSTMForecaster(window_size=10)

        data = sample_train_df[['y', 'y_lag_1']].values
        X, y = forecaster._create_sequences(data, window_size=10)

        assert X.shape[0] == len(data) - 10
        assert X.shape[1] == 10
        assert X.shape[2] == 2
        assert len(y) == len(data) - 10

    def test_build_model(self):
        """Test model building."""
        forecaster = LSTMForecaster(n_units=16, n_layers=2)
        model = forecaster._build_model(input_shape=(10, 3))

        assert model is not None
        assert len(model.layers) > 0

    def test_fit(self, sample_train_df):
        """Test model fitting (minimal epochs for speed)."""
        forecaster = LSTMForecaster(
            n_units=8,
            n_layers=1,
            window_size=10,
            epochs=2,
            batch_size=16
        )

        feature_cols = ['y_lag_1', 'y_rolling_mean_7']
        forecaster.fit(sample_train_df, feature_cols=feature_cols, verbose=0)

        assert forecaster.is_fitted == True
        assert forecaster.model is not None

    def test_predict_test(self, sample_train_df, sample_test_df):
        """Test prediction on test data."""
        forecaster = LSTMForecaster(
            n_units=8,
            n_layers=1,
            window_size=10,
            epochs=2,
            batch_size=16
        )

        feature_cols = ['y_lag_1', 'y_rolling_mean_7']
        forecaster.fit(sample_train_df, feature_cols=feature_cols, verbose=0)

        predictions = forecaster.predict_test(sample_train_df, sample_test_df)

        assert len(predictions) == len(sample_test_df)
        assert 'yhat' in predictions.columns


# =============================================================================
# Test Ensemble Forecaster
# =============================================================================

class TestEnsembleForecaster:
    """Tests for EnsembleForecaster class."""

    def test_init(self):
        """Test forecaster initialization."""
        forecaster = EnsembleForecaster()
        assert forecaster.models == {}
        assert 'prophet' in forecaster.weights

    def test_init_with_weights(self):
        """Test initialization with custom weights."""
        weights = {'prophet': 0.5, 'sarima': 0.3, 'lstm': 0.2}
        forecaster = EnsembleForecaster(weights=weights)

        assert forecaster.weights['prophet'] == 0.5

    def test_add_model(self):
        """Test adding a model."""
        forecaster = EnsembleForecaster()
        mock_model = ProphetForecaster()

        forecaster.add_model('prophet', mock_model, weight=0.5)

        assert 'prophet' in forecaster.models
        assert forecaster.weights['prophet'] == 0.5

    def test_weighted_average(self):
        """Test weighted averaging."""
        forecaster = EnsembleForecaster(
            weights={'a': 0.6, 'b': 0.4}
        )

        predictions = {
            'a': np.array([10.0, 11.0, 12.0]),
            'b': np.array([9.0, 10.0, 11.0])
        }

        result = forecaster.weighted_average(predictions)

        expected = 0.6 * np.array([10.0, 11.0, 12.0]) + 0.4 * np.array([9.0, 10.0, 11.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_weights(self):
        """Test weight normalization."""
        forecaster = EnsembleForecaster(
            weights={'a': 2.0, 'b': 3.0}
        )

        predictions = {
            'a': np.array([10.0]),
            'b': np.array([20.0])
        }

        # Weights should be normalized to sum to 1
        result = forecaster.weighted_average(predictions)

        # (2/5)*10 + (3/5)*20 = 4 + 12 = 16
        expected = np.array([16.0])
        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# Test Forecast Metrics
# =============================================================================

class TestForecastMetrics:
    """Tests for ForecastMetrics class."""

    def test_init(self):
        """Test metrics initialization."""
        metrics = ForecastMetrics()
        assert metrics.results == {}
        assert 'mape' in metrics.THRESHOLDS

    def test_inverse_transform_log(self):
        """Test inverse log transform."""
        y_log = np.array([10.0, 10.5, 11.0])
        y_original = ForecastMetrics.inverse_transform_log(y_log)

        expected = np.expm1(y_log)
        np.testing.assert_array_almost_equal(y_original, expected)

    def test_mape(self):
        """Test MAPE calculation."""
        metrics = ForecastMetrics()
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 320.0])

        mape = metrics.mape(y_true, y_pred)

        # |100-110|/100 + |200-190|/200 + |300-320|/300 = 0.1 + 0.05 + 0.0667 / 3 * 100
        expected = (10/100 + 10/200 + 20/300) / 3 * 100
        assert abs(mape - expected) < 0.01

    def test_mape_zero_handling(self):
        """Test MAPE with zero values."""
        metrics = ForecastMetrics()
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 110.0, 190.0])

        # Should exclude zero values
        mape = metrics.mape(y_true, y_pred)
        assert not np.isinf(mape)

    def test_rmse(self):
        """Test RMSE calculation."""
        metrics = ForecastMetrics()
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])

        rmse = metrics.rmse(y_true, y_pred)

        # sqrt((10^2 + 10^2 + 10^2) / 3) = sqrt(100) = 10
        assert abs(rmse - 10.0) < 0.01

    def test_r2(self):
        """Test R-squared calculation."""
        metrics = ForecastMetrics()
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([100.0, 200.0, 300.0, 400.0])  # Perfect predictions

        r2 = metrics.r2(y_true, y_pred)

        assert abs(r2 - 1.0) < 0.01

    def test_smape(self):
        """Test SMAPE calculation."""
        metrics = ForecastMetrics()
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])

        smape = metrics.smape(y_true, y_pred)

        # Should be between 0 and 200
        assert 0 <= smape <= 200

    def test_evaluate(self):
        """Test full evaluation."""
        metrics = ForecastMetrics()

        # Create log-scale values
        y_true_log = np.array([10.0, 10.2, 10.4, 10.6, 10.8])
        y_pred_log = np.array([10.1, 10.15, 10.45, 10.55, 10.85])

        results = metrics.evaluate(y_true_log, y_pred_log)

        assert 'n_samples' in results
        assert 'metrics_original_scale' in results
        assert 'pass_fail' in results
        assert 'all_passed' in results
        assert results['n_samples'] == 5

    def test_evaluate_original_scale(self):
        """Test evaluation with original scale values."""
        metrics = ForecastMetrics()

        y_true = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        y_pred = np.array([1100.0, 1900.0, 3100.0, 3900.0, 5100.0])

        results = metrics.evaluate_original_scale(y_true, y_pred)

        assert 'metrics_original_scale' in results
        assert results['metrics_original_scale']['mape'] < 15  # Should pass

    def test_compare_models(self):
        """Test model comparison."""
        metrics = ForecastMetrics()

        y_true_log = np.array([10.0, 10.2, 10.4, 10.6, 10.8])
        predictions = {
            'model_a': np.array([10.1, 10.15, 10.45, 10.55, 10.85]),
            'model_b': np.array([9.9, 10.25, 10.35, 10.65, 10.75])
        }

        comparison = metrics.compare_models(y_true_log, predictions)

        assert len(comparison) == 2
        assert 'model' in comparison.columns
        assert 'mape' in comparison.columns
        assert 'all_pass' in comparison.columns

    def test_generate_report(self):
        """Test report generation."""
        metrics = ForecastMetrics()

        y_true_log = np.array([10.0, 10.2, 10.4, 10.6, 10.8])
        y_pred_log = np.array([10.1, 10.15, 10.45, 10.55, 10.85])

        metrics.evaluate(y_true_log, y_pred_log)
        report = metrics.generate_report()

        assert "FORECAST EVALUATION REPORT" in report
        assert "MAPE" in report
        assert "RMSE" in report
        assert "R²" in report

    def test_save_results(self):
        """Test results saving."""
        metrics = ForecastMetrics()

        y_true_log = np.array([10.0, 10.2, 10.4])
        y_pred_log = np.array([10.1, 10.15, 10.45])

        metrics.evaluate(y_true_log, y_pred_log)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.json"
            saved_path = metrics.save_results(filepath)

            assert saved_path.exists()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the forecasting pipeline."""

    def test_prophet_to_metrics(self, sample_prophet_df):
        """Test Prophet model through metrics evaluation."""
        # Train Prophet
        forecaster = ProphetForecaster()
        forecaster.fit(sample_prophet_df)

        # Create test data
        test_dates = pd.date_range(
            start=sample_prophet_df['ds'].max() + timedelta(days=1),
            periods=10,
            freq='D'
        )
        test_df = pd.DataFrame({
            'ds': test_dates,
            'y': np.random.randn(10) * 0.1 + 10.3
        })

        # Predict
        predictions = forecaster.predict_test(test_df)

        # Evaluate
        metrics = ForecastMetrics()
        results = metrics.evaluate(
            test_df['y'].values,
            predictions['yhat'].values
        )

        assert 'metrics_original_scale' in results
        assert 'pass_fail' in results

    def test_stationarity_before_sarima(self, sample_series):
        """Test stationarity analysis before SARIMA fitting."""
        # Analyze stationarity
        analyzer = StationarityAnalyzer()
        stationarity_result = analyzer.analyze(sample_series)

        # Fit SARIMA based on recommendation
        forecaster = SARIMAForecaster(order=(1, 0, 1))
        forecaster.fit(sample_series)

        # Make predictions
        predictions = forecaster.predict(steps=10)

        assert stationarity_result is not None
        assert len(predictions) == 10


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
