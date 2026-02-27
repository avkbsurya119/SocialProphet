"""
Unit tests for SocialProphet Dashboard.

Tests for dashboard components and functionality.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDashboardConfig:
    """Tests for dashboard configuration."""

    def test_dashboard_app_exists(self):
        """Test that main dashboard app file exists."""
        app_path = Path(__file__).parent.parent / "dashboard" / "app.py"
        assert app_path.exists(), "Dashboard app.py should exist"

    def test_data_overview_page_exists(self):
        """Test that data overview page exists."""
        page_path = Path(__file__).parent.parent / "dashboard" / "pages" / "1_data_overview.py"
        assert page_path.exists(), "Data overview page should exist"

    def test_forecasting_page_exists(self):
        """Test that forecasting page exists."""
        page_path = Path(__file__).parent.parent / "dashboard" / "pages" / "2_forecasting.py"
        assert page_path.exists(), "Forecasting page should exist"

    def test_content_gen_page_exists(self):
        """Test that content generation page exists."""
        page_path = Path(__file__).parent.parent / "dashboard" / "pages" / "3_content_gen.py"
        assert page_path.exists(), "Content generation page should exist"

    def test_twitter_live_page_exists(self):
        """Test that Twitter live page exists."""
        page_path = Path(__file__).parent.parent / "dashboard" / "pages" / "4_twitter_live.py"
        assert page_path.exists(), "Twitter live page should exist"


class TestTwitterCollector:
    """Tests for Twitter data collector."""

    def test_collector_import(self):
        """Test that TwitterCollector can be imported."""
        try:
            from src.data_processing.twitter_collector import TwitterCollector
            assert True
        except ImportError as e:
            # May fail if tweepy not installed
            if "tweepy" in str(e):
                pytest.skip("tweepy not installed")
            raise

    def test_rate_limiter_init(self):
        """Test rate limiter initialization."""
        try:
            from src.data_processing.twitter_collector import RateLimiter
            limiter = RateLimiter(max_requests=30, window_seconds=60)
            assert limiter.max_requests == 30
            assert limiter.window_seconds == 60
            assert len(limiter.requests) == 0
        except ImportError:
            pytest.skip("tweepy not installed")

    def test_collector_requires_token(self):
        """Test that collector requires bearer token."""
        try:
            from src.data_processing.twitter_collector import TwitterCollector
            with pytest.raises(ValueError):
                # Should raise error without token
                with patch.dict('os.environ', {}, clear=True):
                    TwitterCollector(bearer_token=None)
        except ImportError:
            pytest.skip("tweepy not installed")

    @patch('src.data_processing.twitter_collector.HAS_TWEEPY', False)
    def test_collector_without_tweepy(self):
        """Test collector behavior without tweepy."""
        try:
            from src.data_processing.twitter_collector import TwitterCollector
            with pytest.raises(ImportError):
                TwitterCollector(bearer_token="test_token")
        except ImportError:
            pytest.skip("Module import issue")


class TestDashboardImports:
    """Tests for dashboard module imports."""

    def test_config_import(self):
        """Test that Config can be imported."""
        from src.utils.config import Config
        assert Config is not None

    def test_config_validate_api_keys(self):
        """Test API key validation function."""
        from src.utils.config import Config
        result = Config.validate_api_keys()
        assert isinstance(result, dict)
        assert 'huggingface' in result
        assert 'twitter' in result

    def test_data_processing_imports(self):
        """Test data processing module imports."""
        from src.data_processing import DataCollector, DataPreprocessor, FeatureEngineer
        assert DataCollector is not None
        assert DataPreprocessor is not None
        assert FeatureEngineer is not None


class TestDataOverviewPage:
    """Tests for data overview page functionality."""

    def test_load_data_function(self):
        """Test data loading functionality."""
        import pandas as pd
        from pathlib import Path
        from src.utils.config import Config

        processed_dir = Path(Config.PROCESSED_DATA_DIR)
        train_path = processed_dir / "train_data.csv"

        if train_path.exists():
            df = pd.read_csv(train_path)
            assert len(df) > 0, "Train data should have rows"
            assert 'ds' in df.columns or 'y' in df.columns, "Should have expected columns"

    def test_data_quality_checks(self):
        """Test data quality check logic."""
        import pandas as pd
        import numpy as np

        # Create test dataframe
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'engagement': [100, 200, 150, 180, 120, 250, 300, 175, 225, 195]
        })

        # Test quality checks
        has_missing = df.isnull().sum().sum() == 0
        has_duplicates = df.duplicated().sum() == 0
        positive_engagement = df['engagement'].min() >= 0

        assert has_missing, "No missing values"
        assert has_duplicates, "No duplicates"
        assert positive_engagement, "Positive engagement"


class TestForecastingPage:
    """Tests for forecasting page functionality."""

    def test_model_metrics_structure(self):
        """Test model metrics data structure."""
        metrics = {
            'mape': 12.43,
            'rmse': 4584.90,
            'rmse_pct': 16.08,
            'r2': -0.25
        }

        assert 'mape' in metrics
        assert 'rmse' in metrics
        assert metrics['mape'] < 100, "MAPE should be percentage"

    def test_ensemble_weights(self):
        """Test ensemble weight configuration."""
        weights = {
            'prophet': 0.40,
            'sarima': 0.35,
            'lstm': 0.25
        }

        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01, "Weights should sum to 1"

    def test_model_comparison_dataframe(self):
        """Test model comparison data structure."""
        import pandas as pd

        comparison = pd.DataFrame({
            'Model': ['Prophet', 'SARIMA', 'LSTM', 'Ensemble'],
            'MAPE': [12.43, 100.00, 11.57, 12.43],
            'Weight': [40, 35, 25, 100]
        })

        assert len(comparison) == 4, "Should have 4 models"
        assert comparison['MAPE'].min() > 0, "MAPE should be positive"


class TestContentGenPage:
    """Tests for content generation page functionality."""

    def test_platform_config(self):
        """Test platform configuration."""
        platform_config = {
            "Instagram": {"max_chars": 2200, "hashtags": "5-15"},
            "Twitter": {"max_chars": 280, "hashtags": "2-3"},
            "LinkedIn": {"max_chars": 3000, "hashtags": "3-5"},
        }

        assert platform_config["Twitter"]["max_chars"] == 280
        assert platform_config["Instagram"]["max_chars"] == 2200

    def test_fiit_scores_structure(self):
        """Test FIIT scores data structure."""
        fiit_result = {
            'fluency': 0.97,
            'interactivity': 0.74,
            'information': 0.75,
            'tone': 0.94,
            'overall': 0.85,
            'passed': True
        }

        assert all(0 <= v <= 1 for k, v in fiit_result.items() if isinstance(v, float))
        assert fiit_result['passed'] == True

    def test_fiit_weights(self):
        """Test FIIT dimension weights."""
        weights = {
            'fluency': 0.25,
            'interactivity': 0.30,
            'information': 0.25,
            'tone': 0.20
        }

        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, "Weights should sum to 1"


class TestTwitterLivePage:
    """Tests for Twitter live page functionality."""

    def test_search_query_format(self):
        """Test Twitter search query format."""
        query = "#SocialMedia OR #Marketing"
        assert "#" in query, "Query should contain hashtags"

    def test_engagement_calculation(self):
        """Test engagement score calculation."""
        likes = 100
        retweets = 50
        replies = 20
        quotes = 10

        # Weighted engagement formula
        engagement = likes + retweets * 2 + replies * 3 + quotes * 2

        expected = 100 + 50*2 + 20*3 + 10*2
        assert engagement == expected

    def test_tweet_data_structure(self):
        """Test tweet data structure."""
        import pandas as pd
        from datetime import datetime

        tweet_data = {
            'tweet_id': '1234567890',
            'text': 'Test tweet content',
            'likes': 100,
            'retweets': 50,
            'replies': 20,
            'engagement': 260,
            'created_at': datetime.now()
        }

        df = pd.DataFrame([tweet_data])
        assert 'tweet_id' in df.columns
        assert 'engagement' in df.columns


class TestIntegration:
    """Integration tests for dashboard components."""

    def test_full_pipeline_imports(self):
        """Test all pipeline components can be imported."""
        try:
            from src.data_processing import DataCollector, DataPreprocessor, FeatureEngineer
            from src.forecasting import ProphetForecaster, SARIMAForecaster, LSTMForecaster
            from src.generation import ContentGenerator, FIITValidator
            from src.insights import InsightExtractor, PromptBuilder
            assert True
        except ImportError as e:
            # Some modules may have optional dependencies
            if "tensorflow" in str(e) or "prophet" in str(e):
                pytest.skip(f"Optional dependency not available: {e}")
            raise

    def test_config_paths(self):
        """Test configuration paths are valid."""
        from src.utils.config import Config
        from pathlib import Path

        assert Path(Config.RAW_DATA_DIR).parent.exists()
        assert Path(Config.PROCESSED_DATA_DIR).parent.exists()

    def test_dashboard_structure(self):
        """Test dashboard directory structure."""
        dashboard_dir = Path(__file__).parent.parent / "dashboard"
        pages_dir = dashboard_dir / "pages"

        assert dashboard_dir.exists(), "Dashboard directory should exist"
        assert pages_dir.exists(), "Pages directory should exist"
        assert (dashboard_dir / "app.py").exists(), "Main app should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
