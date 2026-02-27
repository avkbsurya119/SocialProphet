"""Data processing module for SocialProphet."""

from .collector import DataCollector
from .preprocessor import DataPreprocessor
from .features import FeatureEngineer

# Optional Twitter collector (requires tweepy)
try:
    from .twitter_collector import TwitterCollector
    __all__ = ["DataCollector", "DataPreprocessor", "FeatureEngineer", "TwitterCollector"]
except ImportError:
    __all__ = ["DataCollector", "DataPreprocessor", "FeatureEngineer"]
