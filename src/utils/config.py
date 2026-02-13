"""
Configuration management for SocialProphet.

Handles environment variables, paths, and model parameters.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for SocialProphet."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    PREDICTIONS_DIR = DATA_DIR / "predictions"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

    # API Keys (loaded from environment)
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    TWITTER_CONSUMER_KEY: Optional[str] = os.getenv("TWITTER_CONSUMER_KEY")
    TWITTER_CONSUMER_SECRET: Optional[str] = os.getenv("TWITTER_CONSUMER_SECRET")
    TWITTER_BEARER_TOKEN: Optional[str] = os.getenv("TWITTER_BEARER_TOKEN")
    KAGGLE_USERNAME: Optional[str] = os.getenv("KAGGLE_USERNAME")
    KAGGLE_KEY: Optional[str] = os.getenv("KAGGLE_KEY")

    # Hugging Face Model Settings
    HF_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

    # Forecasting Parameters
    FORECAST_HORIZON = 30  # Days to forecast ahead
    TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test

    # Prophet Model Parameters
    PROPHET_PARAMS = {
        "daily_seasonality": True,
        "weekly_seasonality": True,
        "yearly_seasonality": True,
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
    }

    # SARIMA Parameters (default, will be auto-tuned)
    SARIMA_PARAMS = {
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 7),  # Weekly seasonality
    }

    # Ensemble Weights
    ENSEMBLE_WEIGHTS = {
        "prophet": 0.4,
        "sarima": 0.35,
        "lstm": 0.25,  # Optional
    }

    # LLM Generation Parameters
    LLM_PARAMS = {
        "max_new_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    }

    # FIIT Framework Thresholds
    FIIT_THRESHOLDS = {
        "fluency": 0.7,
        "interactivity": 0.7,
        "information": 0.7,
        "tone": 0.7,
    }

    # Data Collection Settings
    DATA_COLLECTION = {
        "min_posts": 500,
        "min_days": 90,
        "max_days": 180,
    }

    @classmethod
    def validate_api_keys(cls) -> dict:
        """Check which API keys are configured."""
        return {
            "huggingface": cls.HF_TOKEN is not None,
            "twitter": all([
                cls.TWITTER_CONSUMER_KEY,
                cls.TWITTER_CONSUMER_SECRET,
                cls.TWITTER_BEARER_TOKEN,
            ]),
            "kaggle": all([
                cls.KAGGLE_USERNAME,
                cls.KAGGLE_KEY,
            ]),
        }

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        for directory in [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.PREDICTIONS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_data_path(cls, filename: str, data_type: str = "raw") -> Path:
        """Get full path for a data file."""
        if data_type == "raw":
            return cls.RAW_DATA_DIR / filename
        elif data_type == "processed":
            return cls.PROCESSED_DATA_DIR / filename
        elif data_type == "predictions":
            return cls.PREDICTIONS_DIR / filename
        else:
            raise ValueError(f"Unknown data_type: {data_type}")


# Create singleton instance
config = Config()
