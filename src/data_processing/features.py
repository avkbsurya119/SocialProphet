"""
Feature Engineering Module for SocialProphet.

Creates temporal, content, and historical features for forecasting.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """Feature engineering for social media engagement forecasting."""

    def __init__(self):
        """Initialize the feature engineer."""
        pass

    def add_temporal_features(self, df: pd.DataFrame, date_column: str = "timestamp") -> pd.DataFrame:
        """
        Add temporal features from datetime column.

        Args:
            df: DataFrame
            date_column: Name of date column

        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Basic temporal features
        df["hour"] = df[date_column].dt.hour
        df["day_of_week"] = df[date_column].dt.dayofweek
        df["day_of_month"] = df[date_column].dt.day
        df["month"] = df[date_column].dt.month
        df["year"] = df[date_column].dt.year
        df["week_of_year"] = df[date_column].dt.isocalendar().week.astype(int)

        # Binary features
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_month_start"] = df[date_column].dt.is_month_start.astype(int)
        df["is_month_end"] = df[date_column].dt.is_month_end.astype(int)

        # Cyclical encoding (for better model performance)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def add_lag_features(
        self,
        df: pd.DataFrame,
        value_column: str = "engagement",
        lags: List[int] = [1, 7, 14, 30]
    ) -> pd.DataFrame:
        """
        Add lag features for time series.

        Args:
            df: DataFrame (sorted by time)
            value_column: Column to create lags for
            lags: List of lag periods

        Returns:
            DataFrame with lag features
        """
        df = df.copy()

        for lag in lags:
            df[f"{value_column}_lag_{lag}"] = df[value_column].shift(lag)

        return df

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        value_column: str = "engagement",
        windows: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """
        Add rolling window statistics.

        Args:
            df: DataFrame (sorted by time)
            value_column: Column to calculate rolling stats for
            windows: List of window sizes

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()

        for window in windows:
            # Rolling mean
            df[f"{value_column}_rolling_mean_{window}"] = (
                df[value_column].rolling(window=window, min_periods=1).mean()
            )

            # Rolling std
            df[f"{value_column}_rolling_std_{window}"] = (
                df[value_column].rolling(window=window, min_periods=1).std()
            )

            # Rolling min/max
            df[f"{value_column}_rolling_min_{window}"] = (
                df[value_column].rolling(window=window, min_periods=1).min()
            )
            df[f"{value_column}_rolling_max_{window}"] = (
                df[value_column].rolling(window=window, min_periods=1).max()
            )

        return df

    def add_rate_of_change(
        self,
        df: pd.DataFrame,
        value_column: str = "engagement",
        periods: List[int] = [1, 7]
    ) -> pd.DataFrame:
        """
        Add rate of change features.

        Args:
            df: DataFrame (sorted by time)
            value_column: Column to calculate ROC for
            periods: List of periods for ROC calculation

        Returns:
            DataFrame with ROC features
        """
        df = df.copy()

        for period in periods:
            df[f"{value_column}_roc_{period}"] = (
                df[value_column].pct_change(periods=period)
            )

        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def add_content_features(
        self,
        df: pd.DataFrame,
        text_column: str = "text"
    ) -> pd.DataFrame:
        """
        Add content-based features from text.

        Args:
            df: DataFrame
            text_column: Name of text column

        Returns:
            DataFrame with content features
        """
        df = df.copy()

        if text_column not in df.columns:
            return df

        # Text length
        df["text_length"] = df[text_column].astype(str).str.len()

        # Word count
        df["word_count"] = df[text_column].astype(str).str.split().str.len()

        # Hashtag count
        df["hashtag_count"] = df[text_column].astype(str).str.count(r"#\w+")

        # Mention count
        df["mention_count"] = df[text_column].astype(str).str.count(r"@\w+")

        # URL count
        df["url_count"] = df[text_column].astype(str).str.count(r"https?://\S+")

        # Has emoji (simplified check)
        df["has_emoji"] = df[text_column].astype(str).str.contains(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]",
            regex=True
        ).astype(int)

        # Capitalization ratio
        df["caps_ratio"] = df[text_column].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )

        return df

    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        text_column: str = "text"
    ) -> pd.DataFrame:
        """
        Add sentiment analysis features.

        Args:
            df: DataFrame
            text_column: Name of text column

        Returns:
            DataFrame with sentiment features
        """
        df = df.copy()

        if text_column not in df.columns:
            return df

        try:
            from textblob import TextBlob

            df["sentiment_polarity"] = df[text_column].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity
            )
            df["sentiment_subjectivity"] = df[text_column].apply(
                lambda x: TextBlob(str(x)).sentiment.subjectivity
            )

        except ImportError:
            print("TextBlob not installed. Skipping sentiment features.")
            df["sentiment_polarity"] = 0
            df["sentiment_subjectivity"] = 0

        return df

    def create_all_features(
        self,
        df: pd.DataFrame,
        date_column: str = "timestamp",
        value_column: str = "engagement",
        text_column: Optional[str] = "text",
        include_sentiment: bool = False
    ) -> pd.DataFrame:
        """
        Create all features for the dataset.

        Args:
            df: DataFrame
            date_column: Name of date column
            value_column: Name of value column
            text_column: Name of text column (optional)
            include_sentiment: Whether to include sentiment analysis

        Returns:
            DataFrame with all features
        """
        df = df.copy()

        # Temporal features
        df = self.add_temporal_features(df, date_column)

        # Lag features
        df = self.add_lag_features(df, value_column)

        # Rolling features
        df = self.add_rolling_features(df, value_column)

        # Rate of change
        df = self.add_rate_of_change(df, value_column)

        # Content features
        if text_column and text_column in df.columns:
            df = self.add_content_features(df, text_column)

            if include_sentiment:
                df = self.add_sentiment_features(df, text_column)

        # Fill NaN values created by lag/rolling operations
        df = df.fillna(0)

        return df

    def get_feature_names(self, df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
        """
        Get list of feature column names.

        Args:
            df: DataFrame with features
            exclude: Columns to exclude

        Returns:
            List of feature column names
        """
        if exclude is None:
            exclude = ["timestamp", "ds", "y", "text", "engagement"]

        feature_cols = [c for c in df.columns if c not in exclude]
        return feature_cols
