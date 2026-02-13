"""
Data Preprocessing Module for SocialProphet.

Handles data cleaning, validation, and transformation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from datetime import datetime

from ..utils.config import Config


class DataPreprocessor:
    """Preprocessor for social media engagement data."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.config = Config()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Remove rows with all missing values
        df = df.dropna(how="all")

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

        # Ensure engagement metrics are numeric
        numeric_cols = ["likes", "comments", "shares", "retweets", "replies"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Calculate total engagement if not present
        if "engagement" not in df.columns:
            engagement_cols = [c for c in numeric_cols if c in df.columns]
            if engagement_cols:
                df["engagement"] = df[engagement_cols].sum(axis=1)

        # Remove negative values
        if "engagement" in df.columns:
            df = df[df["engagement"] >= 0]

        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"Cleaned data: {len(df)} rows remaining")
        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "forward_fill"
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: DataFrame with potential missing values
            method: Method to handle missing values
                   ("forward_fill", "backward_fill", "interpolate", "mean", "drop")

        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == "forward_fill":
            df[numeric_cols] = df[numeric_cols].ffill()
        elif method == "backward_fill":
            df[numeric_cols] = df[numeric_cols].bfill()
        elif method == "interpolate":
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
        elif method == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == "drop":
            df = df.dropna(subset=numeric_cols)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Fill remaining NaN with 0
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        column: str = "engagement",
        method: str = "iqr",
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from the dataset.

        Args:
            df: DataFrame
            column: Column to check for outliers
            method: Outlier detection method ("iqr", "zscore")
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        original_len = len(df)

        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        elif method == "zscore":
            mean = df[column].mean()
            std = df[column].std()
            df = df[np.abs((df[column] - mean) / std) <= threshold]

        else:
            raise ValueError(f"Unknown method: {method}")

        removed = original_len - len(df)
        print(f"Removed {removed} outliers ({removed/original_len*100:.1f}%)")
        return df

    def create_time_series(
        self,
        df: pd.DataFrame,
        date_column: str = "timestamp",
        value_column: str = "engagement",
        freq: str = "D"
    ) -> pd.DataFrame:
        """
        Convert data to time series format.

        Args:
            df: DataFrame
            date_column: Name of date column
            value_column: Name of value column
            freq: Frequency for resampling ("D"=daily, "H"=hourly, "W"=weekly)

        Returns:
            Time series DataFrame
        """
        df = df.copy()

        # Set datetime index
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

        # Aggregate by frequency
        ts = df[[value_column]].resample(freq).agg({
            value_column: ["sum", "mean", "count"]
        })

        # Flatten column names
        ts.columns = [f"{value_column}_{agg}" for agg in ["sum", "mean", "count"]]

        # Fill missing dates
        ts = ts.asfreq(freq, fill_value=0)

        return ts.reset_index()

    def temporal_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        date_column: str = "timestamp"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally (not randomly) for time series.

        Args:
            df: DataFrame
            test_size: Proportion of data for testing
            date_column: Name of date column

        Returns:
            Tuple of (train_df, test_df)
        """
        df = df.sort_values(date_column).reset_index(drop=True)

        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        print(f"Train period: {train_df[date_column].min()} to {train_df[date_column].max()}")
        print(f"Test period: {test_df[date_column].min()} to {test_df[date_column].max()}")

        return train_df, test_df

    def prepare_prophet_data(
        self,
        df: pd.DataFrame,
        date_column: str = "timestamp",
        value_column: str = "engagement"
    ) -> pd.DataFrame:
        """
        Prepare data for Facebook Prophet.

        Prophet requires 'ds' (date) and 'y' (value) columns.

        Args:
            df: DataFrame
            date_column: Name of date column
            value_column: Name of value column

        Returns:
            Prophet-formatted DataFrame
        """
        prophet_df = df[[date_column, value_column]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        return prophet_df

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate data quality.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "stats": {},
        }

        # Check minimum rows
        if len(df) < Config.DATA_COLLECTION["min_posts"]:
            validation["issues"].append(
                f"Insufficient data: {len(df)} rows "
                f"(minimum: {Config.DATA_COLLECTION['min_posts']})"
            )
            validation["is_valid"] = False

        # Check date range
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            date_range = (df["timestamp"].max() - df["timestamp"].min()).days
            if date_range < Config.DATA_COLLECTION["min_days"]:
                validation["issues"].append(
                    f"Insufficient date range: {date_range} days "
                    f"(minimum: {Config.DATA_COLLECTION['min_days']})"
                )

            validation["stats"]["date_range_days"] = date_range

        # Check for required columns
        required_cols = ["timestamp", "engagement"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            validation["issues"].append(f"Missing columns: {missing_cols}")
            validation["is_valid"] = False

        # Check for missing values
        if "engagement" in df.columns:
            missing_pct = df["engagement"].isna().mean() * 100
            if missing_pct > 10:
                validation["issues"].append(
                    f"High missing value rate: {missing_pct:.1f}%"
                )

            validation["stats"]["missing_engagement_pct"] = missing_pct

        validation["stats"]["total_rows"] = len(df)

        return validation
