"""
Data Collection Module for SocialProphet.

Handles data collection from various sources:
- Kaggle datasets
- Twitter API
- CSV/JSON files
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import requests
import json

from ..utils.config import Config


class DataCollector:
    """Unified data collector for social media data."""

    def __init__(self):
        """Initialize the data collector."""
        self.config = Config()
        Config.ensure_directories()

    def load_csv(
        self,
        filepath: str,
        date_column: str = "timestamp",
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            filepath: Path to the CSV file
            date_column: Name of the date/timestamp column
            parse_dates: Whether to parse dates automatically

        Returns:
            DataFrame with loaded data
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if parse_dates and date_column:
            df = pd.read_csv(path, parse_dates=[date_column])
        else:
            df = pd.read_csv(path)

        print(f"Loaded {len(df)} rows from {path.name}")
        return df

    def load_kaggle_dataset(
        self,
        dataset_name: str,
        filename: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download and load a dataset from Kaggle.

        Args:
            dataset_name: Kaggle dataset identifier (e.g., "username/dataset-name")
            filename: Specific file to load from the dataset

        Returns:
            DataFrame with loaded data
        """
        try:
            import kaggle
        except ImportError:
            raise ImportError(
                "Kaggle API not installed. Run: pip install kaggle"
            )

        # Download dataset
        download_path = Config.RAW_DATA_DIR
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(download_path),
            unzip=True
        )

        # Find and load the file
        if filename:
            filepath = download_path / filename
        else:
            # Find first CSV file
            csv_files = list(download_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in downloaded dataset")
            filepath = csv_files[0]

        return self.load_csv(str(filepath))

    def collect_twitter_data(
        self,
        query: str,
        max_results: int = 100,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Collect tweets using Twitter API v2.

        Args:
            query: Search query for tweets
            max_results: Maximum number of tweets to collect
            days_back: Number of days to look back

        Returns:
            DataFrame with tweet data
        """
        if not Config.TWITTER_BEARER_TOKEN:
            raise ValueError(
                "Twitter Bearer Token not configured. "
                "Set TWITTER_BEARER_TOKEN in .env file"
            )

        headers = {
            "Authorization": f"Bearer {Config.TWITTER_BEARER_TOKEN}",
        }

        # Calculate date range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)

        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            "query": query,
            "max_results": min(max_results, 100),
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tweet.fields": "created_at,public_metrics,author_id,lang",
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f"Twitter API error: {response.status_code} - {response.text}")

        data = response.json()
        tweets = data.get("data", [])

        # Convert to DataFrame
        records = []
        for tweet in tweets:
            metrics = tweet.get("public_metrics", {})
            records.append({
                "timestamp": tweet.get("created_at"),
                "text": tweet.get("text"),
                "likes": metrics.get("like_count", 0),
                "retweets": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
                "quotes": metrics.get("quote_count", 0),
                "author_id": tweet.get("author_id"),
                "lang": tweet.get("lang"),
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["engagement"] = df["likes"] + df["retweets"] + df["replies"]

        print(f"Collected {len(df)} tweets for query: '{query}'")
        return df

    def load_instagram_data(self, filepath: str) -> pd.DataFrame:
        """
        Load Instagram analytics data from file.

        Args:
            filepath: Path to Instagram data file

        Returns:
            DataFrame with standardized Instagram data
        """
        df = self.load_csv(filepath)

        # Standardize column names
        column_mapping = {
            "Likes": "likes",
            "Comments": "comments",
            "Shares": "shares",
            "Date": "timestamp",
            "Caption": "text",
            "Hashtags": "hashtags",
        }

        df = df.rename(columns=lambda x: column_mapping.get(x, x.lower()))

        # Calculate engagement
        if "engagement" not in df.columns:
            df["engagement"] = df.get("likes", 0) + df.get("comments", 0) + df.get("shares", 0)

        return df

    def combine_datasets(
        self,
        datasets: List[pd.DataFrame],
        deduplicate: bool = True
    ) -> pd.DataFrame:
        """
        Combine multiple datasets into one.

        Args:
            datasets: List of DataFrames to combine
            deduplicate: Whether to remove duplicate rows

        Returns:
            Combined DataFrame
        """
        combined = pd.concat(datasets, ignore_index=True)

        if deduplicate:
            # Remove duplicates based on timestamp and text
            if "text" in combined.columns:
                combined = combined.drop_duplicates(
                    subset=["timestamp", "text"],
                    keep="first"
                )

        combined = combined.sort_values("timestamp").reset_index(drop=True)
        print(f"Combined dataset: {len(combined)} total rows")
        return combined

    def save_data(
        self,
        df: pd.DataFrame,
        filename: str,
        data_type: str = "raw"
    ) -> Path:
        """
        Save DataFrame to file.

        Args:
            df: DataFrame to save
            filename: Output filename
            data_type: Type of data (raw, processed, predictions)

        Returns:
            Path to saved file
        """
        filepath = Config.get_data_path(filename, data_type)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} rows to {filepath}")
        return filepath

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for a dataset.

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "date_range": None,
            "engagement_stats": None,
        }

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            summary["date_range"] = {
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
                "days": (df["timestamp"].max() - df["timestamp"].min()).days,
            }

        if "engagement" in df.columns:
            summary["engagement_stats"] = {
                "mean": df["engagement"].mean(),
                "median": df["engagement"].median(),
                "std": df["engagement"].std(),
                "min": df["engagement"].min(),
                "max": df["engagement"].max(),
            }

        return summary
