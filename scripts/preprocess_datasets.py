"""
Preprocess all raw datasets for SocialProphet.

This script:
1. Loads all raw datasets (Instagram, Social Media, Viral)
2. Standardizes column names
3. Cleans and validates data
4. Handles missing values
5. Calculates engagement metrics
6. Saves processed data

Usage:
    python scripts/preprocess_datasets.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.data_processing.preprocessor import DataPreprocessor


class DatasetProcessor:
    """Process and standardize all raw datasets."""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        Config.ensure_directories()
        self.raw_dir = Config.RAW_DATA_DIR
        self.processed_dir = Config.PROCESSED_DATA_DIR

    def process_instagram(self) -> pd.DataFrame:
        """
        Process Instagram_Analytics.csv dataset.

        Columns: post_datetime, likes, comments, shares, saves, reach, impressions
        """
        print("\n" + "="*60)
        print("Processing: Instagram_Analytics.csv")
        print("="*60)

        filepath = self.raw_dir / "Instagram_Analytics.csv"
        df = pd.read_csv(filepath)
        print(f"Loaded: {len(df)} rows")

        # Standardize column names
        df = df.rename(columns={
            "post_datetime": "timestamp",
            "post_id": "post_id",
            "likes": "likes",
            "comments": "comments",
            "shares": "shares",
            "saves": "saves",
            "reach": "reach",
            "impressions": "impressions",
            "engagement_rate": "engagement_rate_original",
            "caption_length": "text_length",
            "hashtags_count": "hashtag_count",
            "media_type": "content_type",
            "account_type": "account_type",
            "follower_count": "follower_count",
            "content_category": "category",
            "day_of_week": "day_of_week",
            "post_hour": "hour",
        })

        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Calculate total engagement
        df["engagement"] = (
            df["likes"].fillna(0) +
            df["comments"].fillna(0) +
            df["shares"].fillna(0) +
            df["saves"].fillna(0)
        ).astype(int)

        # Add source identifier
        df["source"] = "instagram"
        df["platform"] = "Instagram"

        # Select and order columns
        columns = [
            "post_id", "timestamp", "platform", "source",
            "likes", "comments", "shares", "saves", "engagement",
            "reach", "impressions", "engagement_rate_original",
            "content_type", "category", "account_type", "follower_count",
            "text_length", "hashtag_count", "hour", "day_of_week"
        ]
        df = df[[c for c in columns if c in df.columns]]

        # Clean data
        df = self._clean_dataframe(df)

        print(f"Processed: {len(df)} rows")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Engagement - Mean: {df['engagement'].mean():.2f}, Max: {df['engagement'].max()}")

        return df

    def process_social_media(self) -> pd.DataFrame:
        """
        Process Social Media Engagement Dataset.csv

        Columns: timestamp, likes_count, shares_count, comments_count, text_content
        """
        print("\n" + "="*60)
        print("Processing: Social Media Engagement Dataset.csv")
        print("="*60)

        filepath = self.raw_dir / "Social Media Engagement Dataset.csv"
        df = pd.read_csv(filepath)
        print(f"Loaded: {len(df)} rows")

        # Standardize column names
        df = df.rename(columns={
            "timestamp": "timestamp",
            "post_id": "post_id",
            "likes_count": "likes",
            "comments_count": "comments",
            "shares_count": "shares",
            "impressions": "impressions",
            "engagement_rate": "engagement_rate_original",
            "text_content": "text",
            "hashtags": "hashtags",
            "platform": "platform",
            "topic_category": "category",
            "sentiment_score": "sentiment_score",
            "sentiment_label": "sentiment_label",
            "day_of_week": "day_of_week",
        })

        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Calculate total engagement
        df["engagement"] = (
            df["likes"].fillna(0) +
            df["comments"].fillna(0) +
            df["shares"].fillna(0)
        ).astype(int)

        # Add source identifier
        df["source"] = "social_media_dataset"

        # Calculate text length if text exists
        if "text" in df.columns:
            df["text_length"] = df["text"].astype(str).str.len()

        # Count hashtags
        if "hashtags" in df.columns:
            df["hashtag_count"] = df["hashtags"].astype(str).str.count(r"#?\w+")

        # Extract hour from timestamp
        df["hour"] = df["timestamp"].dt.hour

        # Select and order columns
        columns = [
            "post_id", "timestamp", "platform", "source",
            "likes", "comments", "shares", "engagement",
            "impressions", "engagement_rate_original",
            "category", "sentiment_score", "sentiment_label",
            "text_length", "hashtag_count", "hour", "day_of_week"
        ]
        df = df[[c for c in columns if c in df.columns]]

        # Clean data
        df = self._clean_dataframe(df)

        print(f"Processed: {len(df)} rows")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Engagement - Mean: {df['engagement'].mean():.2f}, Max: {df['engagement'].max()}")

        return df

    def process_viral(self) -> pd.DataFrame:
        """
        Process social_media_viral_content_dataset.csv

        Columns: post_datetime, likes, comments, shares, views, is_viral
        """
        print("\n" + "="*60)
        print("Processing: social_media_viral_content_dataset.csv")
        print("="*60)

        filepath = self.raw_dir / "social_media_viral_content_dataset.csv"
        df = pd.read_csv(filepath)
        print(f"Loaded: {len(df)} rows")

        # Standardize column names
        df = df.rename(columns={
            "post_datetime": "timestamp",
            "post_id": "post_id",
            "likes": "likes",
            "comments": "comments",
            "shares": "shares",
            "views": "views",
            "engagement_rate": "engagement_rate_original",
            "platform": "platform",
            "content_type": "content_type",
            "topic": "category",
            "sentiment_score": "sentiment_score",
            "is_viral": "is_viral",
            "hashtags": "hashtags",
        })

        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Calculate total engagement
        df["engagement"] = (
            df["likes"].fillna(0) +
            df["comments"].fillna(0) +
            df["shares"].fillna(0)
        ).astype(int)

        # Add source identifier
        df["source"] = "viral_dataset"

        # Extract hour and day from timestamp
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.day_name()

        # Count hashtags
        if "hashtags" in df.columns:
            df["hashtag_count"] = df["hashtags"].astype(str).str.count(r"#?\w+")

        # Select and order columns
        columns = [
            "post_id", "timestamp", "platform", "source",
            "likes", "comments", "shares", "views", "engagement",
            "engagement_rate_original", "content_type", "category",
            "sentiment_score", "is_viral",
            "hashtag_count", "hour", "day_of_week"
        ]
        df = df[[c for c in columns if c in df.columns]]

        # Clean data
        df = self._clean_dataframe(df)

        print(f"Processed: {len(df)} rows")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Engagement - Mean: {df['engagement'].mean():.2f}, Max: {df['engagement'].max()}")

        return df

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply common cleaning operations."""
        # Remove rows with invalid timestamps
        df = df.dropna(subset=["timestamp"])

        # Ensure numeric columns are numeric
        numeric_cols = ["likes", "comments", "shares", "engagement", "views", "saves"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Remove negative engagement values
        if "engagement" in df.columns:
            df = df[df["engagement"] >= 0]

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def combine_datasets(
        self,
        instagram_df: pd.DataFrame,
        social_media_df: pd.DataFrame,
        viral_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine all datasets into one."""
        print("\n" + "="*60)
        print("Combining all datasets")
        print("="*60)

        # Common columns for combining
        common_cols = [
            "post_id", "timestamp", "platform", "source",
            "likes", "comments", "shares", "engagement",
            "category", "hour", "day_of_week"
        ]

        # Select common columns from each dataset
        dfs = []
        for df, name in [
            (instagram_df, "Instagram"),
            (social_media_df, "Social Media"),
            (viral_df, "Viral")
        ]:
            available_cols = [c for c in common_cols if c in df.columns]
            dfs.append(df[available_cols].copy())
            print(f"{name}: {len(df)} rows")

        # Combine
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        print(f"\nCombined total: {len(combined)} rows")
        print(f"Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

        return combined

    def save_processed(
        self,
        instagram_df: pd.DataFrame,
        social_media_df: pd.DataFrame,
        viral_df: pd.DataFrame,
        combined_df: pd.DataFrame
    ):
        """Save all processed datasets."""
        print("\n" + "="*60)
        print("Saving processed datasets")
        print("="*60)

        # Save individual datasets
        instagram_df.to_csv(self.processed_dir / "instagram_cleaned.csv", index=False)
        print(f"Saved: instagram_cleaned.csv ({len(instagram_df)} rows)")

        social_media_df.to_csv(self.processed_dir / "social_media_cleaned.csv", index=False)
        print(f"Saved: social_media_cleaned.csv ({len(social_media_df)} rows)")

        viral_df.to_csv(self.processed_dir / "viral_cleaned.csv", index=False)
        print(f"Saved: viral_cleaned.csv ({len(viral_df)} rows)")

        combined_df.to_csv(self.processed_dir / "combined_data.csv", index=False)
        print(f"Saved: combined_data.csv ({len(combined_df)} rows)")

    def run(self):
        """Run complete preprocessing pipeline."""
        print("\n" + "#"*60)
        print("# SocialProphet - Data Preprocessing Pipeline")
        print("#"*60)

        # Process each dataset
        instagram_df = self.process_instagram()
        social_media_df = self.process_social_media()
        viral_df = self.process_viral()

        # Combine datasets
        combined_df = self.combine_datasets(instagram_df, social_media_df, viral_df)

        # Save all processed data
        self.save_processed(instagram_df, social_media_df, viral_df, combined_df)

        print("\n" + "#"*60)
        print("# Preprocessing Complete!")
        print("#"*60)

        # Summary
        print("\nSummary:")
        print(f"  Instagram:    {len(instagram_df):,} rows")
        print(f"  Social Media: {len(social_media_df):,} rows")
        print(f"  Viral:        {len(viral_df):,} rows")
        print(f"  Combined:     {len(combined_df):,} rows")

        return {
            "instagram": instagram_df,
            "social_media": social_media_df,
            "viral": viral_df,
            "combined": combined_df,
        }


def main():
    processor = DatasetProcessor()
    processor.run()


if __name__ == "__main__":
    main()
