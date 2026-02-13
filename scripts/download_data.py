"""
Data Download Script for SocialProphet.

Downloads datasets from Kaggle and prepares them for analysis.

Usage:
    python scripts/download_data.py --source kaggle
    python scripts/download_data.py --source sample
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config


def download_kaggle_datasets():
    """
    Download datasets from Kaggle.

    Requires:
    - Kaggle API installed: pip install kaggle
    - Kaggle credentials in ~/.kaggle/kaggle.json
    """
    try:
        import kaggle
    except ImportError:
        print("Error: Kaggle API not installed. Run: pip install kaggle")
        print("Then configure credentials: https://www.kaggle.com/docs/api")
        return False

    datasets = [
        "subashmaster0411/social-media-engagement-dataset",
        "kundanbedmutha/instagram-analytics-dataset",
    ]

    Config.ensure_directories()
    download_path = Config.RAW_DATA_DIR

    for dataset in datasets:
        print(f"\nDownloading: {dataset}")
        try:
            kaggle.api.dataset_download_files(
                dataset,
                path=str(download_path),
                unzip=True
            )
            print(f"Successfully downloaded: {dataset}")
        except Exception as e:
            print(f"Failed to download {dataset}: {e}")

    # List downloaded files
    print("\nDownloaded files:")
    for f in download_path.glob("*"):
        print(f"  - {f.name}")

    return True


def create_sample_dataset(num_posts: int = 1000, days: int = 180):
    """
    Create a synthetic sample dataset for testing.

    Args:
        num_posts: Number of posts to generate
        days: Number of days to span
    """
    print(f"Creating sample dataset: {num_posts} posts over {days} days")

    Config.ensure_directories()
    np.random.seed(42)

    # Generate timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    timestamps = pd.date_range(start=start_date, end=end_date, periods=num_posts)

    # Generate engagement metrics with realistic patterns
    hours = timestamps.hour
    days_of_week = timestamps.dayofweek

    # Base engagement with temporal patterns
    base_engagement = 100

    # Hour effect (higher engagement in evening)
    hour_effect = 1 + 0.5 * np.sin((hours - 6) * np.pi / 12)

    # Day effect (higher on weekends)
    day_effect = np.where(days_of_week >= 5, 1.3, 1.0)

    # Random noise
    noise = np.random.normal(1, 0.3, num_posts)
    noise = np.clip(noise, 0.3, 3.0)

    # Generate likes
    likes = (base_engagement * hour_effect * day_effect * noise).astype(int)
    likes = np.clip(likes, 0, None)

    # Comments are typically lower than likes
    comments = (likes * np.random.uniform(0.05, 0.15, num_posts)).astype(int)

    # Shares are even lower
    shares = (likes * np.random.uniform(0.01, 0.05, num_posts)).astype(int)

    # Generate content types
    content_types = np.random.choice(
        ["image", "video", "text", "carousel"],
        num_posts,
        p=[0.45, 0.25, 0.20, 0.10]
    )

    # Generate hashtag counts
    hashtag_counts = np.random.choice(range(0, 15), num_posts)

    # Generate sample captions
    topics = [
        "productivity tips", "morning routine", "tech review",
        "travel adventures", "fitness journey", "food recipes",
        "lifestyle hacks", "business insights", "motivation",
        "weekend vibes", "self improvement", "creative ideas"
    ]

    captions = [
        f"Check out our {np.random.choice(topics)}! #content #social"
        for _ in range(num_posts)
    ]

    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "engagement": likes + comments + shares,
        "content_type": content_types,
        "hashtag_count": hashtag_counts,
        "text": captions,
        "post_id": range(1, num_posts + 1),
    })

    # Save to file
    output_path = Config.RAW_DATA_DIR / "sample_social_media_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSample dataset saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nEngagement statistics:")
    print(df[['likes', 'comments', 'shares', 'engagement']].describe())

    return df


def validate_downloaded_data():
    """Check and validate downloaded data files."""
    print("\nValidating downloaded data...")

    data_path = Config.RAW_DATA_DIR
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in data/raw/")
        return False

    for csv_file in csv_files:
        print(f"\n--- {csv_file.name} ---")
        try:
            df = pd.read_csv(csv_file)
            print(f"Rows: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            print(f"Sample data:\n{df.head(2)}")
        except Exception as e:
            print(f"Error reading file: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Download datasets for SocialProphet")
    parser.add_argument(
        "--source",
        choices=["kaggle", "sample", "validate"],
        default="sample",
        help="Data source: 'kaggle' for Kaggle datasets, 'sample' for synthetic data, 'validate' to check existing data"
    )
    parser.add_argument(
        "--num-posts",
        type=int,
        default=1000,
        help="Number of posts for sample data (default: 1000)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days for sample data (default: 180)"
    )

    args = parser.parse_args()

    if args.source == "kaggle":
        download_kaggle_datasets()
    elif args.source == "sample":
        create_sample_dataset(args.num_posts, args.days)
    elif args.source == "validate":
        validate_downloaded_data()

    print("\nDone!")


if __name__ == "__main__":
    main()
