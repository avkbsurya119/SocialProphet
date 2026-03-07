"""
Reprocess data for better forecasting performance.

Instead of daily aggregation (which creates high variance),
we use post-level features with time-based train/test split.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


def load_raw_data() -> pd.DataFrame:
    """Load and combine all raw data files."""
    raw_dir = Config.RAW_DATA_DIR
    dfs = []

    # Instagram Analytics (largest dataset)
    ig_path = raw_dir / "Instagram_Analytics.csv"
    if ig_path.exists():
        df = pd.read_csv(ig_path)
        if 'likes' in df.columns and 'comments' in df.columns:
            df['engagement'] = df['likes'] + df['comments'] * 2 + df.get('shares', 0) * 3
            if 'timestamp' not in df.columns and 'post_date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['post_date'])
            elif 'timestamp' not in df.columns:
                # Generate timestamps for the last year
                n = len(df)
                dates = pd.date_range(end=datetime.now(), periods=n, freq='H')
                df['timestamp'] = dates
            dfs.append(df[['timestamp', 'engagement', 'likes', 'comments']].copy())

    # Social Media Engagement Dataset
    sm_path = raw_dir / "Social Media Engagement Dataset.csv"
    if sm_path.exists():
        df = pd.read_csv(sm_path)
        if 'likes' in df.columns:
            df['engagement'] = df['likes'] + df.get('comments', 0) * 2 + df.get('shares', 0) * 3
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                n = len(df)
                dates = pd.date_range(end=datetime.now(), periods=n, freq='H')
                df['timestamp'] = dates
            cols = ['timestamp', 'engagement']
            if 'likes' in df.columns:
                cols.append('likes')
            if 'comments' in df.columns:
                cols.append('comments')
            dfs.append(df[cols].copy())

    # Sample data
    sample_path = raw_dir / "sample_social_media_data.csv"
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        if 'engagement' in df.columns:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            dfs.append(df[['timestamp', 'engagement']].copy())

    if not dfs:
        raise ValueError("No data files found")

    combined = pd.concat(dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'], errors='coerce')
    combined = combined.dropna(subset=['timestamp', 'engagement'])
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    print(f"Loaded {len(combined)} records")
    return combined


def create_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily aggregated features with better signal."""
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour

    # Aggregate to daily
    daily = df.groupby('date').agg({
        'engagement': ['sum', 'mean', 'std', 'count', 'min', 'max']
    }).reset_index()
    daily.columns = ['ds', 'engagement_sum', 'engagement_mean', 'engagement_std',
                     'post_count', 'engagement_min', 'engagement_max']
    daily['ds'] = pd.to_datetime(daily['ds'])

    # Target variable - use mean engagement per post (more stable than sum)
    daily['y_raw'] = daily['engagement_mean']
    daily['y'] = np.log1p(daily['y_raw'])

    # Fill missing std
    daily['engagement_std'] = daily['engagement_std'].fillna(0)

    # Sort by date
    daily = daily.sort_values('ds').reset_index(drop=True)

    # Add temporal features
    daily['day_of_week'] = daily['ds'].dt.dayofweek
    daily['day_of_month'] = daily['ds'].dt.day
    daily['month'] = daily['ds'].dt.month
    daily['week_of_year'] = daily['ds'].dt.isocalendar().week.astype(int)
    daily['is_weekend'] = (daily['day_of_week'] >= 5).astype(int)

    # Cyclical encoding
    daily['day_sin'] = np.sin(2 * np.pi * daily['day_of_week'] / 7)
    daily['day_cos'] = np.cos(2 * np.pi * daily['day_of_week'] / 7)
    daily['month_sin'] = np.sin(2 * np.pi * daily['month'] / 12)
    daily['month_cos'] = np.cos(2 * np.pi * daily['month'] / 12)

    # Lag features on mean engagement (more stable)
    for lag in [1, 7, 14, 30]:
        daily[f'y_lag_{lag}'] = daily['y'].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        daily[f'y_rolling_mean_{window}'] = daily['y'].rolling(window, min_periods=1).mean()
        daily[f'y_rolling_std_{window}'] = daily['y'].rolling(window, min_periods=1).std().fillna(0)

    # Percent change
    daily['y_pct_change_1'] = daily['y'].pct_change(1).fillna(0)
    daily['y_pct_change_7'] = daily['y'].pct_change(7).fillna(0)

    return daily


def create_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
    """Temporal train/test split."""
    n = len(df)
    train_size = int(n * (1 - test_ratio))

    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    return train_df, test_df


def main():
    """Main reprocessing pipeline."""
    print("=" * 60)
    print("REPROCESSING DATA FOR BETTER FORECASTING")
    print("=" * 60)

    # Load raw data
    print("\n1. Loading raw data...")
    raw_df = load_raw_data()
    print(f"   Total records: {len(raw_df)}")
    print(f"   Date range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")

    # Create daily features
    print("\n2. Creating daily features...")
    daily_df = create_daily_features(raw_df)
    print(f"   Daily records: {len(daily_df)}")
    print(f"   y_raw (engagement mean) range: [{daily_df['y_raw'].min():.0f}, {daily_df['y_raw'].max():.0f}]")
    print(f"   y_raw std: {daily_df['y_raw'].std():.0f}")

    # Drop rows with NaN (from lag features)
    daily_df = daily_df.dropna()
    print(f"   After dropping NaN: {len(daily_df)} rows")

    # Train/test split
    print("\n3. Creating train/test split...")
    train_df, test_df = create_train_test_split(daily_df)
    print(f"   Train: {len(train_df)} rows")
    print(f"   Test: {len(test_df)} rows")

    # Save
    print("\n4. Saving processed data...")
    Config.ensure_directories()

    train_df.to_csv(Config.PROCESSED_DATA_DIR / "train_data.csv", index=False)
    test_df.to_csv(Config.PROCESSED_DATA_DIR / "test_data.csv", index=False)

    # Prophet format
    train_prophet = train_df[['ds', 'y']].copy()
    test_prophet = test_df[['ds', 'y']].copy()
    train_prophet.to_csv(Config.PROCESSED_DATA_DIR / "train_prophet.csv", index=False)
    test_prophet.to_csv(Config.PROCESSED_DATA_DIR / "test_prophet.csv", index=False)

    print(f"   Saved to {Config.PROCESSED_DATA_DIR}")

    # Quick stats
    print("\n5. Data Statistics:")
    print(f"   Train y_raw mean: {train_df['y_raw'].mean():.0f}")
    print(f"   Train y_raw std: {train_df['y_raw'].std():.0f}")
    print(f"   Test y_raw mean: {test_df['y_raw'].mean():.0f}")
    print(f"   Test y_raw std: {test_df['y_raw'].std():.0f}")

    # Check predictability
    print("\n6. Baseline Predictability Check:")
    # Day-of-week means
    dow_means = train_df.groupby('day_of_week')['y_raw'].mean().to_dict()
    y_true = test_df['y_raw'].values
    y_pred_dow = test_df['day_of_week'].map(dow_means).values

    ss_res = np.sum((y_true - y_pred_dow) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_dow = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"   Day-of-week baseline R²: {r2_dow:.4f}")

    print("\n" + "=" * 60)
    print("REPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
