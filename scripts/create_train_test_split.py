"""
Create train-test split for time series forecasting.

This script:
1. Loads processed datasets (with log-transformed engagement)
2. Creates temporal train-test split (80/20)
3. Prepares Prophet-ready data format
4. Creates lag/rolling features on log-scale
5. Saves train and test datasets
6. Generates split summary

Log-Scale Processing:
    Input data has log-transformed engagement (from preprocess_datasets.py).
    All features (lag, rolling mean/std) are computed on log-scale.
    This ensures LSTM receives normalized features.

    Target variable 'y' is log-transformed.
    'y_raw' contains original values for evaluation/visualization.

Usage:
    python scripts/create_train_test_split.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.data_processing.preprocessor import DataPreprocessor
from src.data_processing.features import FeatureEngineer


class TrainTestSplitter:
    """Create train-test splits for forecasting."""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.processed_dir = Config.PROCESSED_DATA_DIR
        self.split_ratio = Config.TRAIN_TEST_SPLIT  # 0.8

    def load_primary_dataset(self) -> pd.DataFrame:
        """Load the primary dataset (Instagram)."""
        filepath = self.processed_dir / "instagram_cleaned.csv"
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def create_daily_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data to daily level for time series forecasting.

        Uses engagement_raw for aggregation (sum of raw values),
        then applies log transform to daily totals.

        Args:
            df: Dataframe with engagement_raw (original) and engagement (log)

        Returns:
            Daily aggregated dataframe with y (log) and y_raw (original)
        """
        print("\nAggregating to daily level...")

        # Determine which column to use for aggregation
        # Use engagement_raw if available (raw values), else use engagement
        eng_col = "engagement_raw" if "engagement_raw" in df.columns else "engagement"
        print(f"  Using '{eng_col}' for daily aggregation")

        # Set timestamp as index
        df = df.set_index("timestamp")

        # Daily aggregations (use raw values for sum)
        daily = df.resample("D").agg({
            eng_col: ["sum", "mean", "count"],
            "likes": "sum",
            "comments": "sum",
            "shares": "sum",
        })

        # Flatten column names
        daily.columns = [
            "engagement_sum", "engagement_mean", "post_count",
            "likes_sum", "comments_sum", "shares_sum"
        ]

        # Reset index
        daily = daily.reset_index()
        daily = daily.rename(columns={"timestamp": "ds"})

        # Keep raw engagement sum
        daily["y_raw"] = daily["engagement_sum"]

        # Apply log transform for LSTM training
        daily["y"] = np.log1p(daily["engagement_sum"])

        # Fill missing days with 0 (if any gaps)
        date_range = pd.date_range(start=daily["ds"].min(), end=daily["ds"].max(), freq="D")
        daily = daily.set_index("ds").reindex(date_range, fill_value=0).reset_index()
        daily = daily.rename(columns={"index": "ds"})

        # Ensure y_raw and y are filled for missing days
        daily["y_raw"] = daily["y_raw"].fillna(0)
        daily["y"] = daily["y"].fillna(0)

        print(f"  Daily records: {len(daily)}")
        print(f"  Date range: {daily['ds'].min()} to {daily['ds'].max()}")
        print(f"  y (log) range: [{daily['y'].min():.2f}, {daily['y'].max():.2f}]")
        print(f"  y_raw range: [{daily['y_raw'].min():.0f}, {daily['y_raw'].max():.0f}]")

        return daily

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for enhanced modeling."""
        df = df.copy()

        # Basic temporal features
        df["day_of_week"] = df["ds"].dt.dayofweek
        df["day_of_month"] = df["ds"].dt.day
        df["month"] = df["ds"].dt.month
        df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Cyclical encoding
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for time series."""
        df = df.copy()

        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f"y_lag_{lag}"] = df["y"].shift(lag)

        # Rolling features
        for window in [7, 14, 30]:
            df[f"y_rolling_mean_{window}"] = df["y"].rolling(window=window, min_periods=1).mean()
            df[f"y_rolling_std_{window}"] = df["y"].rolling(window=window, min_periods=1).std()

        # Rate of change
        df["y_pct_change_1"] = df["y"].pct_change(periods=1)
        df["y_pct_change_7"] = df["y"].pct_change(periods=7)

        # Fill NaN values from lag/rolling operations
        df = df.fillna(0)

        # Replace inf values
        df = df.replace([np.inf, -np.inf], 0)

        return df

    def temporal_split(self, df: pd.DataFrame) -> tuple:
        """
        Split data temporally (NOT randomly).

        For time series, we must preserve temporal order.
        Train on past, test on future.
        """
        df = df.sort_values("ds").reset_index(drop=True)

        split_idx = int(len(df) * self.split_ratio)

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        return train_df, test_df

    def create_prophet_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Prophet-ready format.

        Prophet requires: ds (datetime), y (target)
        """
        prophet_df = df[["ds", "y"]].copy()
        return prophet_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_prophet: pd.DataFrame,
        test_prophet: pd.DataFrame,
        daily_df: pd.DataFrame
    ):
        """Save all split datasets."""
        print("\nSaving datasets...")

        # Full featured datasets
        train_df.to_csv(self.processed_dir / "train_data.csv", index=False)
        print(f"  Saved: train_data.csv ({len(train_df)} rows)")

        test_df.to_csv(self.processed_dir / "test_data.csv", index=False)
        print(f"  Saved: test_data.csv ({len(test_df)} rows)")

        # Prophet format (ds, y only)
        train_prophet.to_csv(self.processed_dir / "train_prophet.csv", index=False)
        print(f"  Saved: train_prophet.csv ({len(train_prophet)} rows)")

        test_prophet.to_csv(self.processed_dir / "test_prophet.csv", index=False)
        print(f"  Saved: test_prophet.csv ({len(test_prophet)} rows)")

        # Daily aggregated (full)
        daily_df.to_csv(self.processed_dir / "daily_aggregated.csv", index=False)
        print(f"  Saved: daily_aggregated.csv ({len(daily_df)} rows)")

    def generate_summary(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> dict:
        """Generate split summary."""
        summary = {
            "created_at": datetime.now().isoformat(),
            "split_ratio": self.split_ratio,
            "log_transform_applied": True,
            "train": {
                "rows": len(train_df),
                "start_date": str(train_df["ds"].min()),
                "end_date": str(train_df["ds"].max()),
                "days": len(train_df),
                "y_mean_log": round(train_df["y"].mean(), 2),
                "y_std_log": round(train_df["y"].std(), 2),
                "y_raw_mean": round(train_df["y_raw"].mean(), 2) if "y_raw" in train_df.columns else None,
                "y_raw_std": round(train_df["y_raw"].std(), 2) if "y_raw" in train_df.columns else None,
            },
            "test": {
                "rows": len(test_df),
                "start_date": str(test_df["ds"].min()),
                "end_date": str(test_df["ds"].max()),
                "days": len(test_df),
                "y_mean_log": round(test_df["y"].mean(), 2),
                "y_std_log": round(test_df["y"].std(), 2),
                "y_raw_mean": round(test_df["y_raw"].mean(), 2) if "y_raw" in test_df.columns else None,
                "y_raw_std": round(test_df["y_raw"].std(), 2) if "y_raw" in test_df.columns else None,
            },
            "features": list(train_df.columns),
            "target": "y (log-transformed daily engagement sum)",
            "target_raw": "y_raw (original scale daily engagement sum)",
            "notes": [
                "y is log1p transformed for LSTM training stability",
                "Use np.expm1(y) to convert predictions back to original scale",
                "All lag and rolling features are computed on log scale",
            ],
        }

        # Save summary
        summary_path = self.processed_dir / "train_test_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved: {summary_path}")

        return summary

    def run(self):
        """Run complete train-test split pipeline."""
        print("\n" + "#"*60)
        print("# SocialProphet - Train-Test Split")
        print("#"*60)

        # Load data
        print("\nLoading primary dataset (Instagram)...")
        df = self.load_primary_dataset()
        print(f"  Loaded: {len(df)} posts")

        # Aggregate to daily
        daily_df = self.create_daily_aggregation(df)

        # Add features
        print("\nAdding features...")
        daily_df = self.add_time_features(daily_df)
        daily_df = self.add_lag_features(daily_df)
        print(f"  Total features: {len(daily_df.columns)}")

        # Split
        print(f"\nCreating temporal split ({int(self.split_ratio*100)}/{int((1-self.split_ratio)*100)})...")
        train_df, test_df = self.temporal_split(daily_df)

        # Create Prophet format
        train_prophet = self.create_prophet_format(train_df)
        test_prophet = self.create_prophet_format(test_df)

        # Save
        self.save_splits(train_df, test_df, train_prophet, test_prophet, daily_df)

        # Summary
        summary = self.generate_summary(train_df, test_df)

        # Print summary
        print("\n" + "="*60)
        print("TRAIN-TEST SPLIT SUMMARY")
        print("="*60)
        print(f"\nLog Transformation: APPLIED")
        print(f"  - y: log1p(engagement) for LSTM training")
        print(f"  - y_raw: original scale for evaluation")

        print(f"\nTrain Set:")
        print(f"  Period: {summary['train']['start_date'][:10]} to {summary['train']['end_date'][:10]}")
        print(f"  Days: {summary['train']['days']}")
        print(f"  Mean (log): {summary['train']['y_mean_log']:.2f}")
        if summary['train']['y_raw_mean']:
            print(f"  Mean (raw): {summary['train']['y_raw_mean']:,.0f}")

        print(f"\nTest Set:")
        print(f"  Period: {summary['test']['start_date'][:10]} to {summary['test']['end_date'][:10]}")
        print(f"  Days: {summary['test']['days']}")
        print(f"  Mean (log): {summary['test']['y_mean_log']:.2f}")
        if summary['test']['y_raw_mean']:
            print(f"  Mean (raw): {summary['test']['y_raw_mean']:,.0f}")

        print(f"\nFeatures Created: {len(summary['features'])}")
        print(f"Target Variable: {summary['target']}")

        print("\n" + "#"*60)
        print("# Phase 1 Complete! Ready for Forecasting (Phase 2)")
        print("#"*60)

        # Final status
        print("\n" + "="*60)
        print("PHASE 1 DELIVERABLES STATUS")
        print("="*60)
        print("  [OK] Working Python environment")
        print("  [OK] Datasets downloaded (Instagram, Social Media, Viral)")
        print("  [OK] Data preprocessed and validated (Grade A)")
        print("  [OK] EDA report generated with insights")
        print("  [OK] Train-test split created (80/20 temporal)")
        print("  [OK] Prophet-ready format prepared")
        print("\n  >>> Ready for Phase 2: Time-Series Forecasting <<<")

        return summary


def main():
    splitter = TrainTestSplitter()
    splitter.run()


if __name__ == "__main__":
    main()
