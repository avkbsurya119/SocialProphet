"""
Validate processed datasets and generate statistics report.

This script:
1. Validates all processed datasets
2. Checks data quality
3. Generates statistics report
4. Saves validation results

Usage:
    python scripts/validate_data.py
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


class DataValidator:
    """Validate processed datasets and generate reports."""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.processed_dir = Config.PROCESSED_DATA_DIR
        self.validation_results = {}

    def validate_dataset(self, filename: str) -> dict:
        """
        Validate a single dataset.

        Args:
            filename: Name of the CSV file in processed/

        Returns:
            Validation results dictionary
        """
        filepath = self.processed_dir / filename
        if not filepath.exists():
            return {"error": f"File not found: {filename}"}

        df = pd.read_csv(filepath)

        # Parse timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        results = {
            "filename": filename,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": list(df.columns),
            "validation": {},
            "statistics": {},
            "quality_score": 100,  # Start with perfect score
        }

        # 1. Check for required columns
        required_cols = ["timestamp", "engagement"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        results["validation"]["required_columns"] = {
            "passed": len(missing_cols) == 0,
            "missing": missing_cols,
        }
        if missing_cols:
            results["quality_score"] -= 20

        # 2. Check for missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
        results["validation"]["missing_values"] = {
            "passed": missing_pct < 5,
            "total_missing": int(total_missing),
            "percentage": round(missing_pct, 2),
            "by_column": {k: int(v) for k, v in missing_counts[missing_counts > 0].items()},
        }
        if missing_pct >= 5:
            results["quality_score"] -= 15

        # 3. Check date range
        if "timestamp" in df.columns:
            date_range_days = (df["timestamp"].max() - df["timestamp"].min()).days
            results["validation"]["date_range"] = {
                "passed": date_range_days >= 90,
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
                "days": date_range_days,
            }
            if date_range_days < 90:
                results["quality_score"] -= 10

        # 4. Check minimum rows
        min_rows = 500
        results["validation"]["minimum_rows"] = {
            "passed": len(df) >= min_rows,
            "count": len(df),
            "required": min_rows,
        }
        if len(df) < min_rows:
            results["quality_score"] -= 15

        # 5. Check for duplicates
        if "post_id" in df.columns:
            duplicates = df["post_id"].duplicated().sum()
        else:
            duplicates = df.duplicated().sum()
        results["validation"]["duplicates"] = {
            "passed": duplicates == 0,
            "count": int(duplicates),
        }
        if duplicates > 0:
            results["quality_score"] -= 5

        # 6. Check engagement values
        if "engagement" in df.columns:
            negative_engagement = (df["engagement"] < 0).sum()
            zero_engagement = (df["engagement"] == 0).sum()
            results["validation"]["engagement_values"] = {
                "passed": negative_engagement == 0,
                "negative_count": int(negative_engagement),
                "zero_count": int(zero_engagement),
                "zero_percentage": round((zero_engagement / len(df)) * 100, 2),
            }
            if negative_engagement > 0:
                results["quality_score"] -= 10

        # Generate statistics
        if "engagement" in df.columns:
            results["statistics"]["engagement"] = {
                "mean": round(df["engagement"].mean(), 2),
                "median": round(df["engagement"].median(), 2),
                "std": round(df["engagement"].std(), 2),
                "min": int(df["engagement"].min()),
                "max": int(df["engagement"].max()),
                "q25": round(df["engagement"].quantile(0.25), 2),
                "q75": round(df["engagement"].quantile(0.75), 2),
            }

        # Engagement by time
        if "timestamp" in df.columns and "engagement" in df.columns:
            hourly = df.groupby(df["timestamp"].dt.hour)["engagement"].mean()
            results["statistics"]["hourly_engagement"] = {
                "best_hour": int(hourly.idxmax()),
                "worst_hour": int(hourly.idxmin()),
                "best_hour_avg": round(hourly.max(), 2),
                "worst_hour_avg": round(hourly.min(), 2),
            }

        # Platform distribution
        if "platform" in df.columns:
            platform_counts = df["platform"].value_counts().to_dict()
            results["statistics"]["platform_distribution"] = platform_counts

        # Final quality assessment
        results["quality_score"] = max(0, results["quality_score"])
        results["quality_grade"] = self._get_grade(results["quality_score"])

        return results

    def _get_grade(self, score: int) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def validate_all(self) -> dict:
        """Validate all processed datasets."""
        print("\n" + "#"*60)
        print("# SocialProphet - Data Validation")
        print("#"*60)

        datasets = [
            "instagram_cleaned.csv",
            "social_media_cleaned.csv",
            "viral_cleaned.csv",
            "combined_data.csv",
        ]

        all_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "datasets": {},
            "summary": {},
        }

        for dataset in datasets:
            print(f"\nValidating: {dataset}")
            results = self.validate_dataset(dataset)
            all_results["datasets"][dataset] = results

            if "error" not in results:
                grade = results["quality_grade"]
                score = results["quality_score"]
                rows = results["shape"]["rows"]
                print(f"  Rows: {rows:,}")
                print(f"  Quality Score: {score}/100 (Grade: {grade})")

                # Print any failed validations
                for check, result in results["validation"].items():
                    if not result.get("passed", True):
                        print(f"  Warning: {check} check failed")

        # Generate summary
        total_rows = sum(
            r["shape"]["rows"]
            for r in all_results["datasets"].values()
            if "error" not in r and r["filename"] != "combined_data.csv"
        )
        avg_score = np.mean([
            r["quality_score"]
            for r in all_results["datasets"].values()
            if "error" not in r
        ])

        all_results["summary"] = {
            "total_datasets": len(datasets),
            "total_rows_individual": total_rows,
            "combined_rows": all_results["datasets"].get("combined_data.csv", {}).get("shape", {}).get("rows", 0),
            "average_quality_score": round(avg_score, 1),
            "overall_grade": self._get_grade(int(avg_score)),
        }

        # Save results
        output_path = self.processed_dir / "validation_report.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nValidation report saved: {output_path}")

        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Individual Rows: {total_rows:,}")
        print(f"Combined Dataset Rows: {all_results['summary']['combined_rows']:,}")
        print(f"Average Quality Score: {avg_score:.1f}/100")
        print(f"Overall Grade: {all_results['summary']['overall_grade']}")

        return all_results

    def print_detailed_stats(self):
        """Print detailed statistics for primary dataset."""
        print("\n" + "="*60)
        print("DETAILED STATISTICS - Instagram Dataset (Primary)")
        print("="*60)

        filepath = self.processed_dir / "instagram_cleaned.csv"
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        print(f"\nDataset Shape: {df.shape}")
        print(f"\nEngagement Distribution:")
        print(df["engagement"].describe())

        print(f"\nContent Type Distribution:")
        if "content_type" in df.columns:
            print(df["content_type"].value_counts())

        print(f"\nEngagement by Day of Week:")
        if "day_of_week" in df.columns:
            daily = df.groupby("day_of_week")["engagement"].mean().sort_values(ascending=False)
            for day, eng in daily.items():
                print(f"  {day}: {eng:.1f}")

        print(f"\nTop 5 Engagement Hours:")
        if "hour" in df.columns:
            hourly = df.groupby("hour")["engagement"].mean().sort_values(ascending=False).head(5)
            for hour, eng in hourly.items():
                print(f"  {hour}:00 - {eng:.1f} avg engagement")


def main():
    validator = DataValidator()
    validator.validate_all()
    validator.print_detailed_stats()
    print("\n" + "#"*60)
    print("# Validation Complete!")
    print("#"*60)


if __name__ == "__main__":
    main()
