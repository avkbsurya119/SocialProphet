"""
Stationarity Analysis Module for SocialProphet.

Performs ADF (Augmented Dickey-Fuller) and KPSS tests to determine
if the time series is stationary before applying forecasting models.

Stationarity is required for ARIMA-based models. This module:
1. Runs ADF and KPSS tests
2. Determines optimal differencing order
3. Saves comprehensive stationarity report
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

from statsmodels.tsa.stattools import adfuller, kpss

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class StationarityAnalyzer:
    """
    Analyze time series stationarity for forecasting model selection.

    Uses ADF and KPSS tests to determine stationarity:
    - ADF: Null hypothesis = series has unit root (non-stationary)
    - KPSS: Null hypothesis = series is stationary

    Interpretation:
    - ADF rejects + KPSS fails to reject = Stationary
    - ADF fails + KPSS rejects = Non-stationary
    - Both reject = Trend stationary
    - Neither rejects = Difference stationary
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize analyzer.

        Args:
            significance_level: p-value threshold for hypothesis tests (default 0.05)
        """
        self.significance_level = significance_level
        self.results = {}

    def adf_test(
        self,
        series: pd.Series,
        regression: str = 'c',
        maxlag: Optional[int] = None
    ) -> Dict:
        """
        Perform Augmented Dickey-Fuller test.

        Null Hypothesis: Series has a unit root (non-stationary)
        Alternative: Series is stationary

        If p-value < significance_level, reject null -> series is stationary

        Args:
            series: Time series data
            regression: 'c' (constant), 'ct' (constant+trend), 'n' (none)
            maxlag: Maximum lag for test (None = auto)

        Returns:
            dict with test results
        """
        # Remove NaN values
        clean_series = series.dropna()

        if len(clean_series) < 20:
            return {
                "error": "Series too short for ADF test (need at least 20 observations)",
                "is_stationary": None
            }

        try:
            result = adfuller(clean_series, regression=regression, maxlag=maxlag)

            adf_output = {
                "test_statistic": float(result[0]),
                "p_value": float(result[1]),
                "lags_used": int(result[2]),
                "n_observations": int(result[3]),
                "critical_values": {
                    "1%": float(result[4]['1%']),
                    "5%": float(result[4]['5%']),
                    "10%": float(result[4]['10%']),
                },
                "is_stationary": result[1] < self.significance_level,
                "regression": regression,
                "interpretation": self._interpret_adf(result[1], result[0], result[4])
            }

            return adf_output

        except Exception as e:
            return {
                "error": str(e),
                "is_stationary": None
            }

    def _interpret_adf(
        self,
        p_value: float,
        test_stat: float,
        critical_values: Dict
    ) -> str:
        """Generate human-readable ADF interpretation."""
        if p_value < 0.01:
            return "Strong evidence against unit root (stationary at 1% level)"
        elif p_value < 0.05:
            return "Evidence against unit root (stationary at 5% level)"
        elif p_value < 0.10:
            return "Weak evidence against unit root (stationary at 10% level)"
        else:
            return "Cannot reject unit root (series appears non-stationary)"

    def kpss_test(
        self,
        series: pd.Series,
        regression: str = 'c',
        nlags: str = 'auto'
    ) -> Dict:
        """
        Perform KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test.

        Null Hypothesis: Series is stationary
        Alternative: Series has a unit root (non-stationary)

        If p-value < significance_level, reject null -> series is non-stationary

        Args:
            series: Time series data
            regression: 'c' (level stationary), 'ct' (trend stationary)
            nlags: Number of lags ('auto' or integer)

        Returns:
            dict with test results
        """
        # Remove NaN values
        clean_series = series.dropna()

        if len(clean_series) < 20:
            return {
                "error": "Series too short for KPSS test",
                "is_stationary": None
            }

        try:
            # KPSS may raise warnings for boundary p-values
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = kpss(clean_series, regression=regression, nlags=nlags)

            kpss_output = {
                "test_statistic": float(result[0]),
                "p_value": float(result[1]),
                "lags_used": int(result[2]),
                "critical_values": {
                    "1%": float(result[3]['1%']),
                    "2.5%": float(result[3]['2.5%']),
                    "5%": float(result[3]['5%']),
                    "10%": float(result[3]['10%']),
                },
                # For KPSS, rejecting null means NON-stationary
                "is_stationary": result[1] >= self.significance_level,
                "regression": regression,
                "interpretation": self._interpret_kpss(result[1])
            }

            return kpss_output

        except Exception as e:
            return {
                "error": str(e),
                "is_stationary": None
            }

    def _interpret_kpss(self, p_value: float) -> str:
        """Generate human-readable KPSS interpretation."""
        if p_value < 0.01:
            return "Strong evidence for unit root (non-stationary at 1% level)"
        elif p_value < 0.05:
            return "Evidence for unit root (non-stationary at 5% level)"
        elif p_value < 0.10:
            return "Weak evidence for unit root (non-stationary at 10% level)"
        else:
            return "Cannot reject stationarity (series appears stationary)"

    def analyze(
        self,
        series: pd.Series,
        name: str = "series"
    ) -> Dict:
        """
        Run complete stationarity analysis.

        Combines ADF and KPSS results to determine:
        - Whether series is stationary
        - What type of stationarity/non-stationarity
        - Recommended differencing order

        Args:
            series: Time series data
            name: Name for the series in report

        Returns:
            Comprehensive analysis report
        """
        print(f"\nAnalyzing stationarity for: {name}")
        print("=" * 50)

        # Run both tests
        adf_result = self.adf_test(series)
        kpss_result = self.kpss_test(series)

        # Determine combined interpretation
        adf_stationary = adf_result.get('is_stationary')
        kpss_stationary = kpss_result.get('is_stationary')

        if adf_stationary is None or kpss_stationary is None:
            combined_result = "inconclusive"
            recommendation = "Check data quality - tests could not complete"
        elif adf_stationary and kpss_stationary:
            combined_result = "stationary"
            recommendation = "Series is stationary. No differencing needed (d=0)."
        elif not adf_stationary and not kpss_stationary:
            combined_result = "non_stationary"
            recommendation = "Series is non-stationary. Apply differencing (d=1 or d=2)."
        elif adf_stationary and not kpss_stationary:
            combined_result = "trend_stationary"
            recommendation = "Series is trend-stationary. Consider detrending or differencing."
        else:  # not adf_stationary and kpss_stationary
            combined_result = "difference_stationary"
            recommendation = "Series is difference-stationary. Apply first differencing (d=1)."

        # Get differencing recommendation
        diff_analysis = self.differencing_analysis(series)

        # Build complete report
        analysis = {
            "series_name": name,
            "n_observations": len(series.dropna()),
            "analyzed_at": datetime.now().isoformat(),
            "significance_level": self.significance_level,
            "adf_test": adf_result,
            "kpss_test": kpss_result,
            "combined_result": combined_result,
            "is_stationary": combined_result == "stationary",
            "recommendation": recommendation,
            "differencing_analysis": diff_analysis,
            "suggested_d": diff_analysis.get("optimal_d", 0),
        }

        self.results[name] = analysis

        # Print summary
        print(f"\nADF Test:")
        print(f"  Statistic: {adf_result.get('test_statistic', 'N/A'):.4f}")
        print(f"  P-value: {adf_result.get('p_value', 'N/A'):.4f}")
        print(f"  Stationary: {adf_result.get('is_stationary', 'N/A')}")

        print(f"\nKPSS Test:")
        print(f"  Statistic: {kpss_result.get('test_statistic', 'N/A'):.4f}")
        print(f"  P-value: {kpss_result.get('p_value', 'N/A'):.4f}")
        print(f"  Stationary: {kpss_result.get('is_stationary', 'N/A')}")

        print(f"\nCombined Result: {combined_result.upper()}")
        print(f"Recommendation: {recommendation}")
        print(f"Suggested differencing order (d): {analysis['suggested_d']}")

        return analysis

    def differencing_analysis(
        self,
        series: pd.Series,
        max_d: int = 2
    ) -> Dict:
        """
        Determine optimal differencing order.

        Tests d=0, 1, 2 and returns optimal d where series becomes stationary.

        Args:
            series: Time series data
            max_d: Maximum differencing order to test

        Returns:
            dict with differencing analysis
        """
        results = {}
        optimal_d = 0

        current_series = series.dropna()

        for d in range(max_d + 1):
            if d > 0:
                current_series = current_series.diff().dropna()

            if len(current_series) < 20:
                results[f"d={d}"] = {"error": "Series too short after differencing"}
                break

            adf_result = self.adf_test(current_series)

            results[f"d={d}"] = {
                "adf_statistic": adf_result.get("test_statistic"),
                "adf_pvalue": adf_result.get("p_value"),
                "is_stationary": adf_result.get("is_stationary"),
                "n_observations": len(current_series)
            }

            # If stationary, this is our optimal d
            if adf_result.get("is_stationary") and optimal_d == 0:
                optimal_d = d

        return {
            "tests": results,
            "optimal_d": optimal_d,
            "recommendation": f"Use d={optimal_d} for ARIMA model"
        }

    def save_report(
        self,
        filepath: Optional[Path] = None
    ) -> Path:
        """
        Save stationarity report as JSON.

        Args:
            filepath: Output path (default: data/processed/stationarity_report.json)

        Returns:
            Path to saved file
        """
        if filepath is None:
            Config.ensure_directories()
            filepath = Config.PROCESSED_DATA_DIR / "stationarity_report.json"

        report = {
            "generated_at": datetime.now().isoformat(),
            "significance_level": self.significance_level,
            "analyses": self.results
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\nStationarity report saved: {filepath}")
        return filepath


def run_stationarity_analysis():
    """Run stationarity analysis on training data."""
    print("\n" + "#" * 60)
    print("# SocialProphet - Stationarity Analysis")
    print("#" * 60)

    # Load training data
    train_path = Config.PROCESSED_DATA_DIR / "train_data.csv"
    train_df = pd.read_csv(train_path)

    print(f"\nLoaded training data: {len(train_df)} rows")

    # Initialize analyzer
    analyzer = StationarityAnalyzer(significance_level=0.05)

    # Analyze log-transformed target
    print("\n" + "=" * 60)
    print("Analyzing: y (log-transformed engagement)")
    print("=" * 60)
    analyzer.analyze(train_df['y'], name='y_log_engagement')

    # Also analyze raw target for comparison
    if 'y_raw' in train_df.columns:
        print("\n" + "=" * 60)
        print("Analyzing: y_raw (original scale engagement)")
        print("=" * 60)
        analyzer.analyze(train_df['y_raw'], name='y_raw_engagement')

    # Save report
    report_path = analyzer.save_report()

    print("\n" + "#" * 60)
    print("# Stationarity Analysis Complete!")
    print("#" * 60)

    return analyzer.results


if __name__ == "__main__":
    run_stationarity_analysis()
