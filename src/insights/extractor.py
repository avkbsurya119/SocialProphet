"""
Insight Extraction Module for SocialProphet.

Transforms forecast predictions and historical data into actionable insights
for content generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class InsightExtractor:
    """
    Extract actionable insights from forecast data and historical patterns.

    Analyzes:
    - Temporal patterns (best posting times/days)
    - Trend direction and momentum
    - Content performance patterns
    - Seasonal patterns
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize InsightExtractor.

        Args:
            config: Configuration object (optional)
        """
        self.config = config or Config()
        self.insights = {}

    def extract_all(
        self,
        predictions_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        forecast_horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Extract all insights from predictions and historical data.

        Args:
            predictions_df: DataFrame with forecast predictions
            historical_df: Historical engagement data
            forecast_horizon: Number of days forecasted

        Returns:
            Dictionary containing all extracted insights
        """
        self.insights = {
            'temporal_patterns': self.extract_temporal_patterns(historical_df),
            'trend_analysis': self.extract_trend(predictions_df, historical_df),
            'content_patterns': self.extract_content_patterns(historical_df),
            'seasonality': self.extract_seasonality(historical_df),
            'predictions_summary': self.summarize_predictions(predictions_df),
            'recommendations': self.generate_recommendations(
                predictions_df, historical_df
            ),
            'metadata': {
                'extraction_date': datetime.now().isoformat(),
                'forecast_horizon': forecast_horizon,
                'historical_days': len(historical_df),
            }
        }

        return self.insights

    def extract_temporal_patterns(
        self,
        historical_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze temporal patterns to find best posting times.

        Args:
            historical_df: Historical engagement data

        Returns:
            Dictionary with temporal pattern insights
        """
        df = historical_df.copy()

        # Ensure date column exists
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
            df['day_of_week'] = df['ds'].dt.dayofweek
            df['day_name'] = df['ds'].dt.day_name()

        # Get engagement column (use y_raw if available, else y)
        engagement_col = 'y_raw' if 'y_raw' in df.columns else 'y'

        # Best days of week
        day_engagement = df.groupby('day_of_week')[engagement_col].agg(['mean', 'std', 'count'])
        day_engagement = day_engagement.reset_index()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_engagement['day_name'] = day_engagement['day_of_week'].map(lambda x: day_names[x])

        best_days = day_engagement.nlargest(3, 'mean')
        worst_days = day_engagement.nsmallest(2, 'mean')

        # Weekend vs Weekday comparison
        if 'is_weekend' in df.columns:
            weekend_avg = df[df['is_weekend'] == 1][engagement_col].mean()
            weekday_avg = df[df['is_weekend'] == 0][engagement_col].mean()
        else:
            weekend_avg = df[df['day_of_week'].isin([5, 6])][engagement_col].mean()
            weekday_avg = df[~df['day_of_week'].isin([5, 6])][engagement_col].mean()

        # Monthly patterns
        if 'month' in df.columns:
            monthly_engagement = df.groupby('month')[engagement_col].mean()
            best_month = monthly_engagement.idxmax()
            worst_month = monthly_engagement.idxmin()
        else:
            best_month = None
            worst_month = None

        return {
            'best_days': [
                {
                    'day': row['day_name'],
                    'day_of_week': int(row['day_of_week']),
                    'avg_engagement': float(row['mean']),
                    'std': float(row['std']) if pd.notna(row['std']) else 0.0
                }
                for _, row in best_days.iterrows()
            ],
            'worst_days': [
                {
                    'day': row['day_name'],
                    'day_of_week': int(row['day_of_week']),
                    'avg_engagement': float(row['mean'])
                }
                for _, row in worst_days.iterrows()
            ],
            'weekend_vs_weekday': {
                'weekend_avg': float(weekend_avg) if pd.notna(weekend_avg) else 0.0,
                'weekday_avg': float(weekday_avg) if pd.notna(weekday_avg) else 0.0,
                'better': 'weekend' if weekend_avg > weekday_avg else 'weekday',
                'difference_pct': float(abs(weekend_avg - weekday_avg) / weekday_avg * 100) if weekday_avg > 0 else 0.0
            },
            'best_month': int(best_month) if best_month else None,
            'worst_month': int(worst_month) if worst_month else None
        }

    def extract_trend(
        self,
        predictions_df: pd.DataFrame,
        historical_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze trend direction and momentum.

        Args:
            predictions_df: Forecast predictions
            historical_df: Historical data

        Returns:
            Dictionary with trend analysis
        """
        # Get engagement columns
        hist_col = 'y_raw' if 'y_raw' in historical_df.columns else 'y'
        pred_col = 'ensemble_pred_original' if 'ensemble_pred_original' in predictions_df.columns else 'ensemble_pred'

        if pred_col not in predictions_df.columns:
            # Try to find any prediction column
            pred_cols = [c for c in predictions_df.columns if 'pred' in c.lower()]
            pred_col = pred_cols[0] if pred_cols else hist_col

        # Historical statistics
        historical_mean = historical_df[hist_col].mean()
        historical_std = historical_df[hist_col].std()

        # Recent trend (last 7 days of historical)
        recent_7d = historical_df[hist_col].tail(7).mean()
        recent_14d = historical_df[hist_col].tail(14).mean()
        recent_30d = historical_df[hist_col].tail(30).mean()

        # Predicted trend
        if pred_col in predictions_df.columns:
            predicted_mean = predictions_df[pred_col].mean()
            predicted_first_week = predictions_df[pred_col].head(7).mean()
            predicted_last_week = predictions_df[pred_col].tail(7).mean()
        else:
            predicted_mean = historical_mean
            predicted_first_week = recent_7d
            predicted_last_week = recent_7d

        # Calculate momentum
        momentum_7d = ((recent_7d - recent_14d) / recent_14d * 100) if recent_14d > 0 else 0
        momentum_30d = ((recent_7d - recent_30d) / recent_30d * 100) if recent_30d > 0 else 0

        # Determine trend direction
        if momentum_7d > 5:
            trend_direction = 'increasing'
            trend_strength = 'strong' if momentum_7d > 15 else 'moderate'
        elif momentum_7d < -5:
            trend_direction = 'decreasing'
            trend_strength = 'strong' if momentum_7d < -15 else 'moderate'
        else:
            trend_direction = 'stable'
            trend_strength = 'weak'

        # Predicted change
        predicted_change = ((predicted_mean - historical_mean) / historical_mean * 100) if historical_mean > 0 else 0

        return {
            'direction': trend_direction,
            'strength': trend_strength,
            'momentum_7d': float(momentum_7d),
            'momentum_30d': float(momentum_30d),
            'historical_mean': float(historical_mean),
            'historical_std': float(historical_std),
            'recent_7d_avg': float(recent_7d),
            'recent_30d_avg': float(recent_30d),
            'predicted_mean': float(predicted_mean),
            'predicted_change_pct': float(predicted_change),
            'confidence': self._calculate_trend_confidence(historical_df, hist_col)
        }

    def _calculate_trend_confidence(
        self,
        df: pd.DataFrame,
        col: str
    ) -> str:
        """Calculate confidence level based on data consistency."""
        cv = df[col].std() / df[col].mean() if df[col].mean() > 0 else 1

        if cv < 0.2:
            return 'high'
        elif cv < 0.4:
            return 'medium'
        else:
            return 'low'

    def extract_content_patterns(
        self,
        historical_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze content patterns from historical data.

        Args:
            historical_df: Historical engagement data

        Returns:
            Dictionary with content pattern insights
        """
        engagement_col = 'y_raw' if 'y_raw' in historical_df.columns else 'y'

        # Calculate percentiles for engagement
        percentiles = historical_df[engagement_col].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()

        # High performing threshold (top 25%)
        high_threshold = percentiles[0.75]
        high_performers = historical_df[historical_df[engagement_col] >= high_threshold]

        # Low performing (bottom 25%)
        low_threshold = percentiles[0.25]
        low_performers = historical_df[historical_df[engagement_col] <= low_threshold]

        # Analyze high performers
        high_perf_analysis = {}
        if 'day_of_week' in high_performers.columns:
            high_perf_days = high_performers['day_of_week'].value_counts().head(3)
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            high_perf_analysis['common_days'] = [day_names[d] for d in high_perf_days.index.tolist()]

        # Engagement velocity (rate of change)
        if 'y_pct_change_1' in historical_df.columns:
            avg_velocity = historical_df['y_pct_change_1'].mean()
            velocity_std = historical_df['y_pct_change_1'].std()
        else:
            avg_velocity = 0
            velocity_std = 0

        return {
            'engagement_percentiles': {
                'p25': float(percentiles[0.25]),
                'p50': float(percentiles[0.5]),
                'p75': float(percentiles[0.75]),
                'p90': float(percentiles[0.9])
            },
            'high_performance_threshold': float(high_threshold),
            'low_performance_threshold': float(low_threshold),
            'high_performers_count': len(high_performers),
            'high_performers_analysis': high_perf_analysis,
            'engagement_velocity': {
                'avg_daily_change_pct': float(avg_velocity),
                'volatility': float(velocity_std)
            },
            'total_observations': len(historical_df)
        }

    def extract_seasonality(
        self,
        historical_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Extract seasonal patterns from historical data.

        Args:
            historical_df: Historical engagement data

        Returns:
            Dictionary with seasonality insights
        """
        df = historical_df.copy()
        engagement_col = 'y_raw' if 'y_raw' in df.columns else 'y'

        # Weekly seasonality
        weekly_pattern = {}
        if 'day_of_week' in df.columns:
            weekly = df.groupby('day_of_week')[engagement_col].mean()
            weekly_normalized = (weekly - weekly.min()) / (weekly.max() - weekly.min())
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_pattern = {
                day_names[i]: float(weekly_normalized.get(i, 0))
                for i in range(7)
            }

        # Monthly seasonality
        monthly_pattern = {}
        if 'month' in df.columns:
            monthly = df.groupby('month')[engagement_col].mean()
            if len(monthly) > 1:
                monthly_normalized = (monthly - monthly.min()) / (monthly.max() - monthly.min())
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_pattern = {
                    month_names[int(m)-1]: float(monthly_normalized.get(m, 0))
                    for m in monthly.index
                }

        # Detect weekly cycle strength
        if 'day_of_week' in df.columns:
            weekly_var = df.groupby('day_of_week')[engagement_col].mean().var()
            overall_var = df[engagement_col].var()
            weekly_strength = float(weekly_var / overall_var) if overall_var > 0 else 0
        else:
            weekly_strength = 0

        return {
            'weekly_pattern': weekly_pattern,
            'monthly_pattern': monthly_pattern,
            'weekly_cycle_strength': weekly_strength,
            'has_strong_weekly_pattern': weekly_strength > 0.1,
            'has_strong_monthly_pattern': len(monthly_pattern) > 1
        }

    def summarize_predictions(
        self,
        predictions_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Summarize forecast predictions.

        Args:
            predictions_df: Forecast predictions DataFrame

        Returns:
            Dictionary with prediction summary
        """
        # Find prediction column
        pred_col = None
        for col in ['ensemble_pred_original', 'ensemble_pred', 'yhat_original', 'yhat']:
            if col in predictions_df.columns:
                pred_col = col
                break

        if pred_col is None:
            return {'error': 'No prediction column found'}

        predictions = predictions_df[pred_col]

        # Peak predictions
        peak_idx = predictions.idxmax()
        trough_idx = predictions.idxmin()

        # Date handling
        if 'ds' in predictions_df.columns:
            dates = pd.to_datetime(predictions_df['ds'])
            peak_date = dates.iloc[peak_idx] if isinstance(peak_idx, int) else dates.loc[peak_idx]
            trough_date = dates.iloc[trough_idx] if isinstance(trough_idx, int) else dates.loc[trough_idx]
        else:
            peak_date = None
            trough_date = None

        return {
            'mean_predicted': float(predictions.mean()),
            'std_predicted': float(predictions.std()),
            'min_predicted': float(predictions.min()),
            'max_predicted': float(predictions.max()),
            'peak': {
                'value': float(predictions.max()),
                'date': peak_date.isoformat() if peak_date else None,
                'day_of_week': peak_date.strftime('%A') if peak_date else None
            },
            'trough': {
                'value': float(predictions.min()),
                'date': trough_date.isoformat() if trough_date else None,
                'day_of_week': trough_date.strftime('%A') if trough_date else None
            },
            'range': float(predictions.max() - predictions.min()),
            'coefficient_of_variation': float(predictions.std() / predictions.mean()) if predictions.mean() > 0 else 0
        }

    def generate_recommendations(
        self,
        predictions_df: pd.DataFrame,
        historical_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate actionable recommendations based on insights.

        Args:
            predictions_df: Forecast predictions
            historical_df: Historical data

        Returns:
            Dictionary with recommendations
        """
        # Get temporal patterns
        temporal = self.extract_temporal_patterns(historical_df)
        trend = self.extract_trend(predictions_df, historical_df)

        recommendations = {
            'posting_schedule': [],
            'content_strategy': [],
            'engagement_targets': {},
            'warnings': []
        }

        # Best posting days
        best_days = temporal.get('best_days', [])
        if best_days:
            recommendations['posting_schedule'].append({
                'type': 'best_days',
                'recommendation': f"Post on {', '.join([d['day'] for d in best_days[:2]])} for optimal engagement",
                'days': [d['day'] for d in best_days],
                'priority': 'high'
            })

        # Weekend vs Weekday
        weekend_info = temporal.get('weekend_vs_weekday', {})
        if weekend_info.get('difference_pct', 0) > 10:
            better = weekend_info.get('better', 'weekday')
            recommendations['posting_schedule'].append({
                'type': 'timing',
                'recommendation': f"{better.capitalize()}s show {weekend_info['difference_pct']:.1f}% higher engagement",
                'priority': 'medium'
            })

        # Trend-based recommendations
        if trend['direction'] == 'increasing':
            recommendations['content_strategy'].append({
                'type': 'trend',
                'recommendation': 'Engagement is trending up - maintain current content strategy',
                'priority': 'info'
            })
        elif trend['direction'] == 'decreasing':
            recommendations['content_strategy'].append({
                'type': 'trend',
                'recommendation': 'Engagement is declining - consider refreshing content approach',
                'priority': 'high'
            })
            recommendations['warnings'].append('Declining engagement trend detected')

        # Engagement targets
        hist_mean = trend.get('historical_mean', 0)
        recommendations['engagement_targets'] = {
            'minimum': float(hist_mean * 0.8),
            'target': float(hist_mean),
            'stretch': float(hist_mean * 1.2),
            'based_on': 'historical_average'
        }

        # Confidence warning
        if trend.get('confidence') == 'low':
            recommendations['warnings'].append(
                'High variability in historical data - predictions may be less reliable'
            )

        return recommendations

    def extract_peaks(
        self,
        predictions_df: pd.DataFrame,
        n_peaks: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Extract top N peak engagement days from predictions.

        Args:
            predictions_df: Forecast predictions
            n_peaks: Number of peaks to extract

        Returns:
            List of peak engagement days
        """
        pred_col = None
        for col in ['ensemble_pred_original', 'ensemble_pred', 'yhat']:
            if col in predictions_df.columns:
                pred_col = col
                break

        if pred_col is None:
            return []

        df = predictions_df.copy()
        df = df.nlargest(n_peaks, pred_col)

        peaks = []
        for idx, row in df.iterrows():
            peak = {
                'predicted_engagement': float(row[pred_col]),
                'rank': len(peaks) + 1
            }
            if 'ds' in row:
                date = pd.to_datetime(row['ds'])
                peak['date'] = date.isoformat()
                peak['day_of_week'] = date.strftime('%A')
            peaks.append(peak)

        return peaks

    def to_prompt_context(self) -> str:
        """
        Convert insights to a context string for LLM prompts.

        Returns:
            Formatted string with key insights
        """
        if not self.insights:
            return "No insights available."

        context_parts = []

        # Trend
        trend = self.insights.get('trend_analysis', {})
        context_parts.append(
            f"TREND: Engagement is {trend.get('direction', 'stable')} "
            f"({trend.get('strength', 'moderate')} {trend.get('momentum_7d', 0):.1f}% weekly change)"
        )

        # Best days
        temporal = self.insights.get('temporal_patterns', {})
        best_days = temporal.get('best_days', [])
        if best_days:
            days_str = ', '.join([d['day'] for d in best_days[:3]])
            context_parts.append(f"BEST POSTING DAYS: {days_str}")

        # Weekend vs Weekday
        ww = temporal.get('weekend_vs_weekday', {})
        if ww:
            context_parts.append(
                f"TIMING: {ww.get('better', 'weekday').capitalize()}s perform "
                f"{ww.get('difference_pct', 0):.1f}% better"
            )

        # Predictions
        preds = self.insights.get('predictions_summary', {})
        if preds and 'mean_predicted' in preds:
            context_parts.append(
                f"PREDICTED ENGAGEMENT: {preds['mean_predicted']:,.0f} average "
                f"(range: {preds.get('min_predicted', 0):,.0f} - {preds.get('max_predicted', 0):,.0f})"
            )

        # Peak
        peak = preds.get('peak', {})
        if peak and peak.get('date'):
            context_parts.append(
                f"PEAK DAY: {peak.get('day_of_week', 'Unknown')} with "
                f"{peak.get('value', 0):,.0f} expected engagement"
            )

        return '\n'.join(context_parts)

    def save_insights(self, filepath: Optional[Path] = None) -> Path:
        """
        Save insights to JSON file.

        Args:
            filepath: Output file path (optional)

        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = Path(self.config.PROCESSED_DATA_DIR) / 'insights.json'

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.insights, f, indent=2, default=str)

        print(f"Insights saved to: {filepath}")
        return filepath

    @classmethod
    def load_insights(cls, filepath: Path) -> Dict[str, Any]:
        """
        Load insights from JSON file.

        Args:
            filepath: Path to insights JSON file

        Returns:
            Dictionary of insights
        """
        with open(filepath, 'r') as f:
            return json.load(f)
