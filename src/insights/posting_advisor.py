"""
Posting Time Advisor for SocialProphet.

Analyzes historical engagement data to recommend optimal posting times,
content types, and strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class PostingAdvisor:
    """
    Analyzes engagement data to provide actionable posting recommendations.

    Key outputs:
    - Best hours to post (by day)
    - Best content types (reels, images, carousels)
    - Best categories for different times
    - Personalized posting schedule
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.analysis = {}
        self.raw_data = None

    def load_instagram_data(self) -> pd.DataFrame:
        """Load raw Instagram analytics data."""
        path = self.config.RAW_DATA_DIR / "Instagram_Analytics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Instagram data not found at {path}")

        df = pd.read_csv(path)
        df['post_datetime'] = pd.to_datetime(df['post_datetime'])

        # Calculate engagement score
        df['engagement'] = (
            df['likes'] +
            df['comments'] * 2 +
            df['shares'] * 3 +
            df['saves'] * 2
        )

        # Engagement rate relative to reach
        df['engagement_rate'] = df['engagement'] / df['reach'].clip(lower=1)

        self.raw_data = df
        return df

    def analyze_all(self) -> Dict[str, Any]:
        """Run all analyses and return comprehensive insights."""
        if self.raw_data is None:
            self.load_instagram_data()

        self.analysis = {
            'best_hours': self._analyze_best_hours(),
            'best_days': self._analyze_best_days(),
            'best_hour_by_day': self._analyze_hour_by_day(),
            'content_performance': self._analyze_content_types(),
            'category_insights': self._analyze_categories(),
            'traffic_sources': self._analyze_traffic_sources(),
            'optimal_schedule': self._generate_optimal_schedule(),
            'quick_wins': self._generate_quick_wins(),
            'metadata': {
                'total_posts': len(self.raw_data),
                'date_range': {
                    'start': self.raw_data['post_datetime'].min().isoformat(),
                    'end': self.raw_data['post_datetime'].max().isoformat()
                },
                'analysis_date': datetime.now().isoformat()
            }
        }

        return self.analysis

    def _analyze_best_hours(self) -> Dict[str, Any]:
        """Find the best hours to post based on engagement."""
        df = self.raw_data

        # Group by hour
        hourly = df.groupby('post_hour').agg({
            'engagement': ['mean', 'median', 'std', 'count'],
            'engagement_rate': 'mean',
            'likes': 'mean',
            'comments': 'mean',
            'reach': 'mean'
        }).round(2)

        hourly.columns = ['_'.join(col) for col in hourly.columns]
        hourly = hourly.reset_index()

        # Rank hours
        hourly['rank'] = hourly['engagement_mean'].rank(ascending=False)

        # Find top and bottom hours
        top_hours = hourly.nsmallest(5, 'rank')
        bottom_hours = hourly.nlargest(3, 'rank')

        # Format results
        top_hours_list = []
        for _, row in top_hours.iterrows():
            hour = int(row['post_hour'])
            top_hours_list.append({
                'hour': hour,
                'time_12h': self._format_hour(hour),
                'avg_engagement': float(row['engagement_mean']),
                'avg_engagement_rate': float(row['engagement_rate_mean']),
                'sample_size': int(row['engagement_count']),
                'recommendation': self._get_hour_recommendation(hour)
            })

        return {
            'top_5_hours': top_hours_list,
            'worst_3_hours': [
                {
                    'hour': int(row['post_hour']),
                    'time_12h': self._format_hour(int(row['post_hour'])),
                    'avg_engagement': float(row['engagement_mean'])
                }
                for _, row in bottom_hours.iterrows()
            ],
            'hourly_distribution': hourly.to_dict('records')
        }

    def _analyze_best_days(self) -> Dict[str, Any]:
        """Find the best days to post."""
        df = self.raw_data

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        daily = df.groupby('day_of_week').agg({
            'engagement': ['mean', 'median', 'count'],
            'engagement_rate': 'mean',
            'reach': 'mean'
        }).round(2)

        daily.columns = ['_'.join(col) for col in daily.columns]
        daily = daily.reset_index()

        # Sort by day order
        daily['day_order'] = daily['day_of_week'].map({d: i for i, d in enumerate(day_order)})
        daily = daily.sort_values('day_order')

        # Rank by engagement
        daily['rank'] = daily['engagement_mean'].rank(ascending=False)

        top_days = daily.nsmallest(3, 'rank')

        return {
            'top_3_days': [
                {
                    'day': row['day_of_week'],
                    'avg_engagement': float(row['engagement_mean']),
                    'avg_engagement_rate': float(row['engagement_rate_mean']),
                    'sample_size': int(row['engagement_count'])
                }
                for _, row in top_days.iterrows()
            ],
            'daily_breakdown': [
                {
                    'day': row['day_of_week'],
                    'avg_engagement': float(row['engagement_mean']),
                    'rank': int(row['rank'])
                }
                for _, row in daily.iterrows()
            ],
            'weekend_vs_weekday': self._compare_weekend_weekday()
        }

    def _compare_weekend_weekday(self) -> Dict[str, Any]:
        """Compare weekend vs weekday performance."""
        df = self.raw_data

        weekend = df[df['day_of_week'].isin(['Saturday', 'Sunday'])]
        weekday = df[~df['day_of_week'].isin(['Saturday', 'Sunday'])]

        weekend_avg = weekend['engagement'].mean()
        weekday_avg = weekday['engagement'].mean()

        diff_pct = ((weekend_avg - weekday_avg) / weekday_avg * 100) if weekday_avg > 0 else 0

        return {
            'weekend_avg': float(weekend_avg),
            'weekday_avg': float(weekday_avg),
            'difference_pct': float(diff_pct),
            'recommendation': 'weekend' if diff_pct > 5 else 'weekday' if diff_pct < -5 else 'both'
        }

    def _analyze_hour_by_day(self) -> Dict[str, Any]:
        """Find best hours for each day of the week."""
        df = self.raw_data

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        best_hours_per_day = {}

        for day in day_order:
            day_data = df[df['day_of_week'] == day]
            if len(day_data) < 5:
                continue

            hourly = day_data.groupby('post_hour')['engagement'].mean()

            if len(hourly) > 0:
                top_hours = hourly.nlargest(3)
                best_hours_per_day[day] = {
                    'best_hours': [
                        {
                            'hour': int(h),
                            'time_12h': self._format_hour(int(h)),
                            'avg_engagement': float(hourly[h])
                        }
                        for h in top_hours.index
                    ],
                    'sample_size': len(day_data)
                }

        return best_hours_per_day

    def _analyze_content_types(self) -> Dict[str, Any]:
        """Analyze performance by media type."""
        df = self.raw_data

        media_perf = df.groupby('media_type').agg({
            'engagement': ['mean', 'median', 'count'],
            'engagement_rate': 'mean',
            'reach': 'mean',
            'likes': 'mean',
            'comments': 'mean',
            'shares': 'mean',
            'saves': 'mean'
        }).round(2)

        media_perf.columns = ['_'.join(col) for col in media_perf.columns]
        media_perf = media_perf.reset_index()
        media_perf['rank'] = media_perf['engagement_mean'].rank(ascending=False)

        best_type = media_perf.loc[media_perf['rank'].idxmin()]

        return {
            'best_media_type': {
                'type': best_type['media_type'],
                'avg_engagement': float(best_type['engagement_mean']),
                'avg_engagement_rate': float(best_type['engagement_rate_mean'])
            },
            'all_types': [
                {
                    'type': row['media_type'],
                    'avg_engagement': float(row['engagement_mean']),
                    'avg_engagement_rate': float(row['engagement_rate_mean']),
                    'avg_reach': float(row['reach_mean']),
                    'sample_size': int(row['engagement_count']),
                    'rank': int(row['rank'])
                }
                for _, row in media_perf.sort_values('rank').iterrows()
            ],
            'recommendation': f"Focus on {best_type['media_type']}s - they get {best_type['engagement_mean']:.0f} avg engagement"
        }

    def _analyze_categories(self) -> Dict[str, Any]:
        """Analyze performance by content category."""
        df = self.raw_data

        cat_perf = df.groupby('content_category').agg({
            'engagement': ['mean', 'count'],
            'engagement_rate': 'mean'
        }).round(2)

        cat_perf.columns = ['_'.join(col) for col in cat_perf.columns]
        cat_perf = cat_perf.reset_index()

        # Filter categories with enough data
        cat_perf = cat_perf[cat_perf['engagement_count'] >= 10]
        cat_perf['rank'] = cat_perf['engagement_mean'].rank(ascending=False)

        top_categories = cat_perf.nsmallest(5, 'rank')

        return {
            'top_categories': [
                {
                    'category': row['content_category'],
                    'avg_engagement': float(row['engagement_mean']),
                    'sample_size': int(row['engagement_count'])
                }
                for _, row in top_categories.iterrows()
            ],
            'all_categories': cat_perf.to_dict('records')
        }

    def _analyze_traffic_sources(self) -> Dict[str, Any]:
        """Analyze which traffic sources drive most engagement."""
        df = self.raw_data

        source_perf = df.groupby('traffic_source').agg({
            'engagement': ['mean', 'count'],
            'reach': 'mean'
        }).round(2)

        source_perf.columns = ['_'.join(col) for col in source_perf.columns]
        source_perf = source_perf.reset_index()
        source_perf['rank'] = source_perf['engagement_mean'].rank(ascending=False)

        return {
            'by_source': [
                {
                    'source': row['traffic_source'],
                    'avg_engagement': float(row['engagement_mean']),
                    'avg_reach': float(row['reach_mean']),
                    'sample_size': int(row['engagement_count'])
                }
                for _, row in source_perf.sort_values('rank').iterrows()
            ]
        }

    def _generate_optimal_schedule(self) -> Dict[str, Any]:
        """Generate an optimal weekly posting schedule."""
        best_hour_by_day = self._analyze_hour_by_day()
        best_days_data = self._analyze_best_days()
        content_perf = self._analyze_content_types()

        # Get top 3 days
        top_days = [d['day'] for d in best_days_data['top_3_days']]

        schedule = []

        for day in top_days:
            if day in best_hour_by_day:
                best_hour = best_hour_by_day[day]['best_hours'][0]
                schedule.append({
                    'day': day,
                    'time': best_hour['time_12h'],
                    'hour_24': best_hour['hour'],
                    'expected_engagement': best_hour['avg_engagement'],
                    'content_type': content_perf['best_media_type']['type']
                })

        return {
            'recommended_posts_per_week': len(top_days),
            'schedule': schedule,
            'summary': self._generate_schedule_summary(schedule)
        }

    def _generate_schedule_summary(self, schedule: List[Dict]) -> str:
        """Generate a human-readable schedule summary."""
        if not schedule:
            return "Unable to generate schedule - insufficient data"

        parts = []
        for slot in schedule:
            parts.append(f"{slot['day']} at {slot['time']}")

        return f"Post on: {', '.join(parts)}"

    def _generate_quick_wins(self) -> List[Dict[str, str]]:
        """Generate actionable quick wins based on the analysis."""
        quick_wins = []

        # Get data directly (not from self.analysis to avoid recursion)
        best_hours_data = self._analyze_best_hours()
        content_data = self._analyze_content_types()
        best_days_data = self._analyze_best_days()

        # Best posting time
        best_hours = best_hours_data.get('top_5_hours', [])
        if best_hours:
            best = best_hours[0]
            quick_wins.append({
                'title': 'Optimal Posting Time',
                'insight': f"Post at {best['time_12h']} for maximum engagement",
                'impact': f"+{int((best['avg_engagement'] / self.raw_data['engagement'].mean() - 1) * 100)}% vs average",
                'priority': 'high'
            })

        # Best content type
        if content_data.get('best_media_type'):
            media = content_data['best_media_type']
            quick_wins.append({
                'title': 'Best Content Format',
                'insight': f"Create more {media['type']}s",
                'impact': f"{media['avg_engagement']:.0f} avg engagement",
                'priority': 'high'
            })

        # Best day
        best_days = best_days_data.get('top_3_days', [])
        if best_days:
            day = best_days[0]
            quick_wins.append({
                'title': 'Best Day to Post',
                'insight': f"Prioritize posting on {day['day']}",
                'impact': f"{day['avg_engagement']:.0f} avg engagement",
                'priority': 'medium'
            })

        # Avoid worst hours
        worst_hours = best_hours_data.get('worst_3_hours', [])
        if worst_hours:
            avoid_times = [h['time_12h'] for h in worst_hours[:2]]
            quick_wins.append({
                'title': 'Times to Avoid',
                'insight': f"Avoid posting at {', '.join(avoid_times)}",
                'impact': 'Low engagement periods',
                'priority': 'medium'
            })

        return quick_wins

    def _format_hour(self, hour: int) -> str:
        """Format hour as 12-hour time."""
        if hour == 0:
            return "12:00 AM"
        elif hour < 12:
            return f"{hour}:00 AM"
        elif hour == 12:
            return "12:00 PM"
        else:
            return f"{hour - 12}:00 PM"

    def _get_hour_recommendation(self, hour: int) -> str:
        """Get recommendation text for an hour."""
        if 6 <= hour <= 9:
            return "Morning commute - good for catching early scrollers"
        elif 10 <= hour <= 11:
            return "Mid-morning - work break engagement"
        elif 12 <= hour <= 14:
            return "Lunch hour peak - high activity"
        elif 15 <= hour <= 17:
            return "Afternoon - steady engagement"
        elif 18 <= hour <= 21:
            return "Evening prime time - peak daily usage"
        elif 22 <= hour <= 23:
            return "Late night - dedicated scrollers"
        else:
            return "Off-peak hours"

    def get_posting_recommendation(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a specific posting recommendation.

        Args:
            category: Optional content category to filter for

        Returns:
            Recommendation with when, what, and expected outcome
        """
        if not self.analysis:
            self.analyze_all()

        schedule = self.analysis.get('optimal_schedule', {}).get('schedule', [])
        content = self.analysis.get('content_performance', {})

        if not schedule:
            return {'error': 'No schedule data available'}

        next_slot = schedule[0]

        recommendation = {
            'when': {
                'day': next_slot['day'],
                'time': next_slot['time'],
                'reasoning': 'Based on historical engagement patterns'
            },
            'what': {
                'format': content.get('best_media_type', {}).get('type', 'reel'),
                'reasoning': 'This format has highest engagement'
            },
            'expected_engagement': next_slot['expected_engagement'],
            'confidence': 'high' if len(self.raw_data) > 500 else 'medium'
        }

        return recommendation

    def to_prompt_context(self) -> str:
        """Convert analysis to context string for LLM prompts."""
        if not self.analysis:
            self.analyze_all()

        lines = []
        lines.append("=== POSTING INSIGHTS ===")

        # Quick wins
        quick_wins = self.analysis.get('quick_wins', [])
        if quick_wins:
            lines.append("\nTOP RECOMMENDATIONS:")
            for i, win in enumerate(quick_wins[:3], 1):
                lines.append(f"{i}. {win['title']}: {win['insight']} ({win['impact']})")

        # Schedule
        schedule = self.analysis.get('optimal_schedule', {})
        if schedule.get('summary'):
            lines.append(f"\nOPTIMAL SCHEDULE: {schedule['summary']}")

        # Best content
        content = self.analysis.get('content_performance', {})
        if content.get('best_media_type'):
            media = content['best_media_type']
            lines.append(f"\nBEST FORMAT: {media['type']} ({media['avg_engagement']:.0f} avg engagement)")

        # Top categories
        categories = self.analysis.get('category_insights', {}).get('top_categories', [])
        if categories:
            cat_names = [c['category'] for c in categories[:3]]
            lines.append(f"\nTOP CATEGORIES: {', '.join(cat_names)}")

        return '\n'.join(lines)

    def save_analysis(self, filepath: Optional[Path] = None) -> Path:
        """Save analysis to JSON file."""
        if not self.analysis:
            self.analyze_all()

        if filepath is None:
            filepath = self.config.PROCESSED_DATA_DIR / 'posting_insights.json'

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.analysis, f, indent=2, default=str)

        print(f"Analysis saved to: {filepath}")
        return filepath


def main():
    """Run posting analysis and display results."""
    print("=" * 60)
    print("POSTING TIME ANALYSIS")
    print("=" * 60)

    advisor = PostingAdvisor()

    try:
        advisor.load_instagram_data()
        print(f"\nLoaded {len(advisor.raw_data)} posts")

        analysis = advisor.analyze_all()

        # Print results
        print("\n" + "=" * 60)
        print("QUICK WINS")
        print("=" * 60)
        for win in analysis['quick_wins']:
            print(f"\n[{win['priority'].upper()}] {win['title']}")
            print(f"  → {win['insight']}")
            print(f"  Impact: {win['impact']}")

        print("\n" + "=" * 60)
        print("OPTIMAL SCHEDULE")
        print("=" * 60)
        schedule = analysis['optimal_schedule']
        print(f"\n{schedule['summary']}")
        print(f"\nRecommended posts per week: {schedule['recommended_posts_per_week']}")

        print("\n" + "=" * 60)
        print("BEST POSTING HOURS")
        print("=" * 60)
        for hour_info in analysis['best_hours']['top_5_hours'][:3]:
            print(f"\n{hour_info['time_12h']}: {hour_info['avg_engagement']:.0f} avg engagement")
            print(f"  {hour_info['recommendation']}")

        print("\n" + "=" * 60)
        print("CONTENT PERFORMANCE")
        print("=" * 60)
        for media in analysis['content_performance']['all_types']:
            print(f"\n{media['type'].upper()}: {media['avg_engagement']:.0f} avg engagement (rank #{media['rank']})")

        # Save
        advisor.save_analysis()

        print("\n" + "=" * 60)
        print("LLM CONTEXT")
        print("=" * 60)
        print(advisor.to_prompt_context())

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
