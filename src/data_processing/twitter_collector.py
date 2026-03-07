"""
Twitter Data Collector for SocialProphet.

Real-time Twitter data collection using Twitter API v2.
Includes demo data fallback when API is unavailable.
"""

import os
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

try:
    import tweepy
    HAS_TWEEPY = True
except ImportError:
    HAS_TWEEPY = False

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


# Demo data for when API is unavailable
DEMO_TWEETS = [
    {"text": "Just discovered an amazing productivity hack for developers! Thread below 🧵 #coding #productivity", "likes": 245, "retweets": 67, "replies": 23},
    {"text": "Python tip: Use list comprehensions instead of loops for cleaner code. What's your favorite Python trick? #Python #programming", "likes": 189, "retweets": 45, "replies": 34},
    {"text": "The future of AI is here. Machine learning is transforming every industry. Are you ready? #AI #MachineLearning #Tech", "likes": 567, "retweets": 123, "replies": 45},
    {"text": "Data visualization tip: Always label your axes and use meaningful colors. Your audience will thank you! #datascience #visualization", "likes": 134, "retweets": 28, "replies": 12},
    {"text": "Building my first web app with React and it's going great! The component-based architecture is so intuitive. #webdev #React #JavaScript", "likes": 223, "retweets": 34, "replies": 28},
    {"text": "Hot take: Clean code is more important than clever code. Readability > Performance in 90% of cases. Thoughts? #programming #cleancode", "likes": 456, "retweets": 89, "replies": 67},
    {"text": "Started learning cloud computing today. AWS or Azure - which one should I focus on first? #cloud #AWS #Azure", "likes": 178, "retweets": 23, "replies": 89},
    {"text": "Reminder: Take breaks while coding. Your brain needs rest to solve complex problems effectively. #developerlife #mentalhealth", "likes": 345, "retweets": 78, "replies": 34},
    {"text": "Just deployed my first API! The feeling of seeing it work in production is unmatched. #backend #API #coding", "likes": 267, "retweets": 45, "replies": 23},
    {"text": "Machine learning model achieved 95% accuracy! Months of work finally paying off. #ML #DataScience #AI", "likes": 489, "retweets": 112, "replies": 56},
    {"text": "Git tip: Use meaningful commit messages. Future you will be grateful! #git #versioncontrol #coding", "likes": 234, "retweets": 56, "replies": 18},
    {"text": "The best investment you can make is in yourself. Keep learning, keep growing! #motivation #tech #career", "likes": 567, "retweets": 134, "replies": 45},
    {"text": "SQL optimization tip: Always index your frequently queried columns. Can improve performance 10x! #SQL #database #optimization", "likes": 189, "retweets": 34, "replies": 23},
    {"text": "Working from home productivity tip: Create a dedicated workspace and stick to a routine. #WFH #productivity #remotework", "likes": 312, "retweets": 67, "replies": 34},
    {"text": "Just finished a great tutorial on Docker. Containerization is a game-changer for deployment! #Docker #DevOps #containers", "likes": 234, "retweets": 45, "replies": 28},
]


class TwitterCollector:
    """
    Twitter data collection with demo fallback.

    When API is unavailable (rate limits, no credits), uses realistic demo data.
    """

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        config: Optional[Config] = None,
        use_demo_on_failure: bool = True
    ):
        """Initialize Twitter collector."""
        self.config = config or Config()
        self.use_demo_on_failure = use_demo_on_failure
        self.api_available = False
        self.last_error = None

        # Get credentials
        self.bearer_token = bearer_token or os.getenv('TWITTER_BEARER_TOKEN')

        # Initialize client if possible
        if HAS_TWEEPY and self.bearer_token:
            try:
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    wait_on_rate_limit=True
                )
                self.api_available = True
            except Exception as e:
                self.last_error = str(e)
                self.client = None
        else:
            self.client = None
            if not HAS_TWEEPY:
                self.last_error = "tweepy not installed"
            elif not self.bearer_token:
                self.last_error = "No bearer token"

        # Cache
        self._cache = {}
        self._last_fetch = None

    def search_recent_tweets(
        self,
        query: str,
        max_results: int = 100,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Search recent tweets. Falls back to demo data on API failure.
        """
        max_results = min(max(10, max_results), 100)

        # Try API first
        if self.api_available and self.client:
            try:
                start_time = datetime.utcnow() - timedelta(days=min(days_back, 7))

                tweets = self.client.search_recent_tweets(
                    query=query,
                    max_results=max_results,
                    start_time=start_time,
                    tweet_fields=['created_at', 'public_metrics', 'lang'],
                )

                if tweets.data:
                    tweet_data = []
                    for tweet in tweets.data:
                        metrics = tweet.public_metrics or {}
                        tweet_data.append({
                            'tweet_id': str(tweet.id),
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'likes': metrics.get('like_count', 0),
                            'retweets': metrics.get('retweet_count', 0),
                            'replies': metrics.get('reply_count', 0),
                            'quotes': metrics.get('quote_count', 0),
                            'engagement': self._calc_engagement(metrics),
                            'source': 'twitter_api'
                        })

                    df = pd.DataFrame(tweet_data)
                    df['created_at'] = pd.to_datetime(df['created_at'])
                    self._cache[query] = df
                    self._last_fetch = datetime.now()
                    return df

            except tweepy.TweepyException as e:
                self.last_error = str(e)
                self.api_available = False

        # Fallback to demo data
        if self.use_demo_on_failure:
            return self._generate_demo_data(query, max_results, days_back)

        return pd.DataFrame()

    def _calc_engagement(self, metrics: dict) -> int:
        """Calculate weighted engagement score."""
        return (
            metrics.get('like_count', 0) +
            metrics.get('retweet_count', 0) * 2 +
            metrics.get('reply_count', 0) * 3 +
            metrics.get('quote_count', 0) * 2
        )

    def _generate_demo_data(
        self,
        query: str,
        max_results: int,
        days_back: int
    ) -> pd.DataFrame:
        """Generate realistic demo data."""
        tweets = []
        base_time = datetime.now()

        for i in range(min(max_results, len(DEMO_TWEETS) * 2)):
            template = random.choice(DEMO_TWEETS)

            # Add variation
            likes = int(template['likes'] * random.uniform(0.5, 1.5))
            retweets = int(template['retweets'] * random.uniform(0.5, 1.5))
            replies = int(template['replies'] * random.uniform(0.5, 1.5))

            # Random time in the past days_back days
            hours_ago = random.randint(1, days_back * 24)
            created_at = base_time - timedelta(hours=hours_ago)

            tweets.append({
                'tweet_id': f'demo_{i}_{random.randint(10000, 99999)}',
                'text': template['text'],
                'created_at': created_at,
                'likes': likes,
                'retweets': retweets,
                'replies': replies,
                'quotes': random.randint(0, 10),
                'engagement': likes + retweets * 2 + replies * 3,
                'source': 'demo_data'
            })

        df = pd.DataFrame(tweets)
        df = df.sort_values('created_at', ascending=False).reset_index(drop=True)
        return df

    def get_user_tweets(
        self,
        username: str,
        max_results: int = 100
    ) -> pd.DataFrame:
        """Get user tweets with demo fallback."""
        if self.api_available and self.client:
            try:
                user = self.client.get_user(username=username)
                if user.data:
                    tweets = self.client.get_users_tweets(
                        id=user.data.id,
                        max_results=min(max_results, 100),
                        tweet_fields=['created_at', 'public_metrics']
                    )

                    if tweets.data:
                        data = []
                        for tweet in tweets.data:
                            metrics = tweet.public_metrics or {}
                            data.append({
                                'tweet_id': str(tweet.id),
                                'text': tweet.text,
                                'created_at': tweet.created_at,
                                'likes': metrics.get('like_count', 0),
                                'retweets': metrics.get('retweet_count', 0),
                                'replies': metrics.get('reply_count', 0),
                                'engagement': self._calc_engagement(metrics),
                                'username': username,
                                'source': 'twitter_api'
                            })

                        df = pd.DataFrame(data)
                        df['created_at'] = pd.to_datetime(df['created_at'])
                        return df

            except tweepy.TweepyException as e:
                self.last_error = str(e)

        # Demo fallback
        if self.use_demo_on_failure:
            df = self._generate_demo_data(username, max_results, 7)
            df['username'] = username
            return df

        return pd.DataFrame()

    def analyze_engagement(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze engagement metrics."""
        if df.empty:
            return {'error': 'No data'}

        return {
            'total_tweets': len(df),
            'total_engagement': int(df['engagement'].sum()),
            'avg_engagement': float(df['engagement'].mean()),
            'max_engagement': int(df['engagement'].max()),
            'min_engagement': int(df['engagement'].min()),
            'total_likes': int(df['likes'].sum()),
            'total_retweets': int(df['retweets'].sum()),
            'total_replies': int(df['replies'].sum()),
            'data_source': df['source'].iloc[0] if 'source' in df.columns else 'unknown',
            'api_available': self.api_available,
            'last_error': self.last_error
        }

    def to_forecast_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to forecast format (ds, y)."""
        if df.empty:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['created_at']).dt.date
        daily = df.groupby('date').agg({
            'engagement': 'sum',
            'likes': 'sum',
            'retweets': 'sum',
            'replies': 'sum',
            'tweet_id': 'count'
        }).reset_index()

        daily.columns = ['ds', 'engagement', 'likes', 'retweets', 'replies', 'post_count']
        daily['ds'] = pd.to_datetime(daily['ds'])
        daily['y'] = daily['engagement']

        return daily

    def get_status(self) -> Dict[str, Any]:
        """Get collector status."""
        return {
            'api_available': self.api_available,
            'has_tweepy': HAS_TWEEPY,
            'has_token': bool(self.bearer_token),
            'last_error': self.last_error,
            'cached_queries': list(self._cache.keys()),
            'demo_mode': not self.api_available and self.use_demo_on_failure
        }

    def save_data(self, df: pd.DataFrame, filename: str = 'twitter_data.csv') -> Path:
        """Save data to CSV."""
        output_path = Path(self.config.RAW_DATA_DIR) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path


def test_collector():
    """Test the collector."""
    print("Testing Twitter Collector...")
    collector = TwitterCollector()

    print(f"\nStatus: {collector.get_status()}")

    print("\nSearching tweets...")
    df = collector.search_recent_tweets("python", max_results=20)
    print(f"Got {len(df)} tweets")

    if not df.empty:
        print(f"\nSample: {df['text'].iloc[0][:80]}...")
        analysis = collector.analyze_engagement(df)
        print(f"Avg engagement: {analysis['avg_engagement']:.1f}")
        print(f"Data source: {analysis['data_source']}")


if __name__ == "__main__":
    test_collector()
