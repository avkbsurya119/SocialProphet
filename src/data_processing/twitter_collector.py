"""
Twitter Data Collector for SocialProphet.

Real-time Twitter data collection using Twitter API v2.
Integrates with the dashboard for live data refresh.
"""

import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

try:
    import tweepy
    HAS_TWEEPY = True
except ImportError:
    HAS_TWEEPY = False
    print("Warning: tweepy not installed. Install with: pip install tweepy")

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class TwitterCollector:
    """
    Real-time Twitter data collection using Twitter API v2.

    Features:
    - Search recent tweets by query
    - Get engagement metrics
    - Track trending topics
    - Rate limit handling
    """

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        consumer_key: Optional[str] = None,
        consumer_secret: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize Twitter collector.

        Args:
            bearer_token: Twitter API bearer token
            consumer_key: Twitter API consumer key
            consumer_secret: Twitter API consumer secret
            config: Configuration object
        """
        if not HAS_TWEEPY:
            raise ImportError(
                "tweepy is required for Twitter collection. "
                "Install with: pip install tweepy"
            )

        self.config = config or Config()

        # Get credentials from environment or parameters
        self.bearer_token = bearer_token or os.getenv('TWITTER_BEARER_TOKEN')
        self.consumer_key = consumer_key or os.getenv('TWITTER_CONSUMER_KEY')
        self.consumer_secret = consumer_secret or os.getenv('TWITTER_CONSUMER_SECRET')

        if not self.bearer_token:
            raise ValueError(
                "Twitter bearer token required. Set TWITTER_BEARER_TOKEN "
                "environment variable or pass bearer_token parameter."
            )

        # Initialize client
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
            wait_on_rate_limit=True
        )

        # Cache for collected data
        self._cache = {}
        self._last_fetch = None

        print("Twitter collector initialized!")

    def search_recent_tweets(
        self,
        query: str,
        max_results: int = 100,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Search recent tweets matching a query.

        Args:
            query: Search query (supports Twitter search operators)
            max_results: Maximum number of tweets to return (10-100)
            days_back: Number of days to search back (max 7 for recent search)

        Returns:
            DataFrame with tweet data
        """
        # Ensure max_results is within API limits
        max_results = min(max(10, max_results), 100)

        # Calculate start time
        start_time = datetime.utcnow() - timedelta(days=min(days_back, 7))

        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                start_time=start_time,
                tweet_fields=[
                    'created_at',
                    'public_metrics',
                    'lang',
                    'source',
                    'conversation_id'
                ],
                expansions=['author_id'],
                user_fields=['username', 'public_metrics']
            )

            if not tweets.data:
                print(f"No tweets found for query: {query}")
                return pd.DataFrame()

            # Process tweets
            tweet_data = []
            for tweet in tweets.data:
                metrics = tweet.public_metrics or {}
                tweet_data.append({
                    'tweet_id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'likes': metrics.get('like_count', 0),
                    'retweets': metrics.get('retweet_count', 0),
                    'replies': metrics.get('reply_count', 0),
                    'quotes': metrics.get('quote_count', 0),
                    'engagement': (
                        metrics.get('like_count', 0) +
                        metrics.get('retweet_count', 0) * 2 +
                        metrics.get('reply_count', 0) * 3 +
                        metrics.get('quote_count', 0) * 2
                    ),
                    'lang': tweet.lang,
                    'source': tweet.source
                })

            df = pd.DataFrame(tweet_data)
            df['created_at'] = pd.to_datetime(df['created_at'])

            self._last_fetch = datetime.now()
            self._cache[query] = df

            print(f"Collected {len(df)} tweets for query: {query}")
            return df

        except tweepy.TweepyException as e:
            print(f"Twitter API error: {e}")
            return pd.DataFrame()

    def get_user_tweets(
        self,
        username: str,
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Get recent tweets from a specific user.

        Args:
            username: Twitter username (without @)
            max_results: Maximum number of tweets

        Returns:
            DataFrame with user's tweets
        """
        try:
            # Get user ID
            user = self.client.get_user(username=username)
            if not user.data:
                print(f"User not found: {username}")
                return pd.DataFrame()

            user_id = user.data.id

            # Get user's tweets
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=min(max_results, 100),
                tweet_fields=['created_at', 'public_metrics']
            )

            if not tweets.data:
                return pd.DataFrame()

            tweet_data = []
            for tweet in tweets.data:
                metrics = tweet.public_metrics or {}
                tweet_data.append({
                    'tweet_id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'likes': metrics.get('like_count', 0),
                    'retweets': metrics.get('retweet_count', 0),
                    'replies': metrics.get('reply_count', 0),
                    'engagement': (
                        metrics.get('like_count', 0) +
                        metrics.get('retweet_count', 0) * 2 +
                        metrics.get('reply_count', 0) * 3
                    ),
                    'username': username
                })

            df = pd.DataFrame(tweet_data)
            df['created_at'] = pd.to_datetime(df['created_at'])

            return df

        except tweepy.TweepyException as e:
            print(f"Error fetching user tweets: {e}")
            return pd.DataFrame()

    def get_trending_topics(
        self,
        woeid: int = 1  # 1 = Worldwide
    ) -> List[Dict[str, Any]]:
        """
        Get trending topics.

        Args:
            woeid: Where On Earth ID (1=Worldwide, 23424977=US, etc.)

        Returns:
            List of trending topics with tweet volumes
        """
        try:
            # Note: Trending topics requires different API access
            # This is a placeholder for when that access is available
            print("Trending topics requires elevated API access")
            return []

        except tweepy.TweepyException as e:
            print(f"Error fetching trends: {e}")
            return []

    def analyze_engagement(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze engagement metrics from collected tweets.

        Args:
            df: DataFrame with tweet data

        Returns:
            Dictionary with engagement analysis
        """
        if df.empty:
            return {'error': 'No data to analyze'}

        analysis = {
            'total_tweets': len(df),
            'total_engagement': int(df['engagement'].sum()),
            'avg_engagement': float(df['engagement'].mean()),
            'max_engagement': int(df['engagement'].max()),
            'total_likes': int(df['likes'].sum()),
            'total_retweets': int(df['retweets'].sum()),
            'total_replies': int(df['replies'].sum()),
            'engagement_by_day': {},
            'top_tweets': []
        }

        # Engagement by day
        if 'created_at' in df.columns:
            df['date'] = df['created_at'].dt.date
            daily = df.groupby('date')['engagement'].sum()
            analysis['engagement_by_day'] = {
                str(k): int(v) for k, v in daily.items()
            }

        # Top tweets
        top = df.nlargest(5, 'engagement')
        analysis['top_tweets'] = [
            {
                'text': row['text'][:100] + '...' if len(row['text']) > 100 else row['text'],
                'engagement': int(row['engagement']),
                'likes': int(row['likes']),
                'retweets': int(row['retweets'])
            }
            for _, row in top.iterrows()
        ]

        return analysis

    def to_forecast_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Twitter data to forecast-compatible format.

        Args:
            df: DataFrame with tweet data

        Returns:
            DataFrame in forecast format (ds, y columns)
        """
        if df.empty:
            return pd.DataFrame()

        # Aggregate by day
        df['date'] = df['created_at'].dt.date
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

    def save_data(
        self,
        df: pd.DataFrame,
        filename: str = 'twitter_data.csv'
    ) -> Path:
        """
        Save collected data to CSV.

        Args:
            df: DataFrame to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = Path(self.config.RAW_DATA_DIR) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"Data saved to: {output_path}")

        return output_path

    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get cache status.

        Returns:
            Cache status dictionary
        """
        return {
            'queries_cached': list(self._cache.keys()),
            'last_fetch': self._last_fetch.isoformat() if self._last_fetch else None,
            'total_cached_tweets': sum(len(df) for df in self._cache.values())
        }

    def clear_cache(self):
        """Clear the data cache."""
        self._cache = {}
        self._last_fetch = None
        print("Cache cleared")


def test_twitter_collector():
    """Test the Twitter collector."""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        collector = TwitterCollector()

        print("\nSearching for tweets...")
        df = collector.search_recent_tweets(
            query="python programming",
            max_results=10
        )

        if not df.empty:
            print(f"\nCollected {len(df)} tweets")
            print(df[['text', 'engagement']].head())

            print("\nEngagement Analysis:")
            analysis = collector.analyze_engagement(df)
            print(f"  Total engagement: {analysis['total_engagement']}")
            print(f"  Avg engagement: {analysis['avg_engagement']:.2f}")

        return True

    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    test_twitter_collector()
