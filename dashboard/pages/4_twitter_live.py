"""
Twitter Live Page - SocialProphet Dashboard.

Real-time Twitter data collection and analysis using Twitter API v2.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config

st.set_page_config(
    page_title="Twitter Live - SocialProphet",
    page_icon="🐦",
    layout="wide"
)

st.title("🐦 Twitter Live Data")
st.markdown("Real-time Twitter data collection and engagement analysis.")

# Check API status
api_status = Config.validate_api_keys()
twitter_available = api_status.get('twitter', False)

# Sidebar configuration
st.sidebar.markdown("### Twitter API Status")
if twitter_available:
    st.sidebar.success("✅ Twitter API Connected")
else:
    st.sidebar.error("❌ Twitter API Not Configured")
    st.sidebar.markdown("""
    **Setup Instructions:**
    1. Apply at developer.twitter.com
    2. Create a project and app
    3. Generate Bearer Token
    4. Add to `.env` file:
    ```
    TWITTER_BEARER_TOKEN=your_token
    ```
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### Search Settings")

search_query = st.sidebar.text_input(
    "Search Query",
    value="#SocialMedia OR #Marketing",
    help="Use Twitter search operators"
)

max_results = st.sidebar.slider(
    "Max Results",
    min_value=10,
    max_value=100,
    value=50
)

days_back = st.sidebar.slider(
    "Days Back",
    min_value=1,
    max_value=7,
    value=7
)

# Main content
st.markdown("---")

if not twitter_available:
    st.warning("""
    ⚠️ **Twitter API not configured**

    To use Twitter Live features, you need to:
    1. Get Twitter API access at [developer.twitter.com](https://developer.twitter.com)
    2. Generate a Bearer Token
    3. Add `TWITTER_BEARER_TOKEN` to your `.env` file

    See the [Setup Guide](/docs/setup_guide.md) for detailed instructions.
    """)

    # Show demo data
    st.markdown("### Demo Data (Sample)")
    demo_df = pd.DataFrame({
        'tweet_id': [f'12345678{i}' for i in range(10)],
        'text': [
            "Great insights on social media marketing! #SocialMedia",
            "Just discovered this amazing analytics tool #Marketing",
            "Content is king, but engagement is queen #ContentMarketing",
            "New study shows 40% higher engagement with video content",
            "Tips for growing your audience: consistency is key!",
            "The algorithm loves authentic content #Authenticity",
            "Scheduling posts at peak times = better reach",
            "Don't forget to engage with your community!",
            "Data-driven decisions lead to better outcomes",
            "Quality over quantity - always #ContentStrategy"
        ],
        'likes': [120, 85, 234, 156, 89, 178, 92, 145, 201, 167],
        'retweets': [45, 32, 78, 54, 28, 62, 35, 48, 73, 55],
        'replies': [12, 8, 25, 18, 7, 22, 11, 15, 28, 19],
        'engagement': [345, 205, 645, 423, 189, 534, 256, 398, 602, 478],
        'created_at': [datetime.now() - timedelta(hours=i*2) for i in range(10)]
    })
    st.dataframe(demo_df, use_container_width=True)

else:
    # Live Twitter collection
    st.markdown("## Live Twitter Collection")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🔄 Fetch Tweets", type="primary", use_container_width=True):
            with st.spinner("Collecting tweets..."):
                try:
                    from src.data_processing.twitter_collector import TwitterCollector

                    collector = TwitterCollector()
                    df = collector.search_recent_tweets(
                        query=search_query,
                        max_results=max_results,
                        days_back=days_back
                    )

                    if not df.empty:
                        st.session_state['twitter_data'] = df
                        st.session_state['last_fetch'] = datetime.now()
                        st.success(f"✅ Collected {len(df)} tweets!")
                    else:
                        st.warning("No tweets found for this query.")

                except Exception as e:
                    st.error(f"Error collecting tweets: {str(e)}")

    with col2:
        if 'last_fetch' in st.session_state:
            st.info(f"Last fetch: {st.session_state['last_fetch'].strftime('%H:%M:%S')}")

    # Display collected data
    if 'twitter_data' in st.session_state and not st.session_state['twitter_data'].empty:
        df = st.session_state['twitter_data']

        st.markdown("---")
        st.markdown("## Collected Tweets")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Tweets", len(df))

        with col2:
            st.metric("Total Engagement", f"{df['engagement'].sum():,}")

        with col3:
            st.metric("Avg Engagement", f"{df['engagement'].mean():.1f}")

        with col4:
            st.metric("Max Engagement", f"{df['engagement'].max():,}")

        # Data table
        st.markdown("### Tweet Data")
        st.dataframe(df, use_container_width=True)

        # Analysis
        st.markdown("---")
        st.markdown("## Engagement Analysis")

        analysis_tabs = st.tabs(["Overview", "Top Tweets", "Timeline"])

        with analysis_tabs[0]:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Engagement Distribution")
                st.bar_chart(df['engagement'].value_counts().head(20))

            with col2:
                st.markdown("### Metrics Breakdown")
                metrics_df = df[['likes', 'retweets', 'replies']].sum()
                st.bar_chart(metrics_df)

        with analysis_tabs[1]:
            st.markdown("### Top Performing Tweets")
            top_tweets = df.nlargest(5, 'engagement')[['text', 'likes', 'retweets', 'engagement']]
            for i, row in top_tweets.iterrows():
                with st.expander(f"Engagement: {row['engagement']} | 👍 {row['likes']} | 🔄 {row['retweets']}"):
                    st.markdown(row['text'])

        with analysis_tabs[2]:
            st.markdown("### Engagement Over Time")
            if 'created_at' in df.columns:
                time_df = df.set_index('created_at')['engagement']
                st.line_chart(time_df)

        # Export
        st.markdown("---")
        st.markdown("## Export Data")

        col1, col2 = st.columns(2)

        with col1:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"twitter_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            if st.button("📊 Convert to Forecast Format"):
                try:
                    from src.data_processing.twitter_collector import TwitterCollector
                    collector = TwitterCollector()
                    forecast_df = collector.to_forecast_format(df)
                    st.session_state['forecast_df'] = forecast_df
                    st.success("Converted to forecast format!")
                    st.dataframe(forecast_df)
                except Exception as e:
                    st.error(f"Conversion error: {e}")

# User search section
st.markdown("---")
st.markdown("## User Tweets Lookup")

col1, col2 = st.columns([2, 1])

with col1:
    username = st.text_input(
        "Twitter Username",
        value="",
        placeholder="Enter username (without @)",
        help="Search for tweets from a specific user"
    )

with col2:
    user_max_results = st.number_input(
        "Max Tweets",
        min_value=10,
        max_value=100,
        value=50
    )

if username and twitter_available:
    if st.button("🔍 Search User Tweets"):
        with st.spinner(f"Fetching tweets from @{username}..."):
            try:
                from src.data_processing.twitter_collector import TwitterCollector

                collector = TwitterCollector()
                user_df = collector.get_user_tweets(
                    username=username,
                    max_results=user_max_results
                )

                if not user_df.empty:
                    st.success(f"Found {len(user_df)} tweets from @{username}")
                    st.dataframe(user_df, use_container_width=True)
                else:
                    st.warning(f"No tweets found for @{username}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Twitter API info
st.markdown("---")
st.markdown("## Twitter API v2 Features")

st.markdown("""
### Available Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Search Recent Tweets** | ✅ | Search tweets from the last 7 days |
| **User Tweets** | ✅ | Get tweets from specific users |
| **Engagement Metrics** | ✅ | Likes, retweets, replies, quotes |
| **Rate Limiting** | ✅ | Automatic wait on rate limit |
| **Caching** | ✅ | Response caching for efficiency |

### API Limits (Free Tier)

| Endpoint | Limit | Window |
|----------|-------|--------|
| Search Recent | 450 req | 15 min |
| User Tweets | 1500 req | 15 min |
| User Lookup | 300 req | 15 min |

### Data Fields Collected

- `tweet_id` - Unique tweet identifier
- `text` - Tweet content
- `created_at` - Timestamp
- `likes` - Like count
- `retweets` - Retweet count
- `replies` - Reply count
- `quotes` - Quote tweet count
- `engagement` - Weighted engagement score
- `lang` - Tweet language
- `source` - Tweet source (device/app)
""")

# Footer
st.markdown("---")
st.caption("SocialProphet - Twitter Live | Twitter API v2 | Real-time Data Collection")
