"""
Twitter Live Page - SocialProphet Dashboard.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import Config

st.set_page_config(page_title="Twitter Live - SocialProphet", page_icon="🐦", layout="wide")

st.title("🐦 Twitter Live Data")
st.markdown("Real-time Twitter data collection and analysis.")

# Check API and get detailed status
api_status = Config.validate_api_keys()
twitter_ok = api_status.get('twitter', False)

# Get detailed collector status
collector_status = None
try:
    from src.data_processing.twitter_collector import TwitterCollector
    collector = TwitterCollector()
    collector_status = collector.get_status()
except Exception as e:
    collector_status = {'error': str(e)}

# Sidebar
st.sidebar.markdown("### API Status")
if collector_status and collector_status.get('credits_exhausted'):
    st.sidebar.warning("⚠️ API Credits Exhausted")
    st.sidebar.markdown("""
    **Twitter API requires payment.**

    Since November 2023, Twitter (X) requires a paid plan for API access.

    **Options:**
    - Use demo data (current mode)
    - Subscribe to X API Basic ($100/mo)
    - Use Instagram Analytics instead

    *Demo mode provides realistic sample data.*
    """)
    twitter_ok = False
elif twitter_ok:
    st.sidebar.success("✅ Twitter API Connected")
else:
    st.sidebar.error("❌ Twitter API Not Configured")
    st.sidebar.markdown("""
    **Setup:**
    1. Get API access at developer.twitter.com
    2. Generate Bearer Token
    3. Add to .env:
    ```
    TWITTER_BEARER_TOKEN=your_token
    ```
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### Search Settings")

query = st.sidebar.text_input("Search Query", value="#Python OR #DataScience")
max_results = st.sidebar.slider("Max Results", 10, 100, 50)
days_back = st.sidebar.slider("Days Back", 1, 7, 7)

# Main content
st.markdown("---")

if not twitter_ok:
    if collector_status and collector_status.get('credits_exhausted'):
        st.warning("⚠️ Twitter API credits exhausted. Showing demo data.")
        st.info("""
        **Note:** Twitter moved to a paid API model in 2023. Free tier no longer allows tweet searches.

        **Alternative:** Use the **Instagram Analytics** data which has 30,000 real posts with engagement metrics.
        Check the **Posting Advisor** page for actionable insights from your Instagram data.
        """)
    else:
        st.warning("⚠️ Twitter API not configured. Set TWITTER_BEARER_TOKEN in .env")

    # Demo data
    st.markdown("### Demo Data (Sample)")
    demo = pd.DataFrame({
        'text': [
            "Python is amazing for data science! #Python #DataScience",
            "Just deployed my first ML model using scikit-learn",
            "Learning about time series forecasting with Prophet",
            "Data visualization with matplotlib is so powerful",
            "Building a dashboard with Streamlit - so easy!"
        ],
        'likes': [150, 89, 234, 123, 178],
        'retweets': [45, 22, 78, 34, 56],
        'replies': [12, 8, 25, 11, 19],
        'engagement': [297, 163, 490, 235, 365]
    })
    st.dataframe(demo, use_container_width=True)

else:
    st.markdown("## Search Tweets")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🔄 Fetch Tweets", type="primary", use_container_width=True):
            with st.spinner(f"Searching: {query}"):
                try:
                    from src.data_processing.twitter_collector import TwitterCollector

                    collector = TwitterCollector()
                    df = collector.search_recent_tweets(
                        query=query,
                        max_results=max_results,
                        days_back=days_back
                    )

                    if not df.empty:
                        st.session_state['tweets'] = df
                        st.session_state['fetch_time'] = datetime.now()
                        st.success(f"✅ Found {len(df)} tweets!")
                    else:
                        st.warning("No tweets found for this query.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with col2:
        if 'fetch_time' in st.session_state:
            st.info(f"Last fetch: {st.session_state['fetch_time'].strftime('%H:%M:%S')}")

    # Display results
    if 'tweets' in st.session_state and not st.session_state['tweets'].empty:
        df = st.session_state['tweets']

        st.markdown("---")
        st.markdown("## Results")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tweets", len(df))
        with col2:
            st.metric("Total Engagement", f"{df['engagement'].sum():,}")
        with col3:
            st.metric("Avg Engagement", f"{df['engagement'].mean():.0f}")
        with col4:
            st.metric("Max Engagement", f"{df['engagement'].max():,}")

        # Data table
        st.dataframe(df, use_container_width=True)

        # Analysis
        st.markdown("---")
        tab1, tab2 = st.tabs(["Top Tweets", "Analysis"])

        with tab1:
            st.markdown("### Top Performing")
            top = df.nlargest(5, 'engagement')
            for _, row in top.iterrows():
                with st.expander(f"👍 {row['likes']} | 🔄 {row['retweets']} | Engagement: {row['engagement']}"):
                    st.write(row['text'])

        with tab2:
            st.markdown("### Engagement Distribution")
            st.bar_chart(df['engagement'])

        # Export
        st.markdown("---")
        st.download_button(
            "📥 Download Data",
            df.to_csv(index=False),
            f"twitter_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )

# User lookup
st.markdown("---")
st.markdown("## User Lookup")

col1, col2 = st.columns([2, 1])

with col1:
    username = st.text_input("Username (without @)", "")

with col2:
    user_max = st.number_input("Max Tweets", 10, 100, 50)

if username and twitter_ok:
    if st.button("🔍 Get User Tweets"):
        with st.spinner(f"Fetching @{username}..."):
            try:
                from src.data_processing.twitter_collector import TwitterCollector

                collector = TwitterCollector()
                user_df = collector.get_user_tweets(username, user_max)

                if not user_df.empty:
                    st.success(f"Found {len(user_df)} tweets")
                    st.dataframe(user_df, use_container_width=True)
                else:
                    st.warning(f"No tweets found for @{username}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Info
st.markdown("---")
with st.expander("ℹ️ Twitter API Info"):
    st.markdown("""
    **Features:**
    - Search recent tweets (7 days)
    - User timeline lookup
    - Engagement metrics (likes, retweets, replies)

    **Rate Limits (Free Tier):**
    - 450 searches / 15 min
    - 1500 user tweets / 15 min

    **Engagement Formula:**
    ```
    engagement = likes + retweets*2 + replies*3 + quotes*2
    ```
    """)
