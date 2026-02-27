"""
Data Overview Page - SocialProphet Dashboard.

Explore and visualize the social media engagement datasets.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config

st.set_page_config(
    page_title="Data Overview - SocialProphet",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Overview")
st.markdown("Explore the social media engagement datasets used for forecasting.")

# Data paths
PROCESSED_DIR = Path(Config.PROCESSED_DATA_DIR)


@st.cache_data
def load_data(filename: str) -> pd.DataFrame:
    """Load and cache dataset."""
    filepath = PROCESSED_DIR / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
        return df
    return pd.DataFrame()


# Dataset selection
st.sidebar.markdown("### Dataset Selection")
dataset_options = {
    "Combined Data": "combined_data.csv",
    "Instagram (Cleaned)": "instagram_cleaned.csv",
    "Social Media (Cleaned)": "social_media_cleaned.csv",
    "Viral (Cleaned)": "viral_cleaned.csv",
    "Training Data": "train_data.csv",
    "Test Data": "test_data.csv",
    "Daily Aggregated": "daily_aggregated.csv"
}

selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    options=list(dataset_options.keys()),
    index=0
)

# Load selected dataset
df = load_data(dataset_options[selected_dataset])

if df.empty:
    st.warning(f"Dataset not found: {dataset_options[selected_dataset]}")
    st.info("Run the preprocessing pipeline first: `python scripts/preprocess_datasets.py`")
    st.stop()

# Dataset overview
st.markdown("---")
st.markdown("## Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Rows", f"{len(df):,}")

with col2:
    st.metric("Total Columns", len(df.columns))

with col3:
    if 'engagement' in df.columns:
        st.metric("Avg Engagement", f"{df['engagement'].mean():,.0f}")
    elif 'y' in df.columns:
        st.metric("Avg Target (log)", f"{df['y'].mean():.2f}")

with col4:
    if 'platform' in df.columns:
        st.metric("Platforms", df['platform'].nunique())
    elif 'source' in df.columns:
        st.metric("Sources", df['source'].nunique())

# Data preview
st.markdown("---")
st.markdown("## Data Preview")

preview_rows = st.slider("Rows to display", 5, 100, 10)
st.dataframe(df.head(preview_rows), use_container_width=True)

# Column info
st.markdown("---")
st.markdown("## Column Information")

col_info = pd.DataFrame({
    'Column': df.columns,
    'Type': df.dtypes.values,
    'Non-Null': df.count().values,
    'Null %': ((df.isnull().sum() / len(df)) * 100).round(2).values,
    'Unique': df.nunique().values
})

st.dataframe(col_info, use_container_width=True)

# Statistics
st.markdown("---")
st.markdown("## Descriptive Statistics")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    stats_df = df[numeric_cols].describe().round(2)
    st.dataframe(stats_df, use_container_width=True)

# Visualizations
st.markdown("---")
st.markdown("## Visualizations")

viz_tabs = st.tabs(["Engagement Distribution", "Time Series", "Platform Analysis"])

with viz_tabs[0]:
    st.markdown("### Engagement Distribution")
    if 'engagement' in df.columns:
        st.bar_chart(df['engagement'].value_counts().head(50))
    elif 'y' in df.columns:
        st.line_chart(df['y'])
    else:
        st.info("No engagement column found in this dataset.")

with viz_tabs[1]:
    st.markdown("### Engagement Over Time")
    date_col = 'timestamp' if 'timestamp' in df.columns else ('ds' if 'ds' in df.columns else None)
    value_col = 'engagement' if 'engagement' in df.columns else ('y' if 'y' in df.columns else None)

    if date_col and value_col:
        time_df = df[[date_col, value_col]].copy()
        time_df = time_df.set_index(date_col)
        if len(time_df) > 1000:
            time_df = time_df.resample('D').mean()
        st.line_chart(time_df)
    else:
        st.info("No time series data available for this dataset.")

with viz_tabs[2]:
    st.markdown("### Platform Distribution")
    if 'platform' in df.columns:
        platform_counts = df['platform'].value_counts()
        st.bar_chart(platform_counts)

        # Platform metrics
        st.markdown("#### Platform Statistics")
        platform_stats = df.groupby('platform').agg({
            'engagement': ['mean', 'median', 'sum', 'count']
        }).round(2)
        platform_stats.columns = ['Mean', 'Median', 'Total', 'Posts']
        st.dataframe(platform_stats, use_container_width=True)
    elif 'source' in df.columns:
        source_counts = df['source'].value_counts()
        st.bar_chart(source_counts)
    else:
        st.info("No platform/source column found in this dataset.")

# Data quality
st.markdown("---")
st.markdown("## Data Quality Check")

quality_checks = {
    "Missing Values": (df.isnull().sum().sum() == 0),
    "No Duplicates": (df.duplicated().sum() == 0),
    "Positive Engagement": (df['engagement'].min() >= 0 if 'engagement' in df.columns else True),
    "Valid Dates": (df[date_col].notna().all() if date_col else True)
}

col1, col2, col3, col4 = st.columns(4)
cols = [col1, col2, col3, col4]

for i, (check, passed) in enumerate(quality_checks.items()):
    with cols[i]:
        if passed:
            st.success(f"✅ {check}")
        else:
            st.error(f"❌ {check}")

# Export options
st.markdown("---")
st.markdown("## Export Data")

col1, col2 = st.columns(2)

with col1:
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"{selected_dataset.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

with col2:
    json_data = df.to_json(orient='records', date_format='iso')
    st.download_button(
        label="📥 Download JSON",
        data=json_data,
        file_name=f"{selected_dataset.lower().replace(' ', '_')}.json",
        mime="application/json"
    )

# Footer
st.markdown("---")
st.caption("SocialProphet - Data Overview | Use the sidebar to switch datasets")
