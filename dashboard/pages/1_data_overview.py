"""
Data Overview Page - SocialProphet Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Data Overview - SocialProphet", page_icon="📊", layout="wide")

st.title("📊 Data Overview")
st.markdown("Explore the social media engagement datasets.")

# Data directory
DATA_DIR = PROJECT_ROOT / "data" / "processed"

@st.cache_data
def load_dataset(filename: str) -> pd.DataFrame:
    """Load dataset from processed folder."""
    filepath = DATA_DIR / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        # Parse date columns
        for col in ['timestamp', 'ds', 'date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    return pd.DataFrame()

# Sidebar - Dataset selection
st.sidebar.markdown("### Select Dataset")

available_files = {
    "Training Data": "train_data.csv",
    "Test Data": "test_data.csv",
    "Combined Data": "combined_data.csv",
    "Instagram (Cleaned)": "instagram_cleaned.csv",
    "Daily Aggregated": "daily_aggregated.csv",
}

# Check which files exist
existing_files = {}
for name, filename in available_files.items():
    if (DATA_DIR / filename).exists():
        existing_files[name] = filename

if not existing_files:
    st.error("No data files found! Run preprocessing first:")
    st.code("python scripts/preprocess_datasets.py")
    st.stop()

selected = st.sidebar.selectbox("Dataset", list(existing_files.keys()))
df = load_dataset(existing_files[selected])

if df.empty:
    st.error(f"Could not load {selected}")
    st.stop()

# Dataset info
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", f"{len(df):,}")
with col2:
    st.metric("Columns", len(df.columns))
with col3:
    if 'engagement' in df.columns:
        st.metric("Avg Engagement", f"{df['engagement'].mean():,.0f}")
    elif 'y' in df.columns:
        st.metric("Avg Target", f"{df['y'].mean():.2f}")
    else:
        st.metric("Numeric Cols", len(df.select_dtypes(include=[np.number]).columns))
with col4:
    if 'platform' in df.columns:
        st.metric("Platforms", df['platform'].nunique())
    else:
        st.metric("Missing %", f"{(df.isnull().sum().sum() / df.size * 100):.1f}%")

# Data preview
st.markdown("---")
st.markdown("### Data Preview")

num_rows = st.slider("Rows to display", 5, 50, 10)
st.dataframe(df.head(num_rows), use_container_width=True)

# Column info
st.markdown("---")
st.markdown("### Column Information")

col_info = []
for col in df.columns:
    col_info.append({
        'Column': col,
        'Type': str(df[col].dtype),
        'Non-Null': df[col].notna().sum(),
        'Null %': f"{(df[col].isna().sum() / len(df) * 100):.1f}%",
        'Unique': df[col].nunique()
    })

st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

# Statistics
st.markdown("---")
st.markdown("### Statistics")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

# Visualizations
st.markdown("---")
st.markdown("### Visualizations")

tab1, tab2, tab3 = st.tabs(["Distribution", "Time Series", "Platforms"])

with tab1:
    if 'engagement' in df.columns:
        st.bar_chart(df['engagement'].value_counts().head(30))
    elif 'y' in df.columns:
        st.line_chart(df['y'])

with tab2:
    date_col = None
    for col in ['ds', 'timestamp', 'date']:
        if col in df.columns:
            date_col = col
            break

    value_col = 'engagement' if 'engagement' in df.columns else ('y' if 'y' in df.columns else None)

    if date_col and value_col:
        chart_df = df[[date_col, value_col]].dropna()
        chart_df = chart_df.set_index(date_col)
        st.line_chart(chart_df)
    else:
        st.info("No time series data available")

with tab3:
    if 'platform' in df.columns:
        platform_counts = df['platform'].value_counts()
        st.bar_chart(platform_counts)
    else:
        st.info("No platform column in this dataset")

# Export
st.markdown("---")
st.markdown("### Export Data")

col1, col2 = st.columns(2)

with col1:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download CSV",
        csv,
        f"{selected.lower().replace(' ', '_')}.csv",
        "text/csv"
    )

with col2:
    st.download_button(
        "📥 Download JSON",
        df.head(1000).to_json(orient='records'),
        f"{selected.lower().replace(' ', '_')}.json",
        "application/json"
    )
