"""
SocialProphet Dashboard - Main Application.

Multi-page Streamlit dashboard for social media engagement forecasting
and AI-powered content generation.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config


def main():
    """Main dashboard application."""
    # Page configuration
    st.set_page_config(
        page_title="SocialProphet",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "SocialProphet - Predict → Generate Pipeline"
        }
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=SocialProphet", width=150)
        st.markdown("---")
        st.markdown("### Navigation")
        st.markdown("""
        - 📊 **Data Overview** - Explore datasets
        - 📈 **Forecasting** - View predictions
        - ✍️ **Content Gen** - Generate posts
        - 🐦 **Twitter Live** - Real-time data
        """)
        st.markdown("---")

        # API Status
        st.markdown("### API Status")
        api_status = Config.validate_api_keys()

        for api, status in api_status.items():
            if status:
                st.success(f"✅ {api.title()}")
            else:
                st.warning(f"⚠️ {api.title()}")

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **SocialProphet** is a hybrid time-series
        forecasting and generative content agent.

        **Predict → Generate Pipeline**
        """)

    # Main content
    st.markdown('<p class="main-header">SocialProphet</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Hybrid Time-Series Forecasting & Generative Content Agent</p>', unsafe_allow_html=True)

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Forecasting MAPE",
            value="12.43%",
            delta="-2.57% vs target",
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="FIIT Score",
            value="0.85",
            delta="Target achieved",
            delta_color="normal"
        )

    with col3:
        st.metric(
            label="Platforms",
            value="8",
            delta="Multi-platform"
        )

    with col4:
        st.metric(
            label="Models",
            value="3",
            delta="Ensemble"
        )

    st.markdown("---")

    # Pipeline overview
    st.markdown("## Pipeline Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Forecasting Engine")
        st.markdown("""
        **Ensemble of 3 models:**
        - **Prophet** (40%) - Trend & Seasonality
        - **SARIMA** (35%) - Statistical Patterns
        - **LSTM** (25%) - Deep Learning

        **Results:**
        - MAPE: 12.43% ✅
        - 30-day forecast horizon
        - Daily engagement predictions
        """)

    with col2:
        st.markdown("### ✍️ Content Generation")
        st.markdown("""
        **LLM-Powered (Llama 3.2):**
        - Insight extraction from forecasts
        - Platform-optimized content
        - Multi-theme support

        **FIIT Validation:**
        - Fluency: 0.97 ✅
        - Interactivity: 0.74 ✅
        - Information: 0.75 ✅
        - Tone: 0.94 ✅
        """)

    st.markdown("---")

    # Quick actions
    st.markdown("## Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 View Data", use_container_width=True):
            st.switch_page("pages/1_data_overview.py")

    with col2:
        if st.button("📈 Forecasts", use_container_width=True):
            st.switch_page("pages/2_forecasting.py")

    with col3:
        if st.button("✍️ Generate Content", use_container_width=True):
            st.switch_page("pages/3_content_gen.py")

    with col4:
        if st.button("🐦 Twitter Live", use_container_width=True):
            st.switch_page("pages/4_twitter_live.py")

    st.markdown("---")

    # Architecture diagram
    st.markdown("## System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    SocialProphet Pipeline                    │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  [Data Sources] ──→ [Preprocessing] ──→ [Feature Engineering]│
    │       │                                        │             │
    │       │         ┌──────────────────────────────┘             │
    │       │         │                                            │
    │       │         ▼                                            │
    │       │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
    │       │   │   Prophet   │  │   SARIMA    │  │    LSTM     │ │
    │       │   │    (40%)    │  │    (35%)    │  │    (25%)    │ │
    │       │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
    │       │          └───────────────┬┘                │        │
    │       │                   ┌──────┴──────┐          │        │
    │       │                   │  ENSEMBLE   │◄─────────┘        │
    │       │                   └──────┬──────┘                   │
    │       │                          │                          │
    │       │                          ▼                          │
    │       │                ┌─────────────────┐                  │
    │       │                │    Insights     │                  │
    │       │                │   Extraction    │                  │
    │       │                └────────┬────────┘                  │
    │       │                         │                           │
    │       │                         ▼                           │
    │       │              ┌─────────────────────┐                │
    │       │              │   LLM Generation    │                │
    │       │              │   (Llama 3.2)       │                │
    │       │              └──────────┬──────────┘                │
    │       │                         │                           │
    │       │                         ▼                           │
    │       │              ┌─────────────────────┐                │
    │       │              │   FIIT Validation   │                │
    │       │              └──────────┬──────────┘                │
    │       │                         │                           │
    │       ▼                         ▼                           │
    │  [Twitter API] ──────→  [Dashboard Output]                  │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>SocialProphet v1.0 | Phase 4 Dashboard | February 2026</p>
        <p>Contributors: navadeep555, avkbsurya119</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
