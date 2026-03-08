"""
Posting Advisor Page - SocialProphet Dashboard.

Provides actionable recommendations on WHEN to post WHAT content.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Posting Advisor - SocialProphet", page_icon="🎯", layout="wide")

st.title("🎯 Posting Advisor")
st.markdown("**Your personalized content strategy** - Know exactly WHEN to post WHAT for maximum engagement.")

# Load or generate insights
DATA_DIR = PROJECT_ROOT / "data" / "processed"
INSIGHTS_PATH = DATA_DIR / "posting_insights.json"
PLAN_PATH = DATA_DIR / "weekly_plan.json"

@st.cache_data
def load_insights():
    """Load pre-computed insights or generate new ones."""
    if INSIGHTS_PATH.exists():
        with open(INSIGHTS_PATH) as f:
            return json.load(f)
    return None

@st.cache_data
def load_weekly_plan():
    """Load pre-computed weekly plan."""
    if PLAN_PATH.exists():
        with open(PLAN_PATH) as f:
            return json.load(f)
    return None

@st.cache_data
def generate_insights():
    """Generate fresh insights from data."""
    try:
        from src.insights.posting_advisor import PostingAdvisor
        advisor = PostingAdvisor()
        advisor.load_instagram_data()
        analysis = advisor.analyze_all()
        advisor.save_analysis()
        return analysis
    except Exception as e:
        st.error(f"Error generating insights: {e}")
        return None

@st.cache_data
def generate_weekly_plan(n_posts=5):
    """Generate fresh weekly plan."""
    try:
        from src.insights.action_planner import ActionPlanner
        planner = ActionPlanner()
        plan = planner.generate_weekly_plan(n_posts=n_posts)
        planner.save_plan()
        return plan
    except Exception as e:
        st.error(f"Error generating plan: {e}")
        return None

# Try loading existing, otherwise generate
insights = load_insights()
if insights is None:
    with st.spinner("Analyzing your posting data..."):
        insights = generate_insights()

weekly_plan = load_weekly_plan()
if weekly_plan is None:
    with st.spinner("Generating your weekly plan..."):
        weekly_plan = generate_weekly_plan()

if insights is None:
    st.error("Unable to load or generate posting insights. Please check your data files.")
    st.stop()

# WEEKLY ACTION PLAN - Main Feature
st.markdown("---")
st.markdown("## 📋 Your Weekly Content Plan")
st.markdown("*Specific posts planned based on your best-performing patterns*")

if weekly_plan and weekly_plan.get('posts'):
    plan_summary = weekly_plan.get('summary', {})

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Posts This Week", plan_summary.get('total_posts', 0))
    with col2:
        st.metric("Expected Engagement", f"{plan_summary.get('expected_engagement', 0):,.0f}")
    with col3:
        st.metric("Best Format", plan_summary.get('best_format', 'N/A').title())
    with col4:
        top_cats = plan_summary.get('top_categories', ['N/A'])
        st.metric("Top Category", top_cats[0] if top_cats else 'N/A')

    st.markdown("### 📅 Post Schedule")

    for post in weekly_plan.get('posts', []):
        priority_badge = "🔴" if post['priority'] == 'high' else "🟡" if post['priority'] == 'medium' else "🟢"

        with st.expander(f"{priority_badge} Post #{post['post_number']}: {post['when']['day']} at {post['when']['time']}", expanded=post['post_number'] == 1):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**When:** {post['when']['day']} at {post['when']['time']}")
                st.markdown(f"**Format:** {post['what']['format'].title()}")
                st.markdown(f"**Category:** {post['what']['category']}")
                st.markdown(f"**Tip:** {post['what']['format_tip']}")

                st.markdown("**Content Ideas:**")
                for idea in post.get('content_ideas', []):
                    st.markdown(f"- {idea}")

            with col2:
                st.metric("Expected Engagement", f"{post['expected_engagement']:,.0f}")
                st.markdown(f"**Priority:** {post['priority'].upper()}")

    # Tips and warnings
    tips = weekly_plan.get('quick_tips', [])
    warnings = weekly_plan.get('warnings', [])

    if tips:
        st.markdown("### 💡 Pro Tips")
        for tip in tips:
            st.info(tip)

    if warnings:
        st.markdown("### ⚠️ Things to Avoid")
        for warning in warnings:
            st.warning(warning)

else:
    st.info("No weekly plan available. Click 'Refresh Analysis' below to generate one.")

# Quick Wins Section
st.markdown("---")
st.markdown("## 🚀 Quick Wins")
st.markdown("Immediate actions to boost your engagement.")

quick_wins = insights.get('quick_wins', [])
if quick_wins:
    cols = st.columns(len(quick_wins))
    for i, win in enumerate(quick_wins):
        with cols[i]:
            priority_color = "🔴" if win['priority'] == 'high' else "🟡" if win['priority'] == 'medium' else "🟢"
            st.markdown(f"### {priority_color} {win['title']}")
            st.markdown(f"**{win['insight']}**")
            st.caption(f"Impact: {win['impact']}")
else:
    st.info("No quick wins identified yet.")

# Optimal Schedule
st.markdown("---")
st.markdown("## 📅 Optimal Posting Schedule")

schedule = insights.get('optimal_schedule', {})
if schedule:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### {schedule.get('summary', 'No schedule available')}")

        if schedule.get('schedule'):
            schedule_df = pd.DataFrame(schedule['schedule'])
            schedule_df = schedule_df.rename(columns={
                'day': 'Day',
                'time': 'Best Time',
                'expected_engagement': 'Expected Engagement',
                'content_type': 'Content Type'
            })
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)

    with col2:
        st.metric("Posts per Week", schedule.get('recommended_posts_per_week', 3))
        if schedule.get('schedule'):
            avg_eng = sum(s['expected_engagement'] for s in schedule['schedule']) / len(schedule['schedule'])
            st.metric("Avg Expected Engagement", f"{avg_eng:,.0f}")

# Best Hours Analysis
st.markdown("---")
st.markdown("## ⏰ Best Posting Hours")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top Hours")
    best_hours = insights.get('best_hours', {}).get('top_5_hours', [])
    if best_hours:
        hours_df = pd.DataFrame(best_hours[:5])
        hours_df = hours_df[['time_12h', 'avg_engagement', 'recommendation']]
        hours_df.columns = ['Time', 'Avg Engagement', 'Why']
        st.dataframe(hours_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### Hours to Avoid")
    worst_hours = insights.get('best_hours', {}).get('worst_3_hours', [])
    if worst_hours:
        avoid_df = pd.DataFrame(worst_hours)
        avoid_df = avoid_df[['time_12h', 'avg_engagement']]
        avoid_df.columns = ['Time', 'Avg Engagement']
        st.dataframe(avoid_df, use_container_width=True, hide_index=True)

# Hourly Distribution Chart
hourly_data = insights.get('best_hours', {}).get('hourly_distribution', [])
if hourly_data:
    st.markdown("### Engagement by Hour")
    hourly_df = pd.DataFrame(hourly_data)
    hourly_df['time_label'] = hourly_df['post_hour'].apply(
        lambda h: f"{h}:00" if h >= 10 else f"0{h}:00"
    )
    chart_df = hourly_df[['time_label', 'engagement_mean']].set_index('time_label')
    st.bar_chart(chart_df)

# Best Days
st.markdown("---")
st.markdown("## 📆 Best Days to Post")

col1, col2 = st.columns(2)

with col1:
    best_days = insights.get('best_days', {})
    if best_days.get('top_3_days'):
        st.markdown("### Top Days")
        days_df = pd.DataFrame(best_days['top_3_days'])
        days_df = days_df[['day', 'avg_engagement', 'sample_size']]
        days_df.columns = ['Day', 'Avg Engagement', 'Posts Analyzed']
        st.dataframe(days_df, use_container_width=True, hide_index=True)

with col2:
    weekend_data = best_days.get('weekend_vs_weekday', {})
    if weekend_data:
        st.markdown("### Weekend vs Weekday")
        better = weekend_data.get('recommendation', 'both')
        diff = weekend_data.get('difference_pct', 0)

        if better == 'weekend':
            st.success(f"Weekends perform {diff:.1f}% better!")
        elif better == 'weekday':
            st.info(f"Weekdays perform {-diff:.1f}% better")
        else:
            st.info("Similar performance on weekends and weekdays")

        wcol1, wcol2 = st.columns(2)
        with wcol1:
            st.metric("Weekend Avg", f"{weekend_data.get('weekend_avg', 0):,.0f}")
        with wcol2:
            st.metric("Weekday Avg", f"{weekend_data.get('weekday_avg', 0):,.0f}")

# Daily breakdown chart
daily_breakdown = best_days.get('daily_breakdown', [])
if daily_breakdown:
    st.markdown("### Engagement by Day")
    daily_df = pd.DataFrame(daily_breakdown)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_df['day_order'] = daily_df['day'].map({d: i for i, d in enumerate(day_order)})
    daily_df = daily_df.sort_values('day_order')
    chart_df = daily_df[['day', 'avg_engagement']].set_index('day')
    st.bar_chart(chart_df)

# Content Performance
st.markdown("---")
st.markdown("## 🎨 Content Performance")

col1, col2 = st.columns(2)

with col1:
    content_perf = insights.get('content_performance', {})
    best_media = content_perf.get('best_media_type', {})

    if best_media:
        st.markdown(f"### Best Format: {best_media.get('type', 'N/A').upper()}")
        st.metric("Avg Engagement", f"{best_media.get('avg_engagement', 0):,.0f}")
        st.caption(content_perf.get('recommendation', ''))

    all_types = content_perf.get('all_types', [])
    if all_types:
        st.markdown("### All Formats")
        types_df = pd.DataFrame(all_types)
        types_df = types_df[['type', 'avg_engagement', 'avg_reach', 'sample_size']]
        types_df.columns = ['Type', 'Avg Engagement', 'Avg Reach', 'Posts']
        st.dataframe(types_df, use_container_width=True, hide_index=True)

with col2:
    categories = insights.get('category_insights', {}).get('top_categories', [])
    if categories:
        st.markdown("### Top Content Categories")
        cat_df = pd.DataFrame(categories)
        cat_df = cat_df[['category', 'avg_engagement', 'sample_size']]
        cat_df.columns = ['Category', 'Avg Engagement', 'Posts']
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

# Traffic Sources
st.markdown("---")
st.markdown("## 🔗 Traffic Sources")

traffic = insights.get('traffic_sources', {}).get('by_source', [])
if traffic:
    traffic_df = pd.DataFrame(traffic)
    traffic_df = traffic_df[['source', 'avg_engagement', 'avg_reach', 'sample_size']]
    traffic_df.columns = ['Source', 'Avg Engagement', 'Avg Reach', 'Posts']
    st.dataframe(traffic_df, use_container_width=True, hide_index=True)

# Hour by Day Matrix
st.markdown("---")
st.markdown("## 🗓️ Best Hour by Day")
st.caption("The optimal posting time varies by day of the week.")

hour_by_day = insights.get('best_hour_by_day', {})
if hour_by_day:
    matrix_data = []
    for day, data in hour_by_day.items():
        if data.get('best_hours'):
            best = data['best_hours'][0]
            matrix_data.append({
                'Day': day,
                'Best Time': best['time_12h'],
                'Expected Engagement': f"{best['avg_engagement']:.0f}",
                'Posts Analyzed': data['sample_size']
            })

    if matrix_data:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        matrix_df = pd.DataFrame(matrix_data)
        matrix_df['order'] = matrix_df['Day'].map({d: i for i, d in enumerate(day_order)})
        matrix_df = matrix_df.sort_values('order').drop('order', axis=1)
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

# Actionable Summary
st.markdown("---")
st.markdown("## 📋 Your Action Plan")

action_col1, action_col2 = st.columns(2)

with action_col1:
    st.markdown("### This Week's Posts")
    if schedule.get('schedule'):
        for i, slot in enumerate(schedule['schedule'], 1):
            st.markdown(f"""
            **Post {i}:**
            - 📅 **{slot['day']}** at **{slot['time']}**
            - 🎬 Format: **{slot['content_type']}**
            - 📈 Expected: **{slot['expected_engagement']:,.0f}** engagement
            """)

with action_col2:
    st.markdown("### Content Tips")
    if content_perf.get('best_media_type'):
        st.markdown(f"- Focus on **{best_media['type']}s** for best results")
    if categories:
        top_cats = [c['category'] for c in categories[:3]]
        st.markdown(f"- Top categories: **{', '.join(top_cats)}**")
    if best_hours:
        st.markdown(f"- Peak hour: **{best_hours[0]['time_12h']}**")
    if worst_hours:
        avoid = [h['time_12h'] for h in worst_hours[:2]]
        st.markdown(f"- Avoid posting at: **{', '.join(avoid)}**")

# Refresh button
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Refresh Analysis & Plan", use_container_width=True):
        st.cache_data.clear()
        with st.spinner("Regenerating insights..."):
            insights = generate_insights()
        with st.spinner("Regenerating weekly plan..."):
            weekly_plan = generate_weekly_plan()
        st.rerun()

# Metadata
with st.expander("Analysis Details"):
    metadata = insights.get('metadata', {})
    st.json(metadata)
