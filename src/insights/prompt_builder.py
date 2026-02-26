"""
Prompt Builder Module for SocialProphet.

Constructs structured prompts for LLM content generation based on
extracted insights and forecast data.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class PromptBuilder:
    """
    Build structured prompts for content generation.

    Supports multiple platforms and content types with
    FIIT framework integration.
    """

    # Platform-specific configurations
    PLATFORM_CONFIGS = {
        'instagram': {
            'max_caption_length': 2200,
            'optimal_caption_length': 150,
            'max_hashtags': 30,
            'optimal_hashtags': 11,
            'tone': 'casual, visual, aspirational',
            'features': ['emojis', 'hashtags', 'call-to-action']
        },
        'twitter': {
            'max_caption_length': 280,
            'optimal_caption_length': 100,
            'max_hashtags': 3,
            'optimal_hashtags': 2,
            'tone': 'concise, witty, conversational',
            'features': ['hashtags', 'mentions', 'threads']
        },
        'linkedin': {
            'max_caption_length': 3000,
            'optimal_caption_length': 300,
            'max_hashtags': 5,
            'optimal_hashtags': 3,
            'tone': 'professional, insightful, thought-leadership',
            'features': ['statistics', 'industry insights', 'call-to-action']
        },
        'tiktok': {
            'max_caption_length': 150,
            'optimal_caption_length': 80,
            'max_hashtags': 5,
            'optimal_hashtags': 4,
            'tone': 'trendy, fun, authentic',
            'features': ['trending sounds', 'hashtags', 'challenges']
        },
        'facebook': {
            'max_caption_length': 63206,
            'optimal_caption_length': 250,
            'max_hashtags': 3,
            'optimal_hashtags': 2,
            'tone': 'friendly, community-focused, informative',
            'features': ['questions', 'stories', 'call-to-action']
        }
    }

    # Content themes
    CONTENT_THEMES = [
        'educational',
        'inspirational',
        'behind-the-scenes',
        'promotional',
        'user-generated',
        'trending',
        'storytelling',
        'interactive'
    ]

    def __init__(
        self,
        platform: str = 'instagram',
        brand_voice: str = 'friendly and engaging'
    ):
        """
        Initialize PromptBuilder.

        Args:
            platform: Target social media platform
            brand_voice: Brand voice description
        """
        self.platform = platform.lower()
        self.brand_voice = brand_voice
        self.config = self.PLATFORM_CONFIGS.get(
            self.platform,
            self.PLATFORM_CONFIGS['instagram']
        )

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for content generation.

        Returns:
            System prompt string
        """
        return f"""You are an expert social media content strategist specializing in {self.platform}.

Your role is to create engaging, high-performing content that:
1. Follows the FIIT framework (Fluency, Interactivity, Information, Tone)
2. Aligns with the brand voice: {self.brand_voice}
3. Optimizes for engagement based on data-driven insights
4. Follows platform best practices for {self.platform}

Platform Guidelines for {self.platform.upper()}:
- Optimal caption length: {self.config['optimal_caption_length']} characters
- Optimal hashtags: {self.config['optimal_hashtags']}
- Tone: {self.config['tone']}
- Key features: {', '.join(self.config['features'])}

Always provide actionable, ready-to-post content."""

    def build_post_prompt(
        self,
        insights: Dict[str, Any],
        theme: Optional[str] = None,
        topic: Optional[str] = None
    ) -> str:
        """
        Build prompt for generating a single post.

        Args:
            insights: Extracted insights dictionary
            theme: Content theme (optional)
            topic: Specific topic (optional)

        Returns:
            Formatted prompt string
        """
        # Extract key insights
        trend = insights.get('trend_analysis', {})
        temporal = insights.get('temporal_patterns', {})
        predictions = insights.get('predictions_summary', {})
        recommendations = insights.get('recommendations', {})

        # Best posting info
        best_days = temporal.get('best_days', [])
        best_days_str = ', '.join([d['day'] for d in best_days[:3]]) if best_days else 'weekdays'

        # Trend info
        trend_direction = trend.get('direction', 'stable')
        trend_strength = trend.get('strength', 'moderate')

        # Predicted engagement
        pred_mean = predictions.get('mean_predicted', 0)
        pred_range = f"{predictions.get('min_predicted', 0):,.0f} - {predictions.get('max_predicted', 0):,.0f}"

        prompt = f"""Based on the following engagement insights, create an engaging {self.platform} post.

ENGAGEMENT INSIGHTS:
- Trend: {trend_direction} ({trend_strength} momentum)
- Best posting days: {best_days_str}
- Expected engagement: {pred_mean:,.0f} (range: {pred_range})
- Historical average: {trend.get('historical_mean', 0):,.0f}

"""

        if theme:
            prompt += f"CONTENT THEME: {theme}\n"

        if topic:
            prompt += f"TOPIC: {topic}\n"

        prompt += f"""
REQUIREMENTS:
1. Write a caption optimized for {self.platform} (max {self.config['optimal_caption_length']} characters)
2. Include {self.config['optimal_hashtags']} relevant hashtags
3. Add a clear call-to-action
4. Use appropriate emojis (2-4)
5. Match the brand voice: {self.brand_voice}

FIIT FRAMEWORK:
- Fluency: Natural, easy-to-read language
- Interactivity: Include question or CTA to drive engagement
- Information: Provide value to the audience
- Tone: Consistent with brand voice

Please provide:
1. CAPTION: [Your caption here]
2. HASHTAGS: [Your hashtags here]
3. BEST TIME TO POST: [Recommended time based on insights]
4. CONTENT TYPE: [Suggested content type - image/video/carousel]
"""

        return prompt

    def build_campaign_prompt(
        self,
        insights: Dict[str, Any],
        n_posts: int = 5,
        campaign_goal: str = 'engagement',
        duration_days: int = 7
    ) -> str:
        """
        Build prompt for generating a content campaign.

        Args:
            insights: Extracted insights dictionary
            n_posts: Number of posts to generate
            campaign_goal: Campaign objective
            duration_days: Campaign duration in days

        Returns:
            Formatted prompt string
        """
        trend = insights.get('trend_analysis', {})
        temporal = insights.get('temporal_patterns', {})
        seasonality = insights.get('seasonality', {})

        # Weekly pattern
        weekly_pattern = seasonality.get('weekly_pattern', {})
        best_days = sorted(weekly_pattern.items(), key=lambda x: x[1], reverse=True)[:3]
        best_days_str = ', '.join([d[0] for d in best_days]) if best_days else 'varies'

        prompt = f"""Create a {duration_days}-day content campaign for {self.platform} with {n_posts} posts.

CAMPAIGN GOAL: {campaign_goal}

ENGAGEMENT INSIGHTS:
- Current trend: {trend.get('direction', 'stable')} ({trend.get('momentum_7d', 0):.1f}% weekly change)
- Best performing days: {best_days_str}
- Historical average engagement: {trend.get('historical_mean', 0):,.0f}
- Weekend performance: {temporal.get('weekend_vs_weekday', {}).get('better', 'similar')} than weekdays

PLATFORM: {self.platform.upper()}
- Optimal caption length: {self.config['optimal_caption_length']} chars
- Optimal hashtags: {self.config['optimal_hashtags']}
- Tone: {self.config['tone']}

REQUIREMENTS FOR EACH POST:
1. Unique angle/theme
2. Caption with call-to-action
3. Relevant hashtags
4. Suggested posting day/time
5. Content type recommendation

FIIT COMPLIANCE:
- Fluency: 60+ readability score
- Interactivity: Questions or CTAs
- Information: Value-driven content
- Tone: Consistent brand voice ({self.brand_voice})

Generate {n_posts} posts in this format:

POST 1:
- Day: [Recommended day]
- Time: [Recommended time]
- Theme: [Content theme]
- Caption: [Full caption]
- Hashtags: [Hashtags]
- Content Type: [image/video/carousel]
- Expected Impact: [High/Medium/Low]

[Continue for all {n_posts} posts]
"""

        return prompt

    def build_hashtag_prompt(
        self,
        content: str,
        insights: Dict[str, Any],
        count: int = 10
    ) -> str:
        """
        Build prompt for generating hashtags.

        Args:
            content: Post content
            insights: Extracted insights
            count: Number of hashtags to generate

        Returns:
            Formatted prompt string
        """
        prompt = f"""Generate {count} relevant hashtags for this {self.platform} post.

POST CONTENT:
{content}

ENGAGEMENT CONTEXT:
- Platform: {self.platform}
- Target audience: engaged social media users
- Goal: maximize reach and engagement

REQUIREMENTS:
1. Mix of popular and niche hashtags
2. Relevant to content and platform
3. No banned or spam hashtags
4. Include 2-3 branded hashtags if applicable

Generate exactly {count} hashtags in this format:
#hashtag1 #hashtag2 #hashtag3 ...

HASHTAGS:"""

        return prompt

    def build_schedule_prompt(
        self,
        insights: Dict[str, Any],
        n_days: int = 7
    ) -> str:
        """
        Build prompt for generating posting schedule.

        Args:
            insights: Extracted insights
            n_days: Number of days to schedule

        Returns:
            Formatted prompt string
        """
        temporal = insights.get('temporal_patterns', {})
        best_days = temporal.get('best_days', [])
        weekend_info = temporal.get('weekend_vs_weekday', {})

        prompt = f"""Create an optimal posting schedule for {self.platform} for the next {n_days} days.

ENGAGEMENT DATA:
- Best days: {', '.join([d['day'] for d in best_days]) if best_days else 'varies'}
- Weekend vs Weekday: {weekend_info.get('better', 'similar')} performs better
- Engagement difference: {weekend_info.get('difference_pct', 0):.1f}%

PLATFORM: {self.platform}
POSTING FREQUENCY: 1-2 posts per day recommended

Create a schedule showing:
1. Day and date
2. Optimal posting time(s)
3. Suggested content type
4. Priority level (High/Medium/Low)

FORMAT:
DAY 1 (Monday):
- Post 1: [Time] - [Content Type] - [Priority]
- Post 2: [Time] - [Content Type] - [Priority]

[Continue for {n_days} days]
"""

        return prompt

    def build_variation_prompt(
        self,
        original_content: str,
        n_variations: int = 3
    ) -> str:
        """
        Build prompt for generating content variations (A/B testing).

        Args:
            original_content: Original post content
            n_variations: Number of variations to generate

        Returns:
            Formatted prompt string
        """
        prompt = f"""Create {n_variations} variations of this {self.platform} post for A/B testing.

ORIGINAL POST:
{original_content}

REQUIREMENTS:
1. Maintain the core message
2. Vary the hook/opening
3. Try different CTAs
4. Experiment with emoji placement
5. Test different hashtag combinations

For each variation, explain what was changed and why.

FORMAT:
VARIATION 1:
Caption: [Varied caption]
Hashtags: [Varied hashtags]
Changes Made: [What was changed]
Hypothesis: [Why this might perform better]

[Continue for {n_variations} variations]
"""

        return prompt

    def build_improvement_prompt(
        self,
        content: str,
        current_score: Dict[str, float],
        target_improvements: List[str]
    ) -> str:
        """
        Build prompt for improving existing content.

        Args:
            content: Current content to improve
            current_score: Current FIIT scores
            target_improvements: Areas to improve

        Returns:
            Formatted prompt string
        """
        prompt = f"""Improve this {self.platform} post based on FIIT framework analysis.

CURRENT POST:
{content}

CURRENT FIIT SCORES:
- Fluency: {current_score.get('fluency', 0):.2f}
- Interactivity: {current_score.get('interactivity', 0):.2f}
- Information: {current_score.get('information', 0):.2f}
- Tone: {current_score.get('tone', 0):.2f}
- Overall: {current_score.get('overall', 0):.2f}

TARGET IMPROVEMENTS: {', '.join(target_improvements)}

REQUIREMENTS:
1. Improve weak areas while maintaining strengths
2. Keep the core message intact
3. Optimize for {self.platform} best practices
4. Target overall FIIT score > 0.85

Provide:
1. IMPROVED CAPTION: [Enhanced version]
2. IMPROVED HASHTAGS: [Optimized hashtags]
3. CHANGES EXPLAINED: [What was improved and why]
4. EXPECTED SCORE IMPROVEMENT: [Projected new scores]
"""

        return prompt

    def format_insights_context(self, insights: Dict[str, Any]) -> str:
        """
        Format insights into a context block for prompts.

        Args:
            insights: Insights dictionary

        Returns:
            Formatted context string
        """
        lines = []
        lines.append("=" * 50)
        lines.append("ENGAGEMENT INSIGHTS SUMMARY")
        lines.append("=" * 50)

        # Trend
        trend = insights.get('trend_analysis', {})
        lines.append(f"\nTREND ANALYSIS:")
        lines.append(f"  Direction: {trend.get('direction', 'unknown')}")
        lines.append(f"  Strength: {trend.get('strength', 'unknown')}")
        lines.append(f"  7-day momentum: {trend.get('momentum_7d', 0):.1f}%")
        lines.append(f"  Historical mean: {trend.get('historical_mean', 0):,.0f}")

        # Temporal
        temporal = insights.get('temporal_patterns', {})
        best_days = temporal.get('best_days', [])
        if best_days:
            lines.append(f"\nBEST POSTING DAYS:")
            for day in best_days[:3]:
                lines.append(f"  - {day['day']}: {day['avg_engagement']:,.0f} avg engagement")

        # Weekend vs Weekday
        ww = temporal.get('weekend_vs_weekday', {})
        if ww:
            lines.append(f"\nWEEKEND VS WEEKDAY:")
            lines.append(f"  Better: {ww.get('better', 'unknown')}")
            lines.append(f"  Difference: {ww.get('difference_pct', 0):.1f}%")

        # Predictions
        preds = insights.get('predictions_summary', {})
        if preds:
            lines.append(f"\nPREDICTED ENGAGEMENT:")
            lines.append(f"  Average: {preds.get('mean_predicted', 0):,.0f}")
            lines.append(f"  Range: {preds.get('min_predicted', 0):,.0f} - {preds.get('max_predicted', 0):,.0f}")

            peak = preds.get('peak', {})
            if peak.get('date'):
                lines.append(f"  Peak day: {peak.get('day_of_week', 'unknown')} ({peak.get('value', 0):,.0f})")

        lines.append("\n" + "=" * 50)

        return '\n'.join(lines)

    def set_platform(self, platform: str):
        """
        Change target platform.

        Args:
            platform: New target platform
        """
        platform = platform.lower()
        if platform not in self.PLATFORM_CONFIGS:
            print(f"Warning: Unknown platform '{platform}'. Using default config.")

        self.platform = platform
        self.config = self.PLATFORM_CONFIGS.get(
            platform,
            self.PLATFORM_CONFIGS['instagram']
        )

    def set_brand_voice(self, brand_voice: str):
        """
        Update brand voice.

        Args:
            brand_voice: New brand voice description
        """
        self.brand_voice = brand_voice

    def get_platform_config(self) -> Dict[str, Any]:
        """
        Get current platform configuration.

        Returns:
            Platform configuration dictionary
        """
        return {
            'platform': self.platform,
            'config': self.config,
            'brand_voice': self.brand_voice
        }
