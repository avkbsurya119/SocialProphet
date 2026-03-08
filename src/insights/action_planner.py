"""
Action Planner for SocialProphet.

Generates actionable content plans by combining:
- Posting time insights (when to post)
- Content type analysis (what format to use)
- Category performance (what topics work)
- Trend analysis (current momentum)

Output: A specific, actionable plan telling users WHEN to post WHAT.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config
from src.insights.posting_advisor import PostingAdvisor


class ActionPlanner:
    """
    Generate actionable content plans.

    Combines all insights to produce specific recommendations:
    - Exact days and times to post
    - Content format to use
    - Topics/categories to cover
    - Expected engagement
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.advisor = PostingAdvisor(config)
        self.plan = {}

    def generate_weekly_plan(self, n_posts: int = 5) -> Dict[str, Any]:
        """
        Generate a complete weekly posting plan.

        Args:
            n_posts: Number of posts to plan

        Returns:
            Detailed weekly plan with specific recommendations
        """
        # Load and analyze data
        self.advisor.load_instagram_data()
        analysis = self.advisor.analyze_all()

        # Get best posting slots
        best_slots = self._get_best_slots(analysis, n_posts)

        # Get content recommendations
        content_recs = self._get_content_recommendations(analysis)

        # Generate specific plans for each slot
        post_plans = []
        for i, slot in enumerate(best_slots):
            post_plan = self._create_post_plan(
                slot,
                content_recs,
                analysis,
                post_number=i + 1
            )
            post_plans.append(post_plan)

        # Calculate expected total engagement
        total_expected = sum(p['expected_engagement'] for p in post_plans)

        self.plan = {
            'summary': {
                'total_posts': n_posts,
                'expected_engagement': total_expected,
                'best_format': content_recs['best_format'],
                'top_categories': content_recs['top_categories'][:3],
                'generated_at': datetime.now().isoformat()
            },
            'posts': post_plans,
            'quick_tips': self._generate_tips(analysis),
            'warnings': self._generate_warnings(analysis),
            'underlying_insights': {
                'best_hours': analysis.get('best_hours', {}).get('top_5_hours', [])[:3],
                'best_days': analysis.get('best_days', {}).get('top_3_days', [])
            }
        }

        return self.plan

    def _get_best_slots(
        self,
        analysis: Dict[str, Any],
        n_slots: int
    ) -> List[Dict[str, Any]]:
        """Get the best posting slots based on analysis."""
        hour_by_day = analysis.get('best_hour_by_day', {})
        best_days = analysis.get('best_days', {}).get('top_3_days', [])

        # Create slots from best day/hour combinations
        slots = []

        # Priority 1: Best hour on best days
        for day_info in best_days:
            day = day_info['day']
            if day in hour_by_day:
                best_hour = hour_by_day[day]['best_hours'][0]
                slots.append({
                    'day': day,
                    'hour': best_hour['hour'],
                    'time': best_hour['time_12h'],
                    'expected_engagement': best_hour['avg_engagement'],
                    'priority': 'high'
                })

        # If we need more slots, add second-best hours
        if len(slots) < n_slots:
            for day_info in best_days:
                day = day_info['day']
                if day in hour_by_day and len(hour_by_day[day]['best_hours']) > 1:
                    for hour_info in hour_by_day[day]['best_hours'][1:]:
                        if len(slots) >= n_slots:
                            break
                        # Don't duplicate same day/hour
                        if not any(s['day'] == day and s['hour'] == hour_info['hour'] for s in slots):
                            slots.append({
                                'day': day,
                                'hour': hour_info['hour'],
                                'time': hour_info['time_12h'],
                                'expected_engagement': hour_info['avg_engagement'],
                                'priority': 'medium'
                            })

        # Sort by expected engagement
        slots.sort(key=lambda x: x['expected_engagement'], reverse=True)

        return slots[:n_slots]

    def _get_content_recommendations(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get content type and category recommendations."""
        content_perf = analysis.get('content_performance', {})
        category_insights = analysis.get('category_insights', {})

        return {
            'best_format': content_perf.get('best_media_type', {}).get('type', 'reel'),
            'all_formats': [
                {
                    'type': t['type'],
                    'engagement': t['avg_engagement'],
                    'recommendation': self._get_format_recommendation(t['type'])
                }
                for t in content_perf.get('all_types', [])
            ],
            'top_categories': [
                c['category'] for c in category_insights.get('top_categories', [])
            ]
        }

    def _get_format_recommendation(self, format_type: str) -> str:
        """Get recommendation for a content format."""
        recommendations = {
            'reel': 'Best for viral reach and new followers',
            'image': 'Good for high-quality visuals and quotes',
            'carousel': 'Great for educational content and tutorials',
            'video': 'Ideal for storytelling and demonstrations'
        }
        return recommendations.get(format_type, 'Use for variety in your feed')

    def _create_post_plan(
        self,
        slot: Dict[str, Any],
        content_recs: Dict[str, Any],
        analysis: Dict[str, Any],
        post_number: int
    ) -> Dict[str, Any]:
        """Create a detailed plan for a single post."""
        # Rotate through formats for variety
        formats = content_recs.get('all_formats', [])
        format_idx = (post_number - 1) % len(formats) if formats else 0
        recommended_format = formats[format_idx] if formats else {'type': 'reel'}

        # Rotate through categories
        categories = content_recs.get('top_categories', ['General'])
        cat_idx = (post_number - 1) % len(categories) if categories else 0
        recommended_category = categories[cat_idx] if categories else 'General'

        return {
            'post_number': post_number,
            'when': {
                'day': slot['day'],
                'time': slot['time'],
                'hour_24': slot['hour']
            },
            'what': {
                'format': recommended_format.get('type', 'reel'),
                'category': recommended_category,
                'format_tip': self._get_format_recommendation(recommended_format.get('type', 'reel'))
            },
            'expected_engagement': slot['expected_engagement'],
            'priority': slot['priority'],
            'content_ideas': self._generate_content_ideas(recommended_category, recommended_format.get('type', 'reel'))
        }

    def _generate_content_ideas(
        self,
        category: str,
        format_type: str
    ) -> List[str]:
        """Generate content ideas for a category and format."""
        ideas_by_category = {
            'Technology': [
                'Product tip or hack',
                'Behind-the-scenes of your workflow',
                'Quick tutorial or how-to',
                'Tech news commentary'
            ],
            'Fitness': [
                'Quick workout routine',
                'Before/after transformation',
                'Nutrition tip',
                'Motivation quote or story'
            ],
            'Food': [
                'Recipe walkthrough',
                'Restaurant review or discovery',
                'Cooking tip or hack',
                'Food styling showcase'
            ],
            'Travel': [
                'Destination highlight',
                'Travel tip or hack',
                'Hidden gem discovery',
                'Photo dump from trip'
            ],
            'Fashion': [
                'Outfit of the day',
                'Styling tips',
                'Trend commentary',
                'Wardrobe essentials'
            ],
            'Beauty': [
                'Makeup tutorial',
                'Product review',
                'Skincare routine',
                'Before/after look'
            ],
            'Lifestyle': [
                'Day in my life',
                'Productivity tips',
                'Home decor or organization',
                'Self-care routine'
            ],
            'Comedy': [
                'Relatable humor',
                'Trending sound/meme',
                'Parody or sketch',
                'Reaction content'
            ],
            'Music': [
                'Cover or original performance',
                'Practice session clip',
                'Song recommendation',
                'Music production tip'
            ],
            'Photography': [
                'Before/after edit',
                'Location showcase',
                'Gear review',
                'Editing tutorial'
            ]
        }

        default_ideas = [
            'Share your expertise',
            'Behind-the-scenes look',
            'User-generated content feature',
            'Q&A or FAQ response'
        ]

        ideas = ideas_by_category.get(category, default_ideas)

        # Adjust ideas based on format
        if format_type == 'carousel':
            ideas = [f"{idea} (step-by-step)" for idea in ideas[:2]] + ideas[2:]
        elif format_type == 'reel':
            ideas = [f"{idea} (under 30 seconds)" for idea in ideas]

        return ideas[:3]

    def _generate_tips(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate quick tips based on analysis."""
        tips = []

        # Weekend vs weekday tip
        weekend_data = analysis.get('best_days', {}).get('weekend_vs_weekday', {})
        if weekend_data:
            if weekend_data.get('recommendation') == 'weekend':
                tips.append(f"Focus on weekends - they get {weekend_data.get('difference_pct', 0):.0f}% more engagement")
            elif weekend_data.get('recommendation') == 'weekday':
                tips.append("Weekdays perform better - post Tuesday through Thursday for best results")

        # Best hour tip
        best_hours = analysis.get('best_hours', {}).get('top_5_hours', [])
        if best_hours:
            top_hour = best_hours[0]
            tips.append(f"Post around {top_hour['time_12h']} for peak engagement")

        # Content format tip
        content = analysis.get('content_performance', {})
        if content.get('best_media_type'):
            tips.append(f"Prioritize {content['best_media_type']['type']}s - they outperform other formats")

        # Traffic source tip
        traffic = analysis.get('traffic_sources', {}).get('by_source', [])
        if traffic:
            best_source = traffic[0]
            tips.append(f"Optimize for {best_source['source']} - your top traffic source")

        return tips[:5]

    def _generate_warnings(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate warnings based on analysis."""
        warnings = []

        # Low engagement hours warning
        worst_hours = analysis.get('best_hours', {}).get('worst_3_hours', [])
        if worst_hours:
            bad_times = [h['time_12h'] for h in worst_hours[:2]]
            warnings.append(f"Avoid posting at {', '.join(bad_times)} - historically low engagement")

        return warnings

    def get_next_post_recommendation(self) -> Dict[str, Any]:
        """Get recommendation for the very next post."""
        if not self.plan:
            self.generate_weekly_plan()

        posts = self.plan.get('posts', [])
        if not posts:
            return {'error': 'No posts planned'}

        # Return first post as next recommendation
        next_post = posts[0]

        return {
            'action': 'Your next post should be:',
            'when': f"{next_post['when']['day']} at {next_post['when']['time']}",
            'what': f"A {next_post['what']['format']} about {next_post['what']['category']}",
            'ideas': next_post['content_ideas'],
            'expected_engagement': f"{next_post['expected_engagement']:,.0f}",
            'tip': next_post['what']['format_tip']
        }

    def to_readable_plan(self) -> str:
        """Convert plan to human-readable format."""
        if not self.plan:
            self.generate_weekly_plan()

        lines = []
        lines.append("=" * 50)
        lines.append("📅 YOUR WEEKLY CONTENT PLAN")
        lines.append("=" * 50)

        summary = self.plan.get('summary', {})
        lines.append(f"\n📊 OVERVIEW")
        lines.append(f"   Posts planned: {summary.get('total_posts', 0)}")
        lines.append(f"   Expected engagement: {summary.get('expected_engagement', 0):,.0f}")
        lines.append(f"   Best format: {summary.get('best_format', 'N/A')}")

        lines.append(f"\n📝 POST SCHEDULE")
        lines.append("-" * 50)

        for post in self.plan.get('posts', []):
            lines.append(f"\nPost #{post['post_number']} [{post['priority'].upper()} PRIORITY]")
            lines.append(f"   📆 When: {post['when']['day']} at {post['when']['time']}")
            lines.append(f"   🎬 Format: {post['what']['format']}")
            lines.append(f"   📂 Category: {post['what']['category']}")
            lines.append(f"   📈 Expected: {post['expected_engagement']:,.0f} engagement")
            lines.append(f"   💡 Ideas:")
            for idea in post['content_ideas'][:2]:
                lines.append(f"      • {idea}")

        tips = self.plan.get('quick_tips', [])
        if tips:
            lines.append(f"\n💡 QUICK TIPS")
            lines.append("-" * 50)
            for tip in tips:
                lines.append(f"   • {tip}")

        warnings = self.plan.get('warnings', [])
        if warnings:
            lines.append(f"\n⚠️ WARNINGS")
            lines.append("-" * 50)
            for warning in warnings:
                lines.append(f"   • {warning}")

        lines.append("\n" + "=" * 50)

        return '\n'.join(lines)

    def save_plan(self, filepath: Optional[Path] = None) -> Path:
        """Save plan to JSON file."""
        if not self.plan:
            self.generate_weekly_plan()

        if filepath is None:
            filepath = self.config.PROCESSED_DATA_DIR / 'weekly_plan.json'

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.plan, f, indent=2, default=str)

        print(f"Plan saved to: {filepath}")
        return filepath


def main():
    """Generate and display weekly plan."""
    planner = ActionPlanner()

    print("Generating weekly content plan...")
    plan = planner.generate_weekly_plan(n_posts=5)

    print(planner.to_readable_plan())

    # Save plan
    planner.save_plan()

    # Show next post recommendation
    print("\n" + "=" * 50)
    print("🎯 NEXT ACTION")
    print("=" * 50)
    next_rec = planner.get_next_post_recommendation()
    print(f"\n{next_rec['action']}")
    print(f"📅 {next_rec['when']}")
    print(f"🎬 {next_rec['what']}")
    print(f"📈 Expected engagement: {next_rec['expected_engagement']}")
    print(f"💡 {next_rec['tip']}")
    print(f"\nContent ideas:")
    for idea in next_rec['ideas']:
        print(f"   • {idea}")


if __name__ == "__main__":
    main()
