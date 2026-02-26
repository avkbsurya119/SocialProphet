"""
Content Generator Module for SocialProphet.

Generates social media content using LLM based on forecast insights.
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class ContentGenerator:
    """
    Generate social media content using LLM.

    Combines insights extraction, prompt building, and LLM generation
    to create optimized social media posts.
    """

    def __init__(
        self,
        llm_client: Any,
        prompt_builder: Any,
        config: Optional[Config] = None
    ):
        """
        Initialize ContentGenerator.

        Args:
            llm_client: HuggingFace client instance
            prompt_builder: PromptBuilder instance
            config: Configuration object (optional)
        """
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.config = config or Config()
        self.generated_content = []

    def generate_post(
        self,
        insights: Dict[str, Any],
        theme: Optional[str] = None,
        topic: Optional[str] = None,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate a single social media post.

        Args:
            insights: Extracted insights dictionary
            theme: Content theme (optional)
            topic: Specific topic (optional)
            max_tokens: Maximum tokens for generation

        Returns:
            Dictionary with generated content
        """
        # Build prompt
        system_prompt = self.prompt_builder.get_system_prompt()
        post_prompt = self.prompt_builder.build_post_prompt(insights, theme, topic)

        full_prompt = f"{system_prompt}\n\n{post_prompt}"

        # Generate content (disable cache for variety)
        print("Generating post content...")
        raw_response = self.llm.generate(
            full_prompt,
            max_new_tokens=max_tokens,
            temperature=0.8,  # Higher for more variety
            use_cache=False   # Disable cache for unique posts
        )

        # Parse response
        parsed = self._parse_post_response(raw_response)
        parsed['raw_response'] = raw_response
        parsed['theme'] = theme
        parsed['topic'] = topic
        parsed['platform'] = self.prompt_builder.platform
        parsed['generated_at'] = datetime.now().isoformat()

        # Store generated content
        self.generated_content.append(parsed)

        return parsed

    def generate_campaign(
        self,
        insights: Dict[str, Any],
        n_posts: int = 5,
        campaign_goal: str = 'engagement',
        duration_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Generate a content campaign with multiple posts.

        Args:
            insights: Extracted insights dictionary
            n_posts: Number of posts to generate
            campaign_goal: Campaign objective
            duration_days: Campaign duration in days

        Returns:
            List of generated posts
        """
        system_prompt = self.prompt_builder.get_system_prompt()
        campaign_prompt = self.prompt_builder.build_campaign_prompt(
            insights, n_posts, campaign_goal, duration_days
        )

        full_prompt = f"{system_prompt}\n\n{campaign_prompt}"

        print(f"Generating {n_posts}-post campaign...")
        raw_response = self.llm.generate(
            full_prompt,
            max_new_tokens=1500,
            temperature=0.8
        )

        # Parse campaign response
        posts = self._parse_campaign_response(raw_response, n_posts)

        # Add metadata
        campaign = {
            'campaign_goal': campaign_goal,
            'duration_days': duration_days,
            'n_posts': len(posts),
            'platform': self.prompt_builder.platform,
            'posts': posts,
            'raw_response': raw_response,
            'generated_at': datetime.now().isoformat()
        }

        return campaign

    def generate_variations(
        self,
        original_content: str,
        n_variations: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate content variations for A/B testing.

        Args:
            original_content: Original post content
            n_variations: Number of variations to generate

        Returns:
            List of variation dictionaries
        """
        prompt = self.prompt_builder.build_variation_prompt(
            original_content, n_variations
        )

        print(f"Generating {n_variations} variations...")
        raw_response = self.llm.generate(
            prompt,
            max_new_tokens=800,
            temperature=0.8
        )

        variations = self._parse_variations_response(raw_response, n_variations)

        return {
            'original': original_content,
            'variations': variations,
            'n_variations': len(variations),
            'raw_response': raw_response,
            'generated_at': datetime.now().isoformat()
        }

    def generate_hashtags(
        self,
        content: str,
        insights: Dict[str, Any],
        count: int = 10
    ) -> List[str]:
        """
        Generate hashtags for content.

        Args:
            content: Post content
            insights: Extracted insights
            count: Number of hashtags to generate

        Returns:
            List of hashtags
        """
        prompt = self.prompt_builder.build_hashtag_prompt(content, insights, count)

        raw_response = self.llm.generate(
            prompt,
            max_new_tokens=150,
            temperature=0.6
        )

        # Extract hashtags
        hashtags = self._extract_hashtags(raw_response)

        return hashtags[:count]

    def generate_schedule(
        self,
        insights: Dict[str, Any],
        n_days: int = 7
    ) -> Dict[str, Any]:
        """
        Generate posting schedule.

        Args:
            insights: Extracted insights
            n_days: Number of days to schedule

        Returns:
            Schedule dictionary
        """
        prompt = self.prompt_builder.build_schedule_prompt(insights, n_days)

        raw_response = self.llm.generate(
            prompt,
            max_new_tokens=600,
            temperature=0.5
        )

        schedule = self._parse_schedule_response(raw_response, n_days)

        return {
            'n_days': n_days,
            'platform': self.prompt_builder.platform,
            'schedule': schedule,
            'raw_response': raw_response,
            'generated_at': datetime.now().isoformat()
        }

    def improve_content(
        self,
        content: str,
        current_scores: Dict[str, float],
        target_improvements: List[str]
    ) -> Dict[str, Any]:
        """
        Improve existing content based on FIIT scores.

        Args:
            content: Current content to improve
            current_scores: Current FIIT scores
            target_improvements: Areas to improve

        Returns:
            Improved content dictionary
        """
        prompt = self.prompt_builder.build_improvement_prompt(
            content, current_scores, target_improvements
        )

        raw_response = self.llm.generate(
            prompt,
            max_new_tokens=600,
            temperature=0.6
        )

        improved = self._parse_improvement_response(raw_response)
        improved['original'] = content
        improved['original_scores'] = current_scores
        improved['target_improvements'] = target_improvements

        return improved

    def _parse_post_response(self, response: str) -> Dict[str, Any]:
        """Parse generated post response."""
        result = {
            'caption': '',
            'hashtags': [],
            'best_time': '',
            'content_type': ''
        }

        lines = response.split('\n')

        # First, try to find "EXAMPLE POST" section - it's usually better
        example_caption = self._extract_example_post(response)
        if example_caption and len(example_caption) > 100:
            result['caption'] = example_caption
            result['hashtags'] = self._extract_hashtags(example_caption)
            # Still try to get best time and content type
            for line in lines:
                line_stripped = line.strip()
                if 'BEST TIME' in line_stripped.upper() or 'TIME TO POST' in line_stripped.upper():
                    if ':' in line_stripped:
                        result['best_time'] = line_stripped.split(':', 1)[-1].strip()
                elif 'CONTENT TYPE' in line_stripped.upper():
                    if ':' in line_stripped:
                        result['content_type'] = line_stripped.split(':', 1)[-1].strip()
            return result

        # Fallback: collect multi-line caption from CAPTION section
        caption_lines = []
        current_section = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_upper = line_stripped.upper()

            # Detect section headers
            if 'CAPTION:' in line_upper or line_stripped.startswith('**Caption'):
                current_section = 'caption'
                if ':' in line_stripped:
                    content = line_stripped.split(':', 1)[-1].strip()
                    content = content.strip('*').strip('"').strip()
                    if content and len(content) > 5:
                        caption_lines.append(content)
                continue

            elif 'HASHTAGS:' in line_upper or line_stripped.startswith('**Hashtags'):
                current_section = 'hashtags'
                content = line_stripped.split(':', 1)[-1].strip() if ':' in line_stripped else ''
                if content:
                    result['hashtags'] = self._extract_hashtags(content)
                continue

            elif 'BEST TIME' in line_upper or 'TIME TO POST' in line_upper:
                current_section = 'time'
                content = line_stripped.split(':', 1)[-1].strip() if ':' in line_stripped else ''
                if content:
                    result['best_time'] = content
                continue

            elif 'CONTENT TYPE' in line_upper:
                current_section = 'type'
                content = line_stripped.split(':', 1)[-1].strip() if ':' in line_stripped else ''
                if content:
                    result['content_type'] = content
                continue

            # Collect content for current section
            if current_section == 'caption' and line_stripped:
                if line_stripped.startswith('**') and ':' in line_stripped:
                    current_section = None
                elif line_stripped.startswith('#') and len(caption_lines) > 0:
                    result['hashtags'].extend(self._extract_hashtags(line_stripped))
                else:
                    clean_line = line_stripped.strip('"').strip('*').strip()
                    if clean_line and not clean_line.startswith('**'):
                        caption_lines.append(clean_line)

            elif current_section == 'hashtags' and line_stripped:
                if line_stripped.startswith('#'):
                    result['hashtags'].extend(self._extract_hashtags(line_stripped))
                elif line_stripped.startswith('**'):
                    current_section = None

        # Join caption lines
        if caption_lines:
            result['caption'] = ' '.join(caption_lines)
            result['caption'] = result['caption'].replace('  ', ' ').strip()

        # Extract hashtags from full response if none found
        if not result['hashtags']:
            result['hashtags'] = self._extract_hashtags(response)

        return result

    def _extract_example_post(self, response: str) -> str:
        """Extract content from EXAMPLE POST section if present."""
        lines = response.split('\n')
        example_lines = []
        in_example = False

        for line in lines:
            line_stripped = line.strip()

            if 'EXAMPLE POST' in line_stripped.upper():
                in_example = True
                continue

            if in_example:
                # Stop at next major section
                if line_stripped.startswith('**') and ':' in line_stripped:
                    if 'TONE' in line_stripped.upper() or 'IMAGE' in line_stripped.upper():
                        break

                # Skip empty lines at start
                if not example_lines and not line_stripped:
                    continue

                # Clean and add line
                clean = line_stripped.strip('"').strip()
                if clean:
                    example_lines.append(clean)

        if example_lines:
            return '\n'.join(example_lines)
        return ''

    def _parse_campaign_response(
        self,
        response: str,
        expected_posts: int
    ) -> List[Dict[str, Any]]:
        """Parse campaign response into individual posts."""
        posts = []

        # Split by POST markers
        post_pattern = r'POST\s*\d+'
        sections = re.split(post_pattern, response, flags=re.IGNORECASE)

        for section in sections[1:]:  # Skip first empty section
            post = self._parse_post_section(section)
            if post.get('caption') or post.get('theme'):
                posts.append(post)

            if len(posts) >= expected_posts:
                break

        return posts

    def _parse_post_section(self, section: str) -> Dict[str, Any]:
        """Parse a single post section from campaign."""
        post = {
            'day': '',
            'time': '',
            'theme': '',
            'caption': '',
            'hashtags': [],
            'content_type': '',
            'expected_impact': ''
        }

        lines = section.split('\n')
        current_field = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for field markers
            lower_line = line.lower()

            if 'day:' in lower_line:
                post['day'] = line.split(':', 1)[-1].strip()
            elif 'time:' in lower_line:
                post['time'] = line.split(':', 1)[-1].strip()
            elif 'theme:' in lower_line:
                post['theme'] = line.split(':', 1)[-1].strip()
            elif 'caption:' in lower_line:
                current_field = 'caption'
                content = line.split(':', 1)[-1].strip()
                if content:
                    post['caption'] = content
            elif 'hashtag' in lower_line:
                post['hashtags'] = self._extract_hashtags(line)
            elif 'content type' in lower_line:
                post['content_type'] = line.split(':', 1)[-1].strip()
            elif 'impact' in lower_line or 'expected' in lower_line:
                post['expected_impact'] = line.split(':', 1)[-1].strip()
            elif current_field == 'caption' and not post['caption']:
                post['caption'] = line

        return post

    def _parse_variations_response(
        self,
        response: str,
        expected_count: int
    ) -> List[Dict[str, Any]]:
        """Parse variations response."""
        variations = []

        # Split by VARIATION markers
        var_pattern = r'VARIATION\s*\d+'
        sections = re.split(var_pattern, response, flags=re.IGNORECASE)

        for section in sections[1:]:
            var = {
                'caption': '',
                'hashtags': [],
                'changes': '',
                'hypothesis': ''
            }

            lines = section.split('\n')
            current_field = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                lower = line.lower()

                if 'caption:' in lower:
                    current_field = 'caption'
                    var['caption'] = line.split(':', 1)[-1].strip()
                elif 'hashtag' in lower:
                    var['hashtags'] = self._extract_hashtags(line)
                elif 'change' in lower:
                    var['changes'] = line.split(':', 1)[-1].strip()
                elif 'hypothesis' in lower:
                    var['hypothesis'] = line.split(':', 1)[-1].strip()
                elif current_field == 'caption' and not var['caption']:
                    var['caption'] = line

            if var['caption']:
                variations.append(var)

            if len(variations) >= expected_count:
                break

        return variations

    def _parse_schedule_response(
        self,
        response: str,
        n_days: int
    ) -> List[Dict[str, Any]]:
        """Parse schedule response."""
        schedule = []

        # Split by DAY markers
        day_pattern = r'DAY\s*\d+'
        sections = re.split(day_pattern, response, flags=re.IGNORECASE)

        for i, section in enumerate(sections[1:], 1):
            day_schedule = {
                'day_number': i,
                'posts': []
            }

            lines = section.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('-'):
                    continue

                # Try to extract time and content type
                if 'post' in line.lower() or ':' in line:
                    post_info = {
                        'time': '',
                        'content_type': '',
                        'priority': 'Medium'
                    }

                    # Extract time (look for patterns like 9:00, 9AM, etc.)
                    time_match = re.search(r'\d{1,2}:\d{2}\s*(AM|PM)?|\d{1,2}\s*(AM|PM)', line, re.IGNORECASE)
                    if time_match:
                        post_info['time'] = time_match.group()

                    # Extract content type
                    for ctype in ['image', 'video', 'carousel', 'story', 'reel']:
                        if ctype in line.lower():
                            post_info['content_type'] = ctype
                            break

                    # Extract priority
                    for priority in ['high', 'medium', 'low']:
                        if priority in line.lower():
                            post_info['priority'] = priority.capitalize()
                            break

                    if post_info['time'] or post_info['content_type']:
                        day_schedule['posts'].append(post_info)

            if day_schedule['posts']:
                schedule.append(day_schedule)

            if len(schedule) >= n_days:
                break

        return schedule

    def _parse_improvement_response(self, response: str) -> Dict[str, Any]:
        """Parse improvement response."""
        result = {
            'improved_caption': '',
            'improved_hashtags': [],
            'changes_explained': '',
            'expected_improvement': ''
        }

        lines = response.split('\n')
        current_field = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            lower = line.lower()

            if 'improved caption' in lower or 'caption:' in lower:
                current_field = 'caption'
                content = line.split(':', 1)[-1].strip()
                if content:
                    result['improved_caption'] = content
            elif 'hashtag' in lower:
                result['improved_hashtags'] = self._extract_hashtags(line)
            elif 'change' in lower or 'explained' in lower:
                result['changes_explained'] = line.split(':', 1)[-1].strip()
            elif 'expected' in lower or 'score' in lower:
                result['expected_improvement'] = line.split(':', 1)[-1].strip()
            elif current_field == 'caption' and not result['improved_caption']:
                result['improved_caption'] = line

        return result

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        # Find all hashtags
        hashtags = re.findall(r'#\w+', text)

        # Clean and deduplicate
        cleaned = []
        seen = set()
        for tag in hashtags:
            tag_lower = tag.lower()
            if tag_lower not in seen:
                seen.add(tag_lower)
                cleaned.append(tag)

        return cleaned

    def get_generated_content(self) -> List[Dict[str, Any]]:
        """Get all generated content."""
        return self.generated_content

    def clear_history(self):
        """Clear generated content history."""
        self.generated_content = []

    def save_content(
        self,
        filepath: Optional[Path] = None
    ) -> Path:
        """
        Save generated content to JSON file.

        Args:
            filepath: Output file path (optional)

        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = Path(self.config.PROCESSED_DATA_DIR) / 'generated_content.json'

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.generated_content, f, indent=2, default=str)

        print(f"Content saved to: {filepath}")
        return filepath
