"""
FIIT Validator Module for SocialProphet.

Implements the FIIT framework for content quality validation:
- Fluency: Readability and natural language flow
- Interactivity: CTAs, questions, engagement hooks
- Information: Relevance and value to audience
- Tone: Sentiment and brand consistency
"""

import re
import math
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import Counter

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    print("Warning: textstat not installed. Install with: pip install textstat")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("Warning: textblob not installed. Install with: pip install textblob")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class FIITValidator:
    """
    FIIT Framework validator for content quality assessment.

    Scores content on four dimensions:
    - Fluency (F): Readability and language quality
    - Interactivity (I): Engagement elements
    - Information (I): Value and relevance
    - Tone (T): Sentiment and consistency

    Target: > 85% overall score
    """

    # Threshold for each dimension
    THRESHOLDS = {
        'fluency': 0.70,
        'interactivity': 0.70,
        'information': 0.70,
        'tone': 0.70
    }

    # Weights for overall score
    WEIGHTS = {
        'fluency': 0.25,
        'interactivity': 0.30,
        'information': 0.25,
        'tone': 0.20
    }

    TARGET_OVERALL = 0.85

    # CTA patterns
    CTA_PATTERNS = [
        r'\b(click|tap|swipe|check|visit|follow|subscribe|sign up|join|shop|buy|get|grab|download|learn|discover|explore|read|watch|listen|try|start|book|order|save|claim|unlock|access)\b',
        r'\b(link in bio|dm me|dm us|comment below|tag a friend|share this|double tap|save this|bookmark)\b',
        r'\b(don\'t miss|limited time|act now|hurry|today only|last chance|while supplies last)\b'
    ]

    # Question patterns
    QUESTION_PATTERNS = [
        r'\?',
        r'\b(what|who|where|when|why|how|which|would|could|should|do you|have you|are you|did you)\b.*\?',
    ]

    # Engagement hooks
    HOOK_PATTERNS = [
        r'^(stop|wait|attention|breaking|exciting|finally|introducing|new|secret|revealed)',
        r'\b(you need|you must|you should|you have to|imagine|picture this|here\'s)\b',
        r'\b(did you know|fun fact|pro tip|hot take|unpopular opinion)\b'
    ]

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize FIITValidator.

        Args:
            config: Configuration object (optional)
        """
        self.config = config or Config()
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if not HAS_TEXTSTAT:
            print("Warning: textstat not available. Fluency scoring will be limited.")
        if not HAS_TEXTBLOB:
            print("Warning: textblob not available. Tone scoring will be limited.")

    def validate(
        self,
        content: str,
        insights: Optional[Dict[str, Any]] = None,
        target_tone: str = 'engaging'
    ) -> Dict[str, Any]:
        """
        Validate content using FIIT framework.

        Args:
            content: Content to validate
            insights: Optional insights for information scoring
            target_tone: Target tone for tone scoring

        Returns:
            Dictionary with FIIT scores and analysis
        """
        # Calculate individual scores
        fluency_score, fluency_details = self.validate_fluency(content)
        interactivity_score, interactivity_details = self.validate_interactivity(content)
        information_score, information_details = self.validate_information(content, insights)
        tone_score, tone_details = self.validate_tone(content, target_tone)

        # Calculate weighted overall score
        overall = (
            self.WEIGHTS['fluency'] * fluency_score +
            self.WEIGHTS['interactivity'] * interactivity_score +
            self.WEIGHTS['information'] * information_score +
            self.WEIGHTS['tone'] * tone_score
        )

        # Check if thresholds are met
        passed = {
            'fluency': fluency_score >= self.THRESHOLDS['fluency'],
            'interactivity': interactivity_score >= self.THRESHOLDS['interactivity'],
            'information': information_score >= self.THRESHOLDS['information'],
            'tone': tone_score >= self.THRESHOLDS['tone'],
            'overall': overall >= self.TARGET_OVERALL
        }

        all_passed = all(passed.values())

        return {
            'scores': {
                'fluency': float(fluency_score),
                'interactivity': float(interactivity_score),
                'information': float(information_score),
                'tone': float(tone_score),
                'overall': float(overall)
            },
            'details': {
                'fluency': fluency_details,
                'interactivity': interactivity_details,
                'information': information_details,
                'tone': tone_details
            },
            'passed': passed,
            'all_passed': all_passed,
            'target_overall': self.TARGET_OVERALL,
            'thresholds': self.THRESHOLDS,
            'improvements_needed': self._get_improvements_needed(
                fluency_score, interactivity_score,
                information_score, tone_score
            )
        }

    def validate_fluency(self, content: str) -> Tuple[float, Dict[str, Any]]:
        """
        Validate fluency using readability metrics.

        Uses Flesch Reading Ease score:
        - 90-100: Very Easy
        - 80-89: Easy
        - 70-79: Fairly Easy
        - 60-69: Standard (ideal for social media)
        - 50-59: Fairly Difficult
        - 30-49: Difficult
        - 0-29: Very Difficult

        Args:
            content: Text content to analyze

        Returns:
            Tuple of (score, details)
        """
        details = {
            'flesch_reading_ease': None,
            'flesch_grade': None,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'syllable_count': 0
        }

        if not content or len(content.strip()) == 0:
            return 0.0, details

        # Clean content (remove hashtags and mentions for readability)
        clean_content = re.sub(r'#\w+', '', content)
        clean_content = re.sub(r'@\w+', '', clean_content)
        clean_content = re.sub(r'https?://\S+', '', clean_content)
        clean_content = clean_content.strip()

        if not clean_content:
            return 0.5, details  # Return neutral score if only hashtags

        # Basic stats
        words = clean_content.split()
        details['word_count'] = len(words)

        sentences = re.split(r'[.!?]+', clean_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        details['sentence_count'] = max(len(sentences), 1)

        if words:
            details['avg_word_length'] = sum(len(w) for w in words) / len(words)

        if HAS_TEXTSTAT:
            try:
                details['flesch_reading_ease'] = textstat.flesch_reading_ease(clean_content)
                details['flesch_grade'] = textstat.flesch_kincaid_grade(clean_content)
                details['syllable_count'] = textstat.syllable_count(clean_content)

                # Convert Flesch Reading Ease to 0-1 score
                # Optimal for social media: 60-80
                fre = details['flesch_reading_ease']

                if fre >= 80:
                    score = 1.0
                elif fre >= 60:
                    score = 0.8 + (fre - 60) / 100  # 0.8-1.0
                elif fre >= 40:
                    score = 0.6 + (fre - 40) / 100  # 0.6-0.8
                elif fre >= 20:
                    score = 0.4 + (fre - 20) / 100  # 0.4-0.6
                else:
                    score = max(0.2, fre / 50)  # 0.2-0.4

                return min(1.0, max(0.0, score)), details

            except Exception as e:
                details['error'] = str(e)

        # Fallback scoring without textstat
        # Simple heuristic based on average word length and sentence length
        avg_word_len = details['avg_word_length']
        avg_sent_len = details['word_count'] / details['sentence_count']

        # Ideal: 4-6 char words, 10-20 words per sentence
        word_score = 1.0 - abs(avg_word_len - 5) / 5
        sent_score = 1.0 - abs(avg_sent_len - 15) / 20

        score = (word_score + sent_score) / 2
        score = min(1.0, max(0.0, score))

        return score, details

    def validate_interactivity(self, content: str) -> Tuple[float, Dict[str, Any]]:
        """
        Validate interactivity elements.

        Checks for:
        - Call-to-actions (CTAs)
        - Questions
        - Engagement hooks
        - Emojis
        - Hashtags

        Args:
            content: Text content to analyze

        Returns:
            Tuple of (score, details)
        """
        details = {
            'has_cta': False,
            'cta_count': 0,
            'cta_matches': [],
            'has_question': False,
            'question_count': 0,
            'has_hook': False,
            'hook_matches': [],
            'emoji_count': 0,
            'hashtag_count': 0
        }

        if not content:
            return 0.0, details

        content_lower = content.lower()

        # Check CTAs
        for pattern in self.CTA_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                details['has_cta'] = True
                details['cta_count'] += len(matches)
                details['cta_matches'].extend(matches[:3])  # Keep first 3

        # Check questions
        question_marks = content.count('?')
        details['question_count'] = question_marks
        details['has_question'] = question_marks > 0

        # Check hooks
        for pattern in self.HOOK_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                details['has_hook'] = True
                details['hook_matches'].extend(matches[:2])

        # Count emojis (simplified detection)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "]+",
            flags=re.UNICODE
        )
        emojis = emoji_pattern.findall(content)
        details['emoji_count'] = sum(len(e) for e in emojis)

        # Count hashtags
        hashtags = re.findall(r'#\w+', content)
        details['hashtag_count'] = len(hashtags)

        # Calculate score (weighted components)
        score = 0.0

        # CTA (40% weight)
        if details['has_cta']:
            score += 0.4
        elif details['question_count'] > 0:
            score += 0.2  # Questions as partial CTA

        # Questions (30% weight)
        if details['has_question']:
            score += 0.3

        # Hooks (15% weight)
        if details['has_hook']:
            score += 0.15

        # Emojis (10% weight - 2-4 is optimal)
        emoji_count = details['emoji_count']
        if 2 <= emoji_count <= 4:
            score += 0.10
        elif 1 <= emoji_count <= 6:
            score += 0.05

        # Hashtags (5% weight - present but not excessive)
        hashtag_count = details['hashtag_count']
        if 3 <= hashtag_count <= 15:
            score += 0.05
        elif 1 <= hashtag_count <= 20:
            score += 0.02

        return min(1.0, score), details

    def validate_information(
        self,
        content: str,
        insights: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Validate information value and relevance.

        Checks for:
        - Keyword relevance to insights
        - Information density
        - Value indicators

        Args:
            content: Text content to analyze
            insights: Optional insights for relevance scoring

        Returns:
            Tuple of (score, details)
        """
        details = {
            'keyword_matches': [],
            'relevance_score': 0.0,
            'has_numbers': False,
            'has_value_indicators': False,
            'content_density': 0.0
        }

        if not content:
            return 0.0, details

        content_lower = content.lower()

        # Check for numbers/statistics (adds credibility)
        numbers = re.findall(r'\d+%?', content)
        details['has_numbers'] = len(numbers) > 0

        # Value indicators
        value_patterns = [
            r'\b(tip|trick|hack|secret|guide|how to|steps|ways|reasons|benefits|learn|discover)\b',
            r'\b(save|free|bonus|exclusive|new|best|top|ultimate|complete)\b',
            r'\b(results|success|growth|improvement|increase|boost)\b'
        ]

        value_matches = []
        for pattern in value_patterns:
            matches = re.findall(pattern, content_lower)
            value_matches.extend(matches)

        details['has_value_indicators'] = len(value_matches) > 0
        details['value_indicators'] = value_matches[:5]

        # Calculate content density (meaningful words / total words)
        words = content_lower.split()
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                     'by', 'from', 'as', 'into', 'through', 'during', 'before',
                     'after', 'above', 'below', 'between', 'under', 'again',
                     'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'so',
                     'yet', 'both', 'either', 'neither', 'not', 'only', 'own',
                     'same', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
                     'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
                     'more', 'most', 'other', 'some', 'such', 'no', 'any', 'i',
                     'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                     'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                     'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                     'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                     'those', 'am'}

        meaningful_words = [w for w in words if w not in stopwords and len(w) > 2]
        details['content_density'] = len(meaningful_words) / max(len(words), 1)

        # Relevance to insights
        if insights:
            insight_keywords = self._extract_insight_keywords(insights)
            matches = [kw for kw in insight_keywords if kw.lower() in content_lower]
            details['keyword_matches'] = matches
            details['relevance_score'] = len(matches) / max(len(insight_keywords), 1)

        # Calculate overall score
        score = 0.0

        # Content density (30%)
        score += min(0.3, details['content_density'] * 0.5)

        # Value indicators (30%)
        if details['has_value_indicators']:
            score += 0.3

        # Numbers/statistics (20%)
        if details['has_numbers']:
            score += 0.2

        # Relevance to insights (20%)
        if insights:
            score += details['relevance_score'] * 0.2
        else:
            score += 0.1  # Base score without insights

        return min(1.0, score), details

    def _extract_insight_keywords(self, insights: Dict[str, Any]) -> List[str]:
        """Extract keywords from insights for relevance matching."""
        keywords = []

        # Trend keywords
        trend = insights.get('trend_analysis', {})
        if trend.get('direction'):
            keywords.append(trend['direction'])

        # Day keywords
        temporal = insights.get('temporal_patterns', {})
        best_days = temporal.get('best_days', [])
        keywords.extend([d.get('day', '') for d in best_days])

        # Add common engagement keywords
        keywords.extend(['engagement', 'post', 'content', 'followers', 'growth'])

        return [k for k in keywords if k]

    def validate_tone(
        self,
        content: str,
        target_tone: str = 'engaging'
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Validate tone and sentiment consistency.

        Args:
            content: Text content to analyze
            target_tone: Target tone ('engaging', 'professional', 'casual', 'inspiring')

        Returns:
            Tuple of (score, details)
        """
        details = {
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.0,
            'target_tone': target_tone,
            'tone_match': False,
            'detected_tone': 'neutral'
        }

        if not content:
            return 0.5, details

        # Clean content for sentiment analysis
        clean_content = re.sub(r'#\w+', '', content)
        clean_content = re.sub(r'@\w+', '', clean_content)
        clean_content = re.sub(r'https?://\S+', '', clean_content)

        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(clean_content)
                details['sentiment_polarity'] = blob.sentiment.polarity
                details['sentiment_subjectivity'] = blob.sentiment.subjectivity

                # Determine detected tone based on polarity
                polarity = blob.sentiment.polarity
                if polarity > 0.3:
                    details['detected_tone'] = 'positive'
                elif polarity > 0.1:
                    details['detected_tone'] = 'slightly_positive'
                elif polarity < -0.3:
                    details['detected_tone'] = 'negative'
                elif polarity < -0.1:
                    details['detected_tone'] = 'slightly_negative'
                else:
                    details['detected_tone'] = 'neutral'

            except Exception as e:
                details['error'] = str(e)

        # Target tone scoring
        polarity = details['sentiment_polarity']

        if target_tone == 'engaging':
            # Engaging content should be positive (0.1 to 0.5)
            if 0.1 <= polarity <= 0.5:
                score = 1.0
                details['tone_match'] = True
            elif 0.0 <= polarity <= 0.6:
                score = 0.7
            elif polarity > 0:
                score = 0.5
            else:
                score = 0.3

        elif target_tone == 'professional':
            # Professional should be neutral to slightly positive (0.0 to 0.3)
            if 0.0 <= polarity <= 0.3:
                score = 1.0
                details['tone_match'] = True
            elif -0.1 <= polarity <= 0.4:
                score = 0.7
            else:
                score = 0.4

        elif target_tone == 'casual':
            # Casual can be more varied (-0.1 to 0.5)
            if -0.1 <= polarity <= 0.5:
                score = 0.9
                details['tone_match'] = True
            else:
                score = 0.6

        elif target_tone == 'inspiring':
            # Inspiring should be positive (0.2 to 0.7)
            if 0.2 <= polarity <= 0.7:
                score = 1.0
                details['tone_match'] = True
            elif polarity > 0.1:
                score = 0.7
            else:
                score = 0.4

        else:
            # Default: positive is good
            score = 0.5 + polarity / 2

        return min(1.0, max(0.0, score)), details

    def _get_improvements_needed(
        self,
        fluency: float,
        interactivity: float,
        information: float,
        tone: float
    ) -> List[str]:
        """Get list of improvements needed based on scores."""
        improvements = []

        if fluency < self.THRESHOLDS['fluency']:
            improvements.append('fluency: simplify language, use shorter sentences')

        if interactivity < self.THRESHOLDS['interactivity']:
            improvements.append('interactivity: add CTA, question, or engagement hook')

        if information < self.THRESHOLDS['information']:
            improvements.append('information: add value-driven content or statistics')

        if tone < self.THRESHOLDS['tone']:
            improvements.append('tone: adjust sentiment to be more engaging/positive')

        return improvements

    def get_score_report(self, validation_result: Dict[str, Any]) -> str:
        """
        Generate a formatted score report.

        Args:
            validation_result: Result from validate()

        Returns:
            Formatted report string
        """
        scores = validation_result['scores']
        passed = validation_result['passed']
        improvements = validation_result.get('improvements_needed', [])

        lines = [
            "=" * 50,
            "FIIT CONTENT VALIDATION REPORT",
            "=" * 50,
            "",
            "SCORES:",
            f"  Fluency:        {scores['fluency']:.2f} {'[PASS]' if passed['fluency'] else '[FAIL]'}",
            f"  Interactivity:  {scores['interactivity']:.2f} {'[PASS]' if passed['interactivity'] else '[FAIL]'}",
            f"  Information:    {scores['information']:.2f} {'[PASS]' if passed['information'] else '[FAIL]'}",
            f"  Tone:           {scores['tone']:.2f} {'[PASS]' if passed['tone'] else '[FAIL]'}",
            "",
            f"  OVERALL:        {scores['overall']:.2f} {'[PASS]' if passed['overall'] else '[FAIL]'}",
            f"  Target:         {self.TARGET_OVERALL}",
            "",
        ]

        if improvements:
            lines.append("IMPROVEMENTS NEEDED:")
            for imp in improvements:
                lines.append(f"  - {imp}")
        else:
            lines.append("STATUS: All thresholds passed!")

        lines.append("")
        lines.append("=" * 50)

        return '\n'.join(lines)


def quick_validate(content: str) -> Dict[str, Any]:
    """
    Quick validation helper function.

    Args:
        content: Content to validate

    Returns:
        Validation result
    """
    validator = FIITValidator()
    return validator.validate(content)
