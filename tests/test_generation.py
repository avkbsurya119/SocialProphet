"""
Unit tests for SocialProphet Generation Module.

Tests:
- InsightExtractor
- PromptBuilder
- HuggingFaceClient (mocked)
- ContentGenerator
- FIITValidator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insights.extractor import InsightExtractor
from src.insights.prompt_builder import PromptBuilder
from src.generation.llm_client import HuggingFaceClient, RateLimiter
from src.generation.content_gen import ContentGenerator
from src.generation.fiit_validator import FIITValidator


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_historical_df():
    """Create sample historical data."""
    dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'ds': dates,
        'y': np.random.uniform(9.5, 11.0, 90),
        'y_raw': np.random.uniform(15000, 50000, 90),
        'day_of_week': [d.dayofweek for d in dates],
        'is_weekend': [1 if d.dayofweek >= 5 else 0 for d in dates],
        'month': [d.month for d in dates],
        'y_pct_change_1': np.random.uniform(-0.1, 0.1, 90)
    })

    return df


@pytest.fixture
def sample_predictions_df():
    """Create sample predictions data."""
    dates = pd.date_range(start='2024-04-01', periods=30, freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'ds': dates,
        'ensemble_pred': np.random.uniform(9.5, 11.0, 30),
        'ensemble_pred_original': np.random.uniform(15000, 50000, 30)
    })

    return df


@pytest.fixture
def sample_insights():
    """Create sample insights dictionary."""
    return {
        'trend_analysis': {
            'direction': 'increasing',
            'strength': 'moderate',
            'momentum_7d': 5.2,
            'historical_mean': 28000
        },
        'temporal_patterns': {
            'best_days': [
                {'day': 'Tuesday', 'day_of_week': 1, 'avg_engagement': 32000},
                {'day': 'Thursday', 'day_of_week': 3, 'avg_engagement': 30000}
            ],
            'weekend_vs_weekday': {
                'better': 'weekday',
                'difference_pct': 12.5
            }
        },
        'predictions_summary': {
            'mean_predicted': 29000,
            'min_predicted': 20000,
            'max_predicted': 40000
        },
        'seasonality': {
            'weekly_pattern': {'Monday': 0.8, 'Tuesday': 1.0, 'Wednesday': 0.7}
        }
    }


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock(spec=HuggingFaceClient)
    client.generate.return_value = """
    CAPTION: Check out our latest tips for boosting engagement!

    HASHTAGS: #socialmedia #marketing #tips #engagement #growth

    BEST TIME TO POST: Tuesday 10:00 AM

    CONTENT TYPE: carousel
    """
    return client


@pytest.fixture
def prompt_builder():
    """Create PromptBuilder instance."""
    return PromptBuilder(platform='instagram', brand_voice='friendly and engaging')


# =============================================================================
# InsightExtractor Tests
# =============================================================================

class TestInsightExtractor:
    """Tests for InsightExtractor class."""

    def test_init(self):
        """Test InsightExtractor initialization."""
        extractor = InsightExtractor()
        assert extractor.insights == {}

    def test_extract_temporal_patterns(self, sample_historical_df):
        """Test temporal pattern extraction."""
        extractor = InsightExtractor()
        patterns = extractor.extract_temporal_patterns(sample_historical_df)

        assert 'best_days' in patterns
        assert 'worst_days' in patterns
        assert 'weekend_vs_weekday' in patterns
        assert len(patterns['best_days']) <= 3

    def test_extract_trend(self, sample_predictions_df, sample_historical_df):
        """Test trend extraction."""
        extractor = InsightExtractor()
        trend = extractor.extract_trend(sample_predictions_df, sample_historical_df)

        assert 'direction' in trend
        assert trend['direction'] in ['increasing', 'decreasing', 'stable']
        assert 'momentum_7d' in trend
        assert 'historical_mean' in trend

    def test_extract_content_patterns(self, sample_historical_df):
        """Test content pattern extraction."""
        extractor = InsightExtractor()
        patterns = extractor.extract_content_patterns(sample_historical_df)

        assert 'engagement_percentiles' in patterns
        assert 'high_performance_threshold' in patterns
        assert patterns['total_observations'] == len(sample_historical_df)

    def test_extract_seasonality(self, sample_historical_df):
        """Test seasonality extraction."""
        extractor = InsightExtractor()
        seasonality = extractor.extract_seasonality(sample_historical_df)

        assert 'weekly_pattern' in seasonality
        assert 'has_strong_weekly_pattern' in seasonality

    def test_extract_all(self, sample_predictions_df, sample_historical_df):
        """Test full insight extraction."""
        extractor = InsightExtractor()
        insights = extractor.extract_all(sample_predictions_df, sample_historical_df)

        assert 'temporal_patterns' in insights
        assert 'trend_analysis' in insights
        assert 'content_patterns' in insights
        assert 'seasonality' in insights
        assert 'recommendations' in insights
        assert 'metadata' in insights

    def test_to_prompt_context(self, sample_predictions_df, sample_historical_df):
        """Test prompt context generation."""
        extractor = InsightExtractor()
        extractor.extract_all(sample_predictions_df, sample_historical_df)
        context = extractor.to_prompt_context()

        assert isinstance(context, str)
        assert 'TREND' in context

    def test_extract_peaks(self, sample_predictions_df):
        """Test peak extraction."""
        extractor = InsightExtractor()
        peaks = extractor.extract_peaks(sample_predictions_df, n_peaks=3)

        assert len(peaks) <= 3
        if peaks:
            assert 'predicted_engagement' in peaks[0]
            assert 'rank' in peaks[0]


# =============================================================================
# PromptBuilder Tests
# =============================================================================

class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    def test_init(self):
        """Test PromptBuilder initialization."""
        builder = PromptBuilder(platform='twitter')
        assert builder.platform == 'twitter'
        assert builder.config['max_caption_length'] == 280

    def test_get_system_prompt(self, prompt_builder):
        """Test system prompt generation."""
        prompt = prompt_builder.get_system_prompt()

        assert 'instagram' in prompt.lower()
        assert 'FIIT' in prompt
        assert 'friendly and engaging' in prompt

    def test_build_post_prompt(self, prompt_builder, sample_insights):
        """Test post prompt building."""
        prompt = prompt_builder.build_post_prompt(sample_insights, theme='educational')

        assert 'ENGAGEMENT INSIGHTS' in prompt
        assert 'educational' in prompt.lower()
        assert 'CAPTION' in prompt

    def test_build_campaign_prompt(self, prompt_builder, sample_insights):
        """Test campaign prompt building."""
        prompt = prompt_builder.build_campaign_prompt(
            sample_insights, n_posts=5, campaign_goal='engagement'
        )

        assert '5' in prompt
        assert 'engagement' in prompt.lower()
        assert 'POST 1' in prompt

    def test_build_hashtag_prompt(self, prompt_builder, sample_insights):
        """Test hashtag prompt building."""
        prompt = prompt_builder.build_hashtag_prompt(
            "Test content", sample_insights, count=10
        )

        assert '10' in prompt
        assert 'hashtag' in prompt.lower()

    def test_build_schedule_prompt(self, prompt_builder, sample_insights):
        """Test schedule prompt building."""
        prompt = prompt_builder.build_schedule_prompt(sample_insights, n_days=7)

        assert '7' in prompt
        assert 'DAY 1' in prompt

    def test_build_variation_prompt(self, prompt_builder):
        """Test variation prompt building."""
        prompt = prompt_builder.build_variation_prompt("Original content", n_variations=3)

        assert '3' in prompt
        assert 'A/B' in prompt or 'variation' in prompt.lower()

    def test_set_platform(self, prompt_builder):
        """Test platform switching."""
        prompt_builder.set_platform('linkedin')
        assert prompt_builder.platform == 'linkedin'
        assert prompt_builder.config['tone'] == 'professional, insightful, thought-leadership'

    def test_format_insights_context(self, prompt_builder, sample_insights):
        """Test insights context formatting."""
        context = prompt_builder.format_insights_context(sample_insights)

        assert 'TREND ANALYSIS' in context
        assert 'BEST POSTING DAYS' in context


# =============================================================================
# HuggingFaceClient Tests (Mocked)
# =============================================================================

class TestHuggingFaceClient:
    """Tests for HuggingFaceClient class (mocked)."""

    def test_rate_limiter(self):
        """Test rate limiter functionality."""
        limiter = RateLimiter(max_requests=5, window_seconds=1)

        # Should not block for first 5 requests
        for _ in range(5):
            limiter.wait_if_needed()

        assert len(limiter.requests) == 5

    @patch.dict('os.environ', {'HF_TOKEN': 'test_token'})
    def test_init_with_env_token(self):
        """Test initialization with environment token."""
        client = HuggingFaceClient()
        assert client.token == 'test_token'

    def test_init_without_token_raises(self):
        """Test initialization without token raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="HuggingFace token required"):
                HuggingFaceClient()

    @patch.dict('os.environ', {'HF_TOKEN': 'test_token'})
    def test_switch_model(self):
        """Test model switching."""
        client = HuggingFaceClient()
        client.switch_model('qwen')
        assert client.model_key == 'qwen'
        assert 'qwen' in client.model_name.lower()

    @patch.dict('os.environ', {'HF_TOKEN': 'test_token'})
    def test_get_usage_stats(self):
        """Test usage statistics."""
        client = HuggingFaceClient()
        stats = client.get_usage_stats()

        assert 'model' in stats
        assert 'cache_size' in stats
        assert 'rate_limit' in stats


# =============================================================================
# ContentGenerator Tests
# =============================================================================

class TestContentGenerator:
    """Tests for ContentGenerator class."""

    def test_init(self, mock_llm_client, prompt_builder):
        """Test ContentGenerator initialization."""
        generator = ContentGenerator(mock_llm_client, prompt_builder)
        assert generator.generated_content == []

    def test_generate_post(self, mock_llm_client, prompt_builder, sample_insights):
        """Test post generation."""
        generator = ContentGenerator(mock_llm_client, prompt_builder)
        result = generator.generate_post(sample_insights, theme='educational')

        assert 'caption' in result
        assert 'hashtags' in result
        assert 'platform' in result
        assert result['theme'] == 'educational'

    def test_generate_variations(self, mock_llm_client, prompt_builder):
        """Test variation generation."""
        mock_llm_client.generate.return_value = """
        VARIATION 1:
        Caption: First variation here
        Hashtags: #test1 #test2
        Changes Made: Changed the hook

        VARIATION 2:
        Caption: Second variation here
        Hashtags: #test3 #test4
        Changes Made: Different CTA
        """

        generator = ContentGenerator(mock_llm_client, prompt_builder)
        result = generator.generate_variations("Original content", n_variations=2)

        assert 'original' in result
        assert 'variations' in result

    def test_generate_hashtags(self, mock_llm_client, prompt_builder, sample_insights):
        """Test hashtag generation."""
        mock_llm_client.generate.return_value = "#social #media #marketing #growth #tips"

        generator = ContentGenerator(mock_llm_client, prompt_builder)
        hashtags = generator.generate_hashtags("Test content", sample_insights, count=5)

        assert isinstance(hashtags, list)
        assert all(tag.startswith('#') for tag in hashtags)

    def test_extract_hashtags(self, mock_llm_client, prompt_builder):
        """Test hashtag extraction helper."""
        generator = ContentGenerator(mock_llm_client, prompt_builder)

        text = "Check this out #social #Media #SOCIAL #growth"
        hashtags = generator._extract_hashtags(text)

        assert len(hashtags) == 3  # Deduplication (case-insensitive)
        assert '#social' in hashtags or '#Social' in hashtags

    def test_get_generated_content(self, mock_llm_client, prompt_builder, sample_insights):
        """Test retrieving generated content."""
        generator = ContentGenerator(mock_llm_client, prompt_builder)
        generator.generate_post(sample_insights)

        content = generator.get_generated_content()
        assert len(content) == 1

    def test_clear_history(self, mock_llm_client, prompt_builder, sample_insights):
        """Test clearing generated content history."""
        generator = ContentGenerator(mock_llm_client, prompt_builder)
        generator.generate_post(sample_insights)
        generator.clear_history()

        assert generator.generated_content == []


# =============================================================================
# FIITValidator Tests
# =============================================================================

class TestFIITValidator:
    """Tests for FIITValidator class."""

    def test_init(self):
        """Test FIITValidator initialization."""
        validator = FIITValidator()
        assert validator.TARGET_OVERALL == 0.85

    def test_validate_fluency_readable(self):
        """Test fluency validation with readable content."""
        validator = FIITValidator()
        content = "Check out our new product. It's amazing and simple to use."
        score, details = validator.validate_fluency(content)

        assert 0 <= score <= 1
        assert 'word_count' in details
        assert details['word_count'] > 0

    def test_validate_fluency_empty(self):
        """Test fluency validation with empty content."""
        validator = FIITValidator()
        score, details = validator.validate_fluency("")

        assert score == 0
        assert details['word_count'] == 0

    def test_validate_interactivity_with_cta(self):
        """Test interactivity with CTA."""
        validator = FIITValidator()
        content = "Click the link in bio to learn more! What do you think?"
        score, details = validator.validate_interactivity(content)

        assert score > 0.5
        assert details['has_cta'] == True
        assert details['has_question'] == True

    def test_validate_interactivity_no_elements(self):
        """Test interactivity without interactive elements."""
        validator = FIITValidator()
        content = "This is a simple statement about our product."
        score, details = validator.validate_interactivity(content)

        assert details['has_cta'] == False
        assert details['has_question'] == False

    def test_validate_information_with_value(self):
        """Test information validation with value indicators."""
        validator = FIITValidator()
        content = "Here are 5 tips to boost your engagement by 50%. Learn how to improve."
        score, details = validator.validate_information(content)

        assert score > 0.3
        assert details['has_numbers'] == True
        assert details['has_value_indicators'] == True

    def test_validate_tone_positive(self):
        """Test tone validation with positive content."""
        validator = FIITValidator()
        content = "We're excited to share this amazing news with you! Great things ahead!"
        score, details = validator.validate_tone(content, target_tone='engaging')

        assert score >= 0.5
        assert details['target_tone'] == 'engaging'

    def test_validate_full(self, sample_insights):
        """Test full validation."""
        validator = FIITValidator()
        content = "Check out these 5 tips! What's your favorite? Click link in bio!"
        result = validator.validate(content, sample_insights)

        assert 'scores' in result
        assert 'details' in result
        assert 'passed' in result
        assert 'all_passed' in result
        assert 0 <= result['scores']['overall'] <= 1

    def test_validate_returns_improvements(self):
        """Test that validation returns improvement suggestions."""
        validator = FIITValidator()
        content = "Simple text."  # Poor content
        result = validator.validate(content)

        # Should have some improvements needed
        assert 'improvements_needed' in result

    def test_get_score_report(self, sample_insights):
        """Test score report generation."""
        validator = FIITValidator()
        content = "Great tips here! Click to learn more. #tips #growth"
        result = validator.validate(content, sample_insights)
        report = validator.get_score_report(result)

        assert 'FIIT' in report
        assert 'Fluency' in report
        assert 'Interactivity' in report
        assert 'OVERALL' in report

    def test_quick_validate(self):
        """Test quick validation helper."""
        from src.generation.fiit_validator import quick_validate

        result = quick_validate("Test content here!")
        assert 'scores' in result
        assert 'overall' in result['scores']


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the generation pipeline."""

    def test_insight_to_prompt_pipeline(self, sample_predictions_df, sample_historical_df):
        """Test insight extraction to prompt building pipeline."""
        # Extract insights
        extractor = InsightExtractor()
        insights = extractor.extract_all(sample_predictions_df, sample_historical_df)

        # Build prompt
        builder = PromptBuilder(platform='instagram')
        prompt = builder.build_post_prompt(insights)

        assert 'ENGAGEMENT INSIGHTS' in prompt
        assert 'CAPTION' in prompt

    def test_full_pipeline_mocked(
        self,
        mock_llm_client,
        sample_predictions_df,
        sample_historical_df
    ):
        """Test full generation pipeline with mocked LLM."""
        # Extract insights
        extractor = InsightExtractor()
        insights = extractor.extract_all(sample_predictions_df, sample_historical_df)

        # Build generator
        builder = PromptBuilder(platform='instagram')
        generator = ContentGenerator(mock_llm_client, builder)

        # Generate content
        result = generator.generate_post(insights)

        # Validate content
        validator = FIITValidator()
        validation = validator.validate(result.get('caption', ''), insights)

        assert 'scores' in validation
        assert result['platform'] == 'instagram'


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
