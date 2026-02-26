"""
HuggingFace LLM Client for SocialProphet.

Provides interface to HuggingFace Inference API for content generation
using available models via chat_completion.
"""

import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta

from huggingface_hub import InferenceClient

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = datetime.now()
        self.requests = [
            r for r in self.requests
            if now - r < timedelta(seconds=self.window_seconds)
        ]

        if len(self.requests) >= self.max_requests:
            oldest = min(self.requests)
            wait_time = (oldest + timedelta(seconds=self.window_seconds) - now).total_seconds()
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time + 0.1)

        self.requests.append(now)


class HuggingFaceClient:
    """
    Client for HuggingFace Inference API.

    Uses huggingface_hub InferenceClient with chat_completion.
    """

    # Model options (currently available on free tier)
    MODELS = {
        'llama': 'meta-llama/Llama-3.2-1B-Instruct',
        'llama3b': 'meta-llama/Llama-3.2-3B-Instruct',
        'qwen': 'Qwen/Qwen2.5-Coder-32B-Instruct',
        'phi': 'microsoft/Phi-3.5-mini-instruct',
    }

    def __init__(
        self,
        token: Optional[str] = None,
        model: str = 'llama',
        max_retries: int = 3,
        timeout: int = 120
    ):
        """
        Initialize HuggingFace client.

        Args:
            token: HuggingFace API token (uses HF_TOKEN env var if not provided)
            model: Model to use ('llama', 'llama3b', 'qwen', 'phi')
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.token = token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        if not self.token:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN environment variable "
                "or pass token parameter."
            )

        self.model_key = model
        self.model_name = self.MODELS.get(model, self.MODELS['llama'])
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limiter = RateLimiter(max_requests=30, window_seconds=60)

        # Initialize InferenceClient
        self.client = InferenceClient(token=self.token)

        # Cache for responses
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour

        print(f"HuggingFace client initialized with model: {self.model_name}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_cache: bool = True
    ) -> str:
        """
        Generate text from prompt using chat_completion.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            use_cache: Whether to use response cache

        Returns:
            Generated text
        """
        # Check cache
        cache_key = f"{prompt[:100]}_{max_new_tokens}_{temperature}"
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached['time'] < self._cache_ttl:
                return cached['response']

        # Build messages for chat completion
        messages = [
            {"role": "user", "content": prompt}
        ]

        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait_if_needed()

                response = self.client.chat_completion(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

                generated = response.choices[0].message.content

                # Cache response
                if use_cache:
                    self._cache[cache_key] = {
                        'response': generated,
                        'time': time.time()
                    }

                return generated

            except Exception as e:
                print(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)[:80]}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

        # Try fallback models
        return self._fallback_generate(prompt, max_new_tokens, temperature)

    def _fallback_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float
    ) -> str:
        """
        Try fallback models if primary fails.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Generated text or error message
        """
        fallback_models = ['llama3b', 'qwen', 'phi']

        for model_key in fallback_models:
            if model_key == self.model_key:
                continue

            model_name = self.MODELS.get(model_key)
            if not model_name:
                continue

            print(f"Trying fallback model: {model_name}")

            try:
                messages = [{"role": "user", "content": prompt}]

                response = self.client.chat_completion(
                    messages=messages,
                    model=model_name,
                    max_tokens=min(max_new_tokens, 300),
                    temperature=temperature,
                )

                return response.choices[0].message.content

            except Exception as e:
                print(f"Fallback {model_key} failed: {str(e)[:60]}")
                continue

        return "[Error: All models failed to generate response. Please try again later.]"

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Generating {i+1}/{len(prompts)}...")
            result = self.generate(prompt, **kwargs)
            results.append(result)
            time.sleep(0.5)
        return results

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response for chat-style messages.

        Args:
            messages: List of {'role': 'user'|'assistant', 'content': '...'}
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        chat_messages = []

        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})

        chat_messages.extend(messages)

        max_tokens = kwargs.get('max_new_tokens', 500)
        temperature = kwargs.get('temperature', 0.7)

        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait_if_needed()

                response = self.client.chat_completion(
                    messages=chat_messages,
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                return response.choices[0].message.content

            except Exception as e:
                print(f"Chat attempt {attempt + 1} failed: {str(e)[:60]}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        return "[Error: Failed to generate chat response]"

    def check_model_status(self) -> Dict[str, Any]:
        """
        Check if the model is loaded and available.

        Returns:
            Status dictionary
        """
        try:
            # Try a simple generation
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model=self.model_name,
                max_tokens=5,
            )

            return {
                'available': True,
                'model': self.model_name,
                'test_response': response.choices[0].message.content
            }
        except Exception as e:
            return {
                'available': False,
                'model': self.model_name,
                'error': str(e)
            }

    def switch_model(self, model: str):
        """
        Switch to a different model.

        Args:
            model: Model key ('llama', 'llama3b', 'qwen', 'phi')
        """
        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}. Choose from: {list(self.MODELS.keys())}")

        self.model_key = model
        self.model_name = self.MODELS[model]
        print(f"Switched to model: {self.model_name}")

    def clear_cache(self):
        """Clear the response cache."""
        self._cache = {}
        print("Cache cleared")

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Usage statistics dictionary
        """
        return {
            'model': self.model_name,
            'cache_size': len(self._cache),
            'recent_requests': len(self.rate_limiter.requests),
            'rate_limit': f"{self.rate_limiter.max_requests}/{self.rate_limiter.window_seconds}s"
        }


def test_client():
    """Test the HuggingFace client."""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        client = HuggingFaceClient()

        print("\nChecking model status...")
        status = client.check_model_status()
        print(f"Status: {status}")

        print("\nGenerating test response...")
        response = client.generate(
            "Write a short, engaging Instagram caption about a morning coffee:",
            max_new_tokens=100,
            temperature=0.7
        )
        print(f"Response: {response}")

        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    test_client()
