"""
HuggingFace LLM Client for SocialProphet.

Provides interface to HuggingFace Inference API for content generation
using Llama 3.1, Mistral, or GPT-2 fallback.
"""

import os
import time
import json
import requests
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from functools import wraps
from datetime import datetime, timedelta

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
        # Remove old requests outside the window
        self.requests = [
            r for r in self.requests
            if now - r < timedelta(seconds=self.window_seconds)
        ]

        if len(self.requests) >= self.max_requests:
            # Wait until oldest request expires
            oldest = min(self.requests)
            wait_time = (oldest + timedelta(seconds=self.window_seconds) - now).total_seconds()
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time + 0.1)

        self.requests.append(now)


class HuggingFaceClient:
    """
    Client for HuggingFace Inference API.

    Supports:
    - Llama 3.1 8B Instruct (primary)
    - Mistral 7B Instruct (backup)
    - GPT-2 (fallback)
    """

    # Model options (in order of preference)
    MODELS = {
        'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
        'gpt2': 'gpt2-large',
    }

    API_URL = "https://api-inference.huggingface.co/models/{model}"

    def __init__(
        self,
        token: Optional[str] = None,
        model: str = 'llama',
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        Initialize HuggingFace client.

        Args:
            token: HuggingFace API token (uses HF_TOKEN env var if not provided)
            model: Model to use ('llama', 'mistral', 'gpt2')
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

        # Cache for responses
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour

        # Headers
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        print(f"HuggingFace client initialized with model: {self.model_name}")

    def _get_api_url(self, model_name: Optional[str] = None) -> str:
        """Get API URL for model."""
        model = model_name or self.model_name
        return self.API_URL.format(model=model)

    def _make_request(
        self,
        payload: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make request to HuggingFace API with retry logic.

        Args:
            payload: Request payload
            model_name: Optional model override

        Returns:
            API response
        """
        url = self._get_api_url(model_name)

        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait_if_needed()

                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                # Handle specific error codes
                if response.status_code == 503:
                    # Model is loading
                    data = response.json()
                    wait_time = data.get('estimated_time', 20)
                    print(f"Model loading... waiting {wait_time}s")
                    time.sleep(min(wait_time, 60))
                    continue

                if response.status_code == 429:
                    # Rate limited
                    wait_time = int(response.headers.get('Retry-After', 60))
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                print(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

        raise RuntimeError(f"Failed after {self.max_retries} attempts")

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
        Generate text from prompt.

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

        # Build payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "return_full_text": False
            }
        }

        try:
            response = self._make_request(payload)

            # Parse response
            if isinstance(response, list) and len(response) > 0:
                generated = response[0].get('generated_text', '')
            elif isinstance(response, dict):
                generated = response.get('generated_text', str(response))
            else:
                generated = str(response)

            # Cache response
            if use_cache:
                self._cache[cache_key] = {
                    'response': generated,
                    'time': time.time()
                }

            return generated

        except Exception as e:
            print(f"Generation error with {self.model_name}: {e}")
            # Try fallback model
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
        fallback_order = ['mistral', 'gpt2']

        for model_key in fallback_order:
            if model_key == self.model_key:
                continue

            model_name = self.MODELS[model_key]
            print(f"Trying fallback model: {model_name}")

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": min(max_new_tokens, 250),  # Reduce for fallback
                    "temperature": temperature,
                    "do_sample": True
                }
            }

            try:
                response = self._make_request(payload, model_name)

                if isinstance(response, list) and len(response) > 0:
                    return response[0].get('generated_text', '')
                elif isinstance(response, dict):
                    return response.get('generated_text', '')

            except Exception as e:
                print(f"Fallback {model_key} failed: {e}")
                continue

        return "[Error: All models failed to generate response]"

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
            # Small delay between requests
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
        # Format as chat prompt for Llama/Mistral
        formatted_prompt = ""

        if system_prompt:
            formatted_prompt += f"<|system|>\n{system_prompt}\n"

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'user':
                formatted_prompt += f"<|user|>\n{content}\n"
            elif role == 'assistant':
                formatted_prompt += f"<|assistant|>\n{content}\n"

        formatted_prompt += "<|assistant|>\n"

        return self.generate(formatted_prompt, **kwargs)

    def check_model_status(self) -> Dict[str, Any]:
        """
        Check if the model is loaded and available.

        Returns:
            Status dictionary
        """
        url = self._get_api_url()

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            data = response.json()

            return {
                'available': response.status_code == 200,
                'model': self.model_name,
                'status_code': response.status_code,
                'details': data
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
            model: Model key ('llama', 'mistral', 'gpt2')
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
