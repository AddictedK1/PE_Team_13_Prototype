"""LLM Provider abstraction layer.

Supports Gemini, OpenAI (and OpenAI-compatible APIs), and Mock providers.
Default temperature is 0.0 for consistent counterfactual testing.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional
from dotenv import load_dotenv

# Load local environment if present
load_dotenv()


class LLMAPIError(Exception):
    """Raised when an external LLM API call fails (rate limit, auth, network, timeout)."""

    def __init__(self, message: str, status_code: Optional[int] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.original_error = original_error


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text completion from prompt with specified temperature."""
        raise NotImplementedError


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini LLM client via google-genai SDK."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise LLMAPIError("Gemini API key not found. Please set GEMINI_API_KEY or LLM_API_KEY.")
        self.model = model or os.getenv("LLM_MODEL") or "gemini-2.5-flash"
        
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise LLMAPIError(f"Failed to initialize Gemini client: {e}", original_error=e)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        try:
            from google.genai import types

            config = types.GenerateContentConfig(temperature=temperature)
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            if not response or not response.text:
                raise LLMAPIError("Gemini returned an empty response.")
            return response.text
        except LLMAPIError:
            raise
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg.lower():
                raise LLMAPIError(f"Gemini API rate limit or quota exceeded: {err_msg}", status_code=429, original_error=e)
            if "401" in err_msg or "403" in err_msg or "API_KEY_INVALID" in err_msg:
                raise LLMAPIError(f"Gemini authentication failed: {err_msg}", status_code=401, original_error=e)
            if "timeout" in err_msg.lower() or "deadline" in err_msg.lower():
                raise LLMAPIError(f"Gemini request timed out: {err_msg}", status_code=408, original_error=e)
            raise LLMAPIError(f"Gemini API error: {err_msg}", original_error=e)


class OpenAILLMClient(BaseLLMClient):
    """OpenAI and OpenAI-compatible (Groq, Ollama, vLLM) LLM client."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise LLMAPIError("OpenAI API key not found. Please set OPENAI_API_KEY or LLM_API_KEY.")
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            raise LLMAPIError(f"Failed to initialize OpenAI client: {e}", original_error=e)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMAPIError("OpenAI returned an empty response.")
            return content
        except LLMAPIError:
            raise
        except Exception as e:
            err_msg = str(e)
            if "rate limit" in err_msg.lower() or "429" in err_msg:
                raise LLMAPIError(f"OpenAI rate limit exceeded: {err_msg}", status_code=429, original_error=e)
            if "auth" in err_msg.lower() or "401" in err_msg:
                raise LLMAPIError(f"OpenAI authentication failed: {err_msg}", status_code=401, original_error=e)
            if "timeout" in err_msg.lower():
                raise LLMAPIError(f"OpenAI request timed out: {err_msg}", status_code=408, original_error=e)
            raise LLMAPIError(f"OpenAI API error: {err_msg}", original_error=e)


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for offline unit testing and test suites.

    Can be primed with a sequence of responses (e.g. invalid then valid on retry).
    """

    def __init__(self, responses: Optional[List[str]] = None, raise_error: Optional[Exception] = None):
        self.responses = list(responses) if responses else []
        self.raise_error = raise_error
        self.call_count = 0
        self.recorded_prompts: List[str] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.recorded_prompts.append(prompt)
        self.call_count += 1

        if self.raise_error:
            raise self.raise_error

        if self.responses:
            return self.responses.pop(0)

        # Default simulated screening response if none queued
        return (
            '{\n'
            '  "decision": "shortlist",\n'
            '  "score": 82,\n'
            '  "reason": "Candidate demonstrates required software engineering skills, data structures, and project experience."\n'
            '}'
        )


def get_llm_client(provider: Optional[str] = None, **kwargs) -> BaseLLMClient:
    """Factory to obtain configured LLM client instance.

    Priority:
    1. Explicit provider argument
    2. LLM_PROVIDER env variable ('gemini', 'openai', 'mock')
    3. Auto-detect based on available keys: GEMINI_API_KEY -> Gemini, OPENAI_API_KEY -> OpenAI
    4. Fallback to MockLLMClient if no provider configured
    """
    prov = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()

    if prov == "mock":
        return MockLLMClient(**kwargs)

    if prov == "gemini":
        return GeminiLLMClient(**kwargs)

    if prov == "openai":
        return OpenAILLMClient(**kwargs)

    # Auto-detection
    if os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY"):
        return GeminiLLMClient(**kwargs)

    if os.getenv("OPENAI_API_KEY"):
        return OpenAILLMClient(**kwargs)

    # Offline fallback
    return MockLLMClient(**kwargs)
