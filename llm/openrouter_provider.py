"""OpenRouter provider implementation."""

from __future__ import annotations

import os
import random
import time
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .base_provider import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter API provider implementation.

    OpenRouter provides a unified API for accessing multiple LLM providers
    (OpenAI, Anthropic, Meta, Google, etc.) through an OpenAI-compatible interface.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model_name: str,
        timeout: int = 120,
        retries: int = 3,
        backoff_min: float = 2.0,
        backoff_max: float = 8.0,
        **kwargs,
    ):
        """Initialize OpenRouter provider."""
        self.config = config
        self.model_name = model_name
        self.timeout = timeout
        self.retries = retries
        self.backoff_min = backoff_min
        self.backoff_max = backoff_max

        # Verbose logging toggle (suppress request logs by default)
        logging_cfg = config.get("logging", {}) if isinstance(config, dict) else {}
        env_verbose = os.environ.get("HOUND_LLM_VERBOSE", "").lower() in {"1", "true", "yes", "on"}
        self.verbose = kwargs.get(
            "verbose", bool(logging_cfg.get("llm_verbose", False) or env_verbose)
        )
        self._last_token_usage = None

        # Get API key from environment
        openrouter_cfg = config.get("openrouter", {}) if isinstance(config, dict) else {}
        api_key_env = openrouter_cfg.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")

        # Get base URL from config or use default OpenRouter endpoint
        base_url = openrouter_cfg.get("base_url", "https://openrouter.ai/api/v1")

        # Optional OpenRouter-specific headers
        self.referer = openrouter_cfg.get("referer", "")
        self.app_title = openrouter_cfg.get("app_title", "Hound")

        # Build default headers for OpenRouter
        default_headers = {}
        if self.referer:
            default_headers["HTTP-Referer"] = self.referer
        if self.app_title:
            default_headers["X-Title"] = self.app_title

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers if default_headers else None,
        )

    def parse(self, *, system: str, user: str, schema: type[T]) -> T:
        """Make a structured call using OpenRouter's API.

        Uses JSON mode with schema instructions since OpenRouter proxies to
        multiple providers with varying structured output support.
        """
        # Log request details
        request_chars = len(system) + len(user)
        if self.verbose:
            print("\n[OpenRouter Request]")
            print(f"  Model: {self.model_name}")
            print(f"  Schema: {schema.__name__}")
            print(f"  Total prompt: {request_chars:,} chars (~{request_chars//4:,} tokens)")

        last_err = None

        for attempt in range(self.retries):
            try:
                attempt_start = time.time()
                if self.verbose:
                    print(f"  Attempt {attempt + 1}/{self.retries}...")

                # Use JSON mode with schema instruction in prompt
                # This works reliably across all OpenRouter models
                json_instruction = (
                    f"\nRespond with valid JSON matching this schema: {schema.model_json_schema()}"
                )
                enhanced_system = system + json_instruction

                messages = [
                    {"role": "system", "content": enhanced_system},
                    {"role": "user", "content": user},
                ]

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    timeout=self.timeout,
                    response_format={"type": "json_object"},
                )

                # Parse JSON response
                json_str = completion.choices[0].message.content
                parsed_result = schema.model_validate_json(json_str)

                # Log response details
                response_time = time.time() - attempt_start
                response_content = completion.choices[0].message.content or ""

                # Store token usage
                if hasattr(completion, "usage") and completion.usage:
                    self._last_token_usage = {
                        "input_tokens": completion.usage.prompt_tokens or 0,
                        "output_tokens": completion.usage.completion_tokens or 0,
                        "total_tokens": completion.usage.total_tokens or 0,
                    }

                if self.verbose:
                    print(f"  Response in {response_time:.2f}s ({len(response_content):,} chars)")
                    if hasattr(completion, "usage") and completion.usage:
                        print(f"  Tokens: {completion.usage.total_tokens}")

                return parsed_result

            except Exception as e:
                last_err = e
                if self.verbose:
                    print(f"  Error: {e}")
                if attempt < self.retries - 1:
                    sleep_time = random.uniform(self.backoff_min, self.backoff_max)
                    if self.verbose:
                        print(f"  Retrying after {sleep_time:.2f}s...")
                    time.sleep(sleep_time)

        raise RuntimeError(f"OpenRouter call failed after {self.retries} attempts: {last_err}")

    def raw(self, *, system: str, user: str) -> str:
        """Make a plain text call."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        last_err = None
        for attempt in range(self.retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, timeout=self.timeout
                )

                # Store token usage
                if hasattr(completion, "usage") and completion.usage:
                    self._last_token_usage = {
                        "input_tokens": completion.usage.prompt_tokens or 0,
                        "output_tokens": completion.usage.completion_tokens or 0,
                        "total_tokens": completion.usage.total_tokens or 0,
                    }

                return completion.choices[0].message.content

            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    sleep_time = random.uniform(self.backoff_min, self.backoff_max)
                    time.sleep(sleep_time)

        raise RuntimeError(f"OpenRouter raw call failed after {self.retries} attempts: {last_err}")

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "OpenRouter"

    @property
    def supports_thinking(self) -> bool:
        """OpenRouter models don't have explicit thinking mode."""
        return False

    def get_last_token_usage(self) -> dict[str, int] | None:
        """Return token usage from the last call if available."""
        return self._last_token_usage
