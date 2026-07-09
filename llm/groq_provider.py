"""Groq provider implementation."""

from __future__ import annotations

import os
import random
import time
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .base_provider import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)


class GroqProvider(BaseLLMProvider):
    """Groq API provider implementation.

    Groq provides high-speed inference for open-source LLMs (Llama, Mixtral, etc.)
    through an OpenAI-compatible API.
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
        """Initialize Groq provider."""
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
        groq_cfg = config.get("groq", {}) if isinstance(config, dict) else {}
        api_key_env = groq_cfg.get("api_key_env", "GROQ_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")

        # Get base URL from config or use default Groq endpoint
        base_url = groq_cfg.get("base_url", "https://api.groq.com/openai/v1")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def parse(self, *, system: str, user: str, schema: type[T]) -> T:
        """Make a structured call using Groq's API.

        Uses JSON mode with schema instructions for reliable structured output.
        """
        # Log request details
        request_chars = len(system) + len(user)
        if self.verbose:
            print("\n[Groq Request]")
            print(f"  Model: {self.model_name}")
            print(f"  Schema: {schema.__name__}")
            print(f"  Total prompt: {request_chars:,} chars (~{request_chars//4:,} tokens)")

        last_err = None

        for attempt in range(self.retries):
            try:
                attempt_start = time.time()
                if self.verbose:
                    print(f"  Attempt {attempt + 1}/{self.retries}...")

                # Try structured output first (Groq supports it for some models)
                try:
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]

                    completion = self.client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=messages,
                        response_format=schema,
                        timeout=self.timeout,
                    )

                    # Get the parsed response if available
                    if completion.choices[0].message.parsed:
                        parsed_result = completion.choices[0].message.parsed
                    elif completion.choices[0].message.refusal:
                        raise RuntimeError(
                            f"Model refused: {completion.choices[0].message.refusal}"
                        )
                    else:
                        # Manual parsing fallback
                        json_str = completion.choices[0].message.content
                        parsed_result = schema.model_validate_json(json_str)

                except Exception as structured_err:
                    # Fallback to JSON mode with schema instruction
                    if self.verbose:
                        print(
                            f"  Structured output failed, falling back to JSON mode: {structured_err}"
                        )

                    json_instruction = f"\nRespond with valid JSON matching this schema: {schema.model_json_schema()}"
                    enhanced_system = system + json_instruction

                    enhanced_messages = [
                        {"role": "system", "content": enhanced_system},
                        {"role": "user", "content": user},
                    ]

                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=enhanced_messages,
                        timeout=self.timeout,
                        response_format={"type": "json_object"},
                    )

                    # Parse JSON response manually
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

        raise RuntimeError(f"Groq call failed after {self.retries} attempts: {last_err}")

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

        raise RuntimeError(f"Groq raw call failed after {self.retries} attempts: {last_err}")

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "Groq"

    @property
    def supports_thinking(self) -> bool:
        """Groq models don't have explicit thinking mode."""
        return False

    def get_last_token_usage(self) -> dict[str, int] | None:
        """Return token usage from the last call if available."""
        return self._last_token_usage
