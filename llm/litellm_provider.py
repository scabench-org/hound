"""LiteLLM provider implementation.

LiteLLM (https://litellm.ai) provides a unified interface to 100+ LLM
providers including OpenAI, Anthropic, Google, Azure, AWS Bedrock, and more.

Models are specified using LiteLLM's provider/model format, e.g.:
  - "anthropic/claude-sonnet-4-6"
  - "azure/gpt-4o"
  - "bedrock/anthropic.claude-3-haiku-20240307-v1:0"
  - "vertex_ai/gemini-2.0-flash"

Provider API keys are read from environment variables automatically by LiteLLM
(e.g. ANTHROPIC_API_KEY, AZURE_API_KEY). No additional configuration is needed
beyond setting the relevant env var for your chosen provider.
"""
from __future__ import annotations

import json as _json
import os
import random
import time
from typing import Any, TypeVar

from pydantic import BaseModel

from .base_provider import BaseLLMProvider

T = TypeVar('T', bound=BaseModel)


def _is_non_retryable(exc: BaseException) -> bool:
    """Check if an exception should NOT be retried (auth errors, bad model, etc.)."""
    qualname = f"{type(exc).__module__}.{type(exc).__qualname__}"
    return qualname in {
        "litellm.exceptions.AuthenticationError",
        "litellm.exceptions.NotFoundError",
        "litellm.exceptions.BadRequestError",
        "litellm.exceptions.PermissionDeniedError",
    }


class LiteLLMProvider(BaseLLMProvider):
    """LiteLLM AI gateway provider implementation."""

    def __init__(
        self,
        config: dict[str, Any],
        model_name: str,
        timeout: int = 120,
        retries: int = 3,
        backoff_min: float = 2.0,
        backoff_max: float = 8.0,
        reasoning_effort: str | None = None,
        **kwargs
    ):
        """Initialize LiteLLM provider."""
        self.config = config
        self.model_name = model_name
        self.timeout = timeout
        self.retries = retries
        self.backoff_min = backoff_min
        self.backoff_max = backoff_max
        self.reasoning_effort = reasoning_effort
        logging_cfg = config.get("logging", {}) if isinstance(config, dict) else {}
        env_verbose = os.environ.get("HOUND_LLM_VERBOSE", "").lower() in {"1", "true", "yes", "on"}
        self.verbose = bool(logging_cfg.get("llm_verbose", False) or env_verbose)
        self._last_token_usage = None

        litellm_cfg = config.get("litellm", {}) if isinstance(config, dict) else {}
        api_key_env = litellm_cfg.get("api_key_env", "LITELLM_API_KEY")
        self.api_key = os.environ.get(api_key_env) or None
        self.api_base = litellm_cfg.get("api_base") or os.environ.get("LITELLM_API_BASE") or None

    def _completion(self, messages: list[dict], **extra_kwargs) -> Any:
        """Call litellm.completion with standard kwargs."""
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for the LiteLLM provider. "
                "Install it with: pip install 'litellm>=1.83.0,<2.0'"
            )

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "timeout": self.timeout,
            "drop_params": True,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        kwargs.update(extra_kwargs)
        return litellm.completion(**kwargs)

    def _store_usage(self, response: Any) -> None:
        """Extract and store token usage from a litellm response."""
        try:
            usage = getattr(response, 'usage', None)
            if usage:
                self._last_token_usage = {
                    'input_tokens': getattr(usage, 'prompt_tokens', 0) or 0,
                    'output_tokens': getattr(usage, 'completion_tokens', 0) or 0,
                    'total_tokens': getattr(usage, 'total_tokens', 0) or 0,
                }
        except Exception:
            pass

    def parse(self, *, system: str, user: str, schema: type[T], **kwargs) -> T:
        """Make a structured call returning an instance of the schema."""
        try:
            json_schema = schema.model_json_schema()
        except Exception:
            json_schema = None

        schema_hint = ""
        if isinstance(json_schema, dict):
            schema_hint = (
                "\nRespond with valid JSON matching this schema exactly "
                "(no extra keys, all required fields):\n"
                + _json.dumps(json_schema)
            )
        json_instruction = schema_hint + "\nReturn ONLY valid JSON. No markdown. No prose."

        messages = [
            {"role": "system", "content": system + json_instruction},
            {"role": "user", "content": user},
        ]

        request_chars = len(system) + len(user)
        if self.verbose:
            print("\n[LiteLLM Request]")
            print(f"  Model: {self.model_name}")
            print(f"  Schema: {schema.__name__}")
            print(f"  Total prompt: {request_chars:,} chars (~{request_chars // 4:,} tokens)")

        last_err = None
        for attempt in range(self.retries):
            try:
                if self.verbose:
                    print(f"  Attempt {attempt + 1}/{self.retries}...")

                response = self._completion(
                    messages,
                    response_format={"type": "json_object"},
                )
                self._store_usage(response)
                json_str = response.choices[0].message.content
                if not json_str:
                    raise ValueError("Empty response content from LiteLLM")
                return schema.model_validate_json(json_str)

            except ImportError:
                raise
            except Exception as e:
                if _is_non_retryable(e):
                    raise RuntimeError(f"LiteLLM call failed: {e}") from e
                last_err = e
                if self.verbose:
                    print(f"  Error: {e}")
                if attempt < self.retries - 1:
                    sleep_time = random.uniform(self.backoff_min, self.backoff_max)
                    if self.verbose:
                        print(f"  Retrying after {sleep_time:.2f}s...")
                    time.sleep(sleep_time)

        raise RuntimeError(f"LiteLLM call failed after {self.retries} attempts: {last_err}")

    def raw(self, *, system: str, user: str, **kwargs) -> str:
        """Make a plain text call without structured output."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_err = None
        for attempt in range(self.retries):
            try:
                response = self._completion(messages)
                self._store_usage(response)
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("Empty response content from LiteLLM")
                return content

            except ImportError:
                raise
            except Exception as e:
                if _is_non_retryable(e):
                    raise RuntimeError(f"LiteLLM raw call failed: {e}") from e
                last_err = e
                if attempt < self.retries - 1:
                    sleep_time = random.uniform(self.backoff_min, self.backoff_max)
                    time.sleep(sleep_time)

        raise RuntimeError(f"LiteLLM raw call failed after {self.retries} attempts: {last_err}")

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "LiteLLM"

    @property
    def supports_thinking(self) -> bool:
        """LiteLLM delegates to the underlying provider; thinking support varies by model."""
        return False

    def get_last_token_usage(self) -> dict[str, int] | None:
        """Return token usage from the last call if available."""
        return self._last_token_usage
