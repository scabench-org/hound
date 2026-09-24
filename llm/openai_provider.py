"""OpenAI provider implementation."""
from __future__ import annotations

import os
import random
import time
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .base_provider import BaseLLMProvider

T = TypeVar('T', bound=BaseModel)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(
        self, 
        config: dict[str, Any], 
        model_name: str,
        timeout: int = 120,
        retries: int = 3,
        backoff_min: float = 2.0,
        backoff_max: float = 8.0,
        reasoning_effort: str | None = None,
        text_verbosity: str | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
        **kwargs
    ):
        """Initialize OpenAI provider."""
        self.config = config
        self.model_name = model_name
        self.timeout = timeout
        self.retries = retries
        self.backoff_min = backoff_min
        self.backoff_max = backoff_max
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        # Verbose logging toggle (suppress request logs by default)
        logging_cfg = config.get("logging", {}) if isinstance(config, dict) else {}
        env_verbose = os.environ.get("HOUND_LLM_VERBOSE", "").lower() in {"1","true","yes","on"}
        self.verbose = bool(logging_cfg.get("llm_verbose", False) or env_verbose)
        self._last_token_usage = None
        
        # Get API key from environment
        resolved_api_key_env = api_key_env or config.get("openai", {}).get(
            "api_key_env", "OPENAI_API_KEY"
        )
        api_key = os.environ.get(resolved_api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {resolved_api_key_env}"
            )
        
        # Allow custom base URL via environment variable; default to public OpenAI endpoint
        # IMPORTANT: OpenAI Python SDK expects base_url to include the "/v1" path.
        # Normalize input so both "https://api.openai.com" and "https://api.openai.com/v1" work.
        raw_base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or config.get("openai", {}).get("base_url")
        )
        base_url = (raw_base_url or "https://api.openai.com/v1").rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        if self.verbose:
            try:
                print(f"[OpenAI Provider] Using base_url: {base_url}")
            except Exception:
                pass
        # Prefer Responses API for GPT-5 family unless overridden
        mdl = (self.model_name or "").lower()
        env_force_resp = os.environ.get("HOUND_OPENAI_USE_RESPONSES", "").lower() in {"1","true","yes","on"}
        self.use_responses = bool(env_force_resp or mdl.startswith("gpt-5"))
    
    def parse(self, *, system: str, user: str, schema: type[T], reasoning_effort: str | None = None) -> T:
        """Make a structured call. Uses Responses API for GPT-5; otherwise Chat Completions parse."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        # Log request details
        request_chars = len(system) + len(user)
        if self.verbose:
            print("\n[OpenAI Request]")
            print(f"  Model: {self.model_name}")
            print(f"  Schema: {schema.__name__}")
            print(f"  Total prompt: {request_chars:,} chars (~{request_chars//4:,} tokens)")
        
        last_err = None
        
        for attempt in range(self.retries):
            try:
                time.time()
                if self.verbose:
                    print(f"  Attempt {attempt + 1}/{self.retries}...")
                if self.use_responses:
                    # Responses API without server-side schema; we instruct strict JSON and validate locally
                    text_params: dict[str, Any] = {}
                    if self.text_verbosity:
                        text_params['verbosity'] = self.text_verbosity
                    # Embed schema hint in instructions to increase compliance
                    try:
                        json_schema = schema.model_json_schema()
                    except Exception:
                        try:
                            json_schema = schema.schema()
                        except Exception:
                            json_schema = None
                    schema_hint = ""
                    if isinstance(json_schema, dict):
                        import json as _json
                        schema_hint = "\nFollow this JSON schema exactly (no extra keys, all required):\n" + _json.dumps(json_schema)
                    strict_instr = "\nReturn ONLY valid JSON. No markdown. No prose."
                    params: dict[str, Any] = {
                        'model': self.model_name,
                        'input': user,
                        'instructions': (system or '') + schema_hint + strict_instr,
                    }
                    if text_params:
                        params['text'] = text_params
                    eff = reasoning_effort or self.reasoning_effort
                    if eff:
                        params['reasoning'] = {'effort': eff}
                    resp = self.client.responses.create(**params)

                    # Usage
                    try:
                        usage = getattr(resp, 'usage', None)
                        if usage:
                            self._last_token_usage = {
                                'input_tokens': getattr(usage, 'input_tokens', 0) or 0,
                                'output_tokens': getattr(usage, 'output_tokens', 0) or 0,
                                'total_tokens': getattr(usage, 'total_tokens', 0) or 0,
                            }
                    except Exception:
                        pass

                    output_text = getattr(resp, 'output_text', None)
                    if not output_text:
                        # Fallback: try to reconstruct from output items
                        try:
                            items = getattr(resp, 'output', []) or []
                            for it in items:
                                cont = it.get('content') if isinstance(it, dict) else None
                                if cont and isinstance(cont, list):
                                    for c in cont:
                                        if isinstance(c, dict) and c.get('type') in {'output_text', 'text'} and c.get('text'):
                                            output_text = c.get('text')
                                            break
                                if output_text:
                                    break
                        except Exception:
                            pass
                    if not output_text:
                        raise RuntimeError("No output_text in response")
                    return schema.model_validate_json(output_text)
                else:
                    # Chat Completions structured output path
                    completion = self.client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=messages,
                        response_format=schema,
                        timeout=self.timeout
                    )
                    if hasattr(completion, 'usage') and completion.usage:
                        self._last_token_usage = {
                            'input_tokens': completion.usage.prompt_tokens or 0,
                            'output_tokens': completion.usage.completion_tokens or 0,
                            'total_tokens': completion.usage.total_tokens or 0
                        }
                    if completion.choices[0].message.parsed:
                        return completion.choices[0].message.parsed
                    elif completion.choices[0].message.refusal:
                        raise RuntimeError(f"Model refused: {completion.choices[0].message.refusal}")
                    else:
                        json_str = completion.choices[0].message.content
                        return schema.model_validate_json(json_str)
                    
            except Exception as e:
                last_err = e
                if self.verbose:
                    print(f"  Error: {e}")
                if attempt < self.retries - 1:
                    sleep_time = random.uniform(self.backoff_min, self.backoff_max)
                    if self.verbose:
                        print(f"  Retrying after {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
        
        raise RuntimeError(f"OpenAI call failed after {self.retries} attempts: {last_err}")
    
    def raw(self, *, system: str, user: str, reasoning_effort: str | None = None) -> str:
        """Make a plain text call."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        last_err = None
        for attempt in range(self.retries):
            try:
                if self.use_responses:
                    # Favor JSON output only when caller explicitly asks for strict JSON in the system message.
                    # This keeps deep_think and other free-text prompts working as normal text.
                    text_params: dict[str, Any] = {}
                    if self.text_verbosity:
                        text_params['verbosity'] = self.text_verbosity
                    params: dict[str, Any] = {
                        'model': self.model_name,
                        'input': user,
                        'instructions': system or '',
                    }
                    # Heuristic: if system asks for "valid JSON", request json_object formatting.
                    try:
                        if isinstance(system, str) and 'valid json' in system.lower():
                            text_params['format'] = {'type': 'json_object'}
                    except Exception:
                        pass
                    if text_params:
                        params['text'] = text_params
                    eff = reasoning_effort or self.reasoning_effort
                    if eff:
                        params['reasoning'] = {'effort': eff}
                    resp = self.client.responses.create(**params)
                    # Usage
                    try:
                        usage = getattr(resp, 'usage', None)
                        if usage:
                            self._last_token_usage = {
                                'input_tokens': getattr(usage, 'input_tokens', 0) or 0,
                                'output_tokens': getattr(usage, 'output_tokens', 0) or 0,
                                'total_tokens': getattr(usage, 'total_tokens', 0) or 0,
                            }
                    except Exception:
                        pass
                    # Prefer SDK convenience property; fall back to concatenating text parts when absent
                    out = getattr(resp, 'output_text', None)
                    if out is not None:
                        return out
                    try:
                        # Aggregate any text parts from the response
                        chunks = []
                        for item in getattr(resp, 'output', []) or []:
                            for c in getattr(item, 'content', []) or []:
                                t = getattr(c, 'text', None)
                                if t and getattr(t, 'value', None):
                                    chunks.append(t.value)
                        return '\n'.join(chunks)
                    except Exception:
                        return ""
                else:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        timeout=self.timeout
                    )
                    # Store token usage
                    if hasattr(completion, 'usage') and completion.usage:
                        self._last_token_usage = {
                            'input_tokens': completion.usage.prompt_tokens or 0,
                            'output_tokens': completion.usage.completion_tokens or 0,
                            'total_tokens': completion.usage.total_tokens or 0
                        }
                    return completion.choices[0].message.content
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    sleep_time = random.uniform(self.backoff_min, self.backoff_max)
                    time.sleep(sleep_time)
        
        raise RuntimeError(f"OpenAI raw call failed after {self.retries} attempts: {last_err}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "OpenAI"
    
    @property
    def supports_thinking(self) -> bool:
        """OpenAI models may support reasoning effort but not explicit thinking mode."""
        return False
    
    def get_last_token_usage(self) -> dict[str, int] | None:
        """Return token usage from the last call if available."""
        return self._last_token_usage
