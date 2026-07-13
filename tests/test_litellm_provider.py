"""Tests for LiteLLM provider."""
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from llm.litellm_provider import LiteLLMProvider, _is_non_retryable


class SampleSchema(BaseModel):
    answer: str
    confidence: float


def _make_provider(**overrides):
    defaults = {
        "config": {},
        "model_name": "anthropic/claude-haiku-4-5",
        "retries": 1,
        "backoff_min": 0.0,
        "backoff_max": 0.0,
    }
    defaults.update(overrides)
    return LiteLLMProvider(**defaults)


def _mock_response(content="Hello", prompt_tokens=10, completion_tokens=5):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


class TestLiteLLMProviderInit(unittest.TestCase):
    def test_default_config(self):
        p = _make_provider()
        assert p.model_name == "anthropic/claude-haiku-4-5"
        assert p.provider_name == "LiteLLM"
        assert p.supports_thinking is False
        assert p.api_key is None
        assert p.api_base is None

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"LITELLM_API_KEY": "sk-test"}):
            p = _make_provider()
            assert p.api_key == "sk-test"

    def test_custom_api_key_env(self):
        with patch.dict("os.environ", {"MY_KEY": "sk-custom"}):
            p = _make_provider(config={"litellm": {"api_key_env": "MY_KEY"}})
            assert p.api_key == "sk-custom"

    def test_api_base_from_config(self):
        p = _make_provider(config={"litellm": {"api_base": "http://proxy:4000"}})
        assert p.api_base == "http://proxy:4000"

    def test_api_base_from_env(self):
        with patch.dict("os.environ", {"LITELLM_API_BASE": "http://env-proxy:8000"}):
            p = _make_provider()
            assert p.api_base == "http://env-proxy:8000"


class TestLiteLLMProviderRaw(unittest.TestCase):
    def test_raw_success(self):
        p = _make_provider()
        mock_resp = _mock_response(content="The answer is 4")
        with patch.object(p, "_completion", return_value=mock_resp) as mock_comp:
            result = p.raw(system="You are helpful", user="What is 2+2?")
            assert result == "The answer is 4"
            mock_comp.assert_called_once()
            args = mock_comp.call_args
            messages = args[0][0]
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    def test_raw_none_content_raises(self):
        p = _make_provider()
        mock_resp = _mock_response(content=None)
        with patch.object(p, "_completion", return_value=mock_resp):
            with self.assertRaises(RuntimeError) as ctx:
                p.raw(system="S", user="U")
            assert "Empty response" in str(ctx.exception)

    def test_raw_retries_on_transient_error(self):
        p = _make_provider(retries=3, backoff_min=0, backoff_max=0)
        call_count = 0

        def _failing_then_success(messages, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("network blip")
            return _mock_response(content="recovered")

        with patch.object(p, "_completion", side_effect=_failing_then_success):
            result = p.raw(system="S", user="U")
            assert result == "recovered"
            assert call_count == 3

    def test_raw_no_retry_on_auth_error(self):
        p = _make_provider(retries=3, backoff_min=0, backoff_max=0)

        auth_exc = type("AuthenticationError", (Exception,), {
            "__module__": "litellm.exceptions",
            "__qualname__": "AuthenticationError",
        })("Invalid API key")

        with patch.object(p, "_completion", side_effect=auth_exc):
            with self.assertRaises(RuntimeError) as ctx:
                p.raw(system="S", user="U")
            assert "Invalid API key" in str(ctx.exception)


class TestLiteLLMProviderParse(unittest.TestCase):
    def test_parse_success(self):
        p = _make_provider()
        json_content = '{"answer": "four", "confidence": 0.95}'
        mock_resp = _mock_response(content=json_content)
        with patch.object(p, "_completion", return_value=mock_resp) as mock_comp:
            result = p.parse(system="Return JSON", user="What is 2+2?", schema=SampleSchema)
            assert isinstance(result, SampleSchema)
            assert result.answer == "four"
            assert result.confidence == 0.95
            call_kwargs = mock_comp.call_args
            assert call_kwargs[1]["response_format"] == {"type": "json_object"}

    def test_parse_none_content_raises(self):
        p = _make_provider()
        mock_resp = _mock_response(content=None)
        with patch.object(p, "_completion", return_value=mock_resp):
            with self.assertRaises(RuntimeError) as ctx:
                p.parse(system="S", user="U", schema=SampleSchema)
            assert "Empty response" in str(ctx.exception)

    def test_parse_invalid_json_retries(self):
        p = _make_provider(retries=2, backoff_min=0, backoff_max=0)
        call_count = 0

        def _bad_then_good(messages, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response(content="not valid json{{{")
            return _mock_response(content='{"answer": "ok", "confidence": 1.0}')

        with patch.object(p, "_completion", side_effect=_bad_then_good):
            result = p.parse(system="S", user="U", schema=SampleSchema)
            assert result.answer == "ok"
            assert call_count == 2

    def test_parse_no_retry_on_not_found(self):
        p = _make_provider(retries=3, backoff_min=0, backoff_max=0)

        not_found = type("NotFoundError", (Exception,), {
            "__module__": "litellm.exceptions",
            "__qualname__": "NotFoundError",
        })("Model not found: bad/model")

        with patch.object(p, "_completion", side_effect=not_found):
            with self.assertRaises(RuntimeError) as ctx:
                p.parse(system="S", user="U", schema=SampleSchema)
            assert "Model not found" in str(ctx.exception)


class TestLiteLLMProviderTokenUsage(unittest.TestCase):
    def test_token_usage_tracked(self):
        p = _make_provider()
        mock_resp = _mock_response(content="hi", prompt_tokens=100, completion_tokens=50)
        with patch.object(p, "_completion", return_value=mock_resp):
            p.raw(system="S", user="U")
            usage = p.get_last_token_usage()
            assert usage is not None
            assert usage["input_tokens"] == 100
            assert usage["output_tokens"] == 50
            assert usage["total_tokens"] == 150

    def test_token_usage_none_initially(self):
        p = _make_provider()
        assert p.get_last_token_usage() is None

    def test_token_usage_survives_missing_usage(self):
        p = _make_provider()
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
        )
        with patch.object(p, "_completion", return_value=resp):
            p.raw(system="S", user="U")
            assert p.get_last_token_usage() is None


class TestLiteLLMProviderCompletion(unittest.TestCase):
    def test_completion_passes_drop_params(self):
        p = _make_provider()
        fake_litellm = MagicMock()
        fake_litellm.completion.return_value = _mock_response(content="ok")
        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            p._completion([{"role": "user", "content": "hi"}])
            call_kwargs = fake_litellm.completion.call_args[1]
            assert call_kwargs["drop_params"] is True

    def test_completion_passes_api_key_when_set(self):
        p = _make_provider()
        p.api_key = "sk-test-key"
        fake_litellm = MagicMock()
        fake_litellm.completion.return_value = _mock_response(content="ok")
        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            p._completion([{"role": "user", "content": "hi"}])
            call_kwargs = fake_litellm.completion.call_args[1]
            assert call_kwargs["api_key"] == "sk-test-key"

    def test_completion_omits_api_key_when_none(self):
        p = _make_provider()
        p.api_key = None
        fake_litellm = MagicMock()
        fake_litellm.completion.return_value = _mock_response(content="ok")
        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            p._completion([{"role": "user", "content": "hi"}])
            call_kwargs = fake_litellm.completion.call_args[1]
            assert "api_key" not in call_kwargs

    def test_completion_passes_api_base_when_set(self):
        p = _make_provider()
        p.api_base = "http://my-proxy:4000"
        fake_litellm = MagicMock()
        fake_litellm.completion.return_value = _mock_response(content="ok")
        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            p._completion([{"role": "user", "content": "hi"}])
            call_kwargs = fake_litellm.completion.call_args[1]
            assert call_kwargs["api_base"] == "http://my-proxy:4000"

    def test_import_error_message(self):
        p = _make_provider()
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "litellm":
                raise ImportError("No module named 'litellm'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with self.assertRaises(ImportError) as ctx:
                p._completion([{"role": "user", "content": "hi"}])
            assert "pip install" in str(ctx.exception)


class TestNonRetryableHelper(unittest.TestCase):
    def test_auth_error_is_non_retryable(self):
        exc = type("AuthenticationError", (Exception,), {
            "__module__": "litellm.exceptions",
            "__qualname__": "AuthenticationError",
        })("bad key")
        assert _is_non_retryable(exc) is True

    def test_not_found_is_non_retryable(self):
        exc = type("NotFoundError", (Exception,), {
            "__module__": "litellm.exceptions",
            "__qualname__": "NotFoundError",
        })("model not found")
        assert _is_non_retryable(exc) is True

    def test_connection_error_is_retryable(self):
        assert _is_non_retryable(ConnectionError("timeout")) is False

    def test_generic_exception_is_retryable(self):
        assert _is_non_retryable(RuntimeError("something broke")) is False

    def test_rate_limit_is_retryable(self):
        exc = type("RateLimitError", (Exception,), {
            "__module__": "litellm.exceptions",
            "__qualname__": "RateLimitError",
        })("429")
        assert _is_non_retryable(exc) is False


class TestUnifiedClientLiteLLMRegistration(unittest.TestCase):
    def test_litellm_provider_selected(self):
        cfg = {"models": {"graph": {"provider": "litellm", "model": "anthropic/claude-haiku-4-5"}}}

        class DummyLiteLLM:
            provider_name = "LiteLLM"
            supports_thinking = False
            def __init__(self, **kw): pass
            def raw(self, *, system, user): return "ok"

        with patch("llm.unified_client.LiteLLMProvider", DummyLiteLLM):
            from llm.unified_client import UnifiedLLMClient
            uc = UnifiedLLMClient(cfg, profile="graph")
            assert uc.provider.provider_name == "LiteLLM"


if __name__ == "__main__":
    unittest.main()
