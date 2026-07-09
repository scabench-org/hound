"""Unit tests for LLM provider token tracking."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.anthropic_provider import AnthropicProvider
from llm.gemini_provider import GeminiProvider
from llm.groq_provider import GroqProvider
from llm.openai_provider import OpenAIProvider
from llm.openrouter_provider import OpenRouterProvider
from llm.xai_provider import XAIProvider


class TestOpenAIProviderTokenTracking(unittest.TestCase):
    """Test OpenAI provider token tracking."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("llm.openai_provider.OpenAI")
    def setUp(self, mock_openai):
        """Set up test fixtures."""
        self.mock_client = Mock()
        mock_openai.return_value = self.mock_client

        self.provider = OpenAIProvider(
            config={"openai": {"api_key_env": "OPENAI_API_KEY"}}, model_name="gpt-4", timeout=30
        )

    def test_initial_token_usage_is_none(self):
        """Test that initial token usage is None."""
        self.assertIsNone(self.provider.get_last_token_usage())

    def test_raw_call_tracks_tokens(self):
        """Test that raw calls track token usage."""
        # Mock response with usage data
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        self.mock_client.chat.completions.create.return_value = mock_response

        # Make a raw call
        result = self.provider.raw(system="Test system", user="Test user")

        # Check result
        self.assertEqual(result, "Test response")

        # Check token tracking
        usage = self.provider.get_last_token_usage()
        self.assertIsNotNone(usage)
        self.assertEqual(usage["input_tokens"], 100)
        self.assertEqual(usage["output_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 150)

    def test_parse_call_tracks_tokens(self):
        """Test that parse calls track token usage."""
        # Mock response with usage data
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = '{"test": "data"}'
        mock_message.parsed = Mock(test="data")
        mock_message.refusal = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_response.usage = Mock(prompt_tokens=200, completion_tokens=100, total_tokens=300)
        self.mock_client.beta.chat.completions.parse.return_value = mock_response

        # Create a mock schema
        class TestSchema:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        # Make a parse call
        self.provider.parse(system="Test", user="Test", schema=TestSchema)

        # Check token tracking
        usage = self.provider.get_last_token_usage()
        self.assertIsNotNone(usage)
        self.assertEqual(usage["input_tokens"], 200)
        self.assertEqual(usage["output_tokens"], 100)
        self.assertEqual(usage["total_tokens"], 300)


class TestGeminiProviderTokenTracking(unittest.TestCase):
    """Test Gemini provider token tracking."""

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"})
    @patch("llm.gemini_provider._USE_NEW_GENAI", False)
    @patch("llm.gemini_provider._genai_new", None)
    @patch("llm.gemini_provider._genai_legacy", None)
    @patch("llm.gemini_provider.genai")
    def setUp(self, mock_genai, *_):
        """Set up test fixtures."""
        mock_genai.configure = Mock()
        self.mock_model = Mock()
        mock_genai.GenerativeModel.return_value = self.mock_model

        self.provider = GeminiProvider(
            config={"gemini": {"api_key_env": "GOOGLE_API_KEY"}}, model_name="gemini-2.0-flash"
        )

    def test_initial_token_usage_is_none(self):
        """Test that initial token usage is None."""
        self.assertIsNone(self.provider.get_last_token_usage())

    def test_raw_call_tracks_tokens(self):
        """Test that raw calls track token usage."""
        # Mock response with usage metadata
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_response.usage_metadata = Mock(
            prompt_token_count=150, candidates_token_count=75, total_token_count=225
        )
        self.mock_model.generate_content.return_value = mock_response

        # Make a raw call
        result = self.provider.raw(system="Test system", user="Test user")

        # Check result
        self.assertEqual(result, "Test response")

        # Check token tracking
        usage = self.provider.get_last_token_usage()
        self.assertIsNotNone(usage)
        self.assertEqual(usage["input_tokens"], 150)
        self.assertEqual(usage["output_tokens"], 75)
        self.assertEqual(usage["total_tokens"], 225)


class TestAnthropicProviderTokenTracking(unittest.TestCase):
    """Test Anthropic provider token tracking."""

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
    @patch("anthropic.Anthropic")
    def setUp(self, mock_anthropic_class):
        """Set up test fixtures."""
        self.mock_client = Mock()
        mock_anthropic_class.return_value = self.mock_client

        self.provider = AnthropicProvider(
            model_name="claude-3-5-sonnet", api_key_env="ANTHROPIC_API_KEY"
        )

    def test_initial_token_usage_is_none(self):
        """Test that initial token usage is None."""
        self.assertIsNone(self.provider.get_last_token_usage())

    def test_raw_call_tracks_tokens(self):
        """Test that raw calls track token usage."""
        # Mock response with usage data
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.usage = Mock(input_tokens=120, output_tokens=60)
        self.mock_client.messages.create.return_value = mock_response

        # Make a raw call
        result = self.provider.raw(system="Test system", user="Test user")

        # Check result
        self.assertEqual(result, "Test response")

        # Check token tracking
        usage = self.provider.get_last_token_usage()
        self.assertIsNotNone(usage)
        self.assertEqual(usage["input_tokens"], 120)
        self.assertEqual(usage["output_tokens"], 60)
        self.assertEqual(usage["total_tokens"], 180)


class TestXAIProviderTokenTracking(unittest.TestCase):
    """Test XAI provider token tracking."""

    @patch.dict("os.environ", {"XAI_API_KEY": "test_key"})
    @patch("llm.xai_provider.OpenAI")
    def setUp(self, mock_openai):
        """Set up test fixtures."""
        self.mock_client = Mock()
        mock_openai.return_value = self.mock_client

        self.provider = XAIProvider(
            config={"xai": {"api_key_env": "XAI_API_KEY"}}, model_name="grok-2"
        )

    def test_initial_token_usage_is_none(self):
        """Test that initial token usage is None."""
        self.assertIsNone(self.provider.get_last_token_usage())

    def test_raw_call_tracks_tokens(self):
        """Test that raw calls track token usage."""
        # Mock response with usage data (XAI uses OpenAI-compatible API)
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(prompt_tokens=90, completion_tokens=45, total_tokens=135)
        self.mock_client.chat.completions.create.return_value = mock_response

        # Make a raw call
        result = self.provider.raw(system="Test system", user="Test user")

        # Check result
        self.assertEqual(result, "Test response")

        # Check token tracking
        usage = self.provider.get_last_token_usage()
        self.assertIsNotNone(usage)
        self.assertEqual(usage["input_tokens"], 90)
        self.assertEqual(usage["output_tokens"], 45)
        self.assertEqual(usage["total_tokens"], 135)


class TestOpenRouterProviderTokenTracking(unittest.TestCase):
    """Test OpenRouter provider token tracking."""

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"})
    @patch("llm.openrouter_provider.OpenAI")
    def setUp(self, mock_openai):
        """Set up test fixtures."""
        self.mock_client = Mock()
        mock_openai.return_value = self.mock_client

        self.provider = OpenRouterProvider(
            config={"openrouter": {"api_key_env": "OPENROUTER_API_KEY"}},
            model_name="meta-llama/llama-3.1-70b-instruct",
        )

    def test_initial_token_usage_is_none(self):
        """Test that initial token usage is None."""
        self.assertIsNone(self.provider.get_last_token_usage())

    def test_raw_call_tracks_tokens(self):
        """Test that raw calls track token usage."""
        # Mock response with usage data (OpenRouter uses OpenAI-compatible API)
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(prompt_tokens=80, completion_tokens=40, total_tokens=120)
        self.mock_client.chat.completions.create.return_value = mock_response

        # Make a raw call
        result = self.provider.raw(system="Test system", user="Test user")

        # Check result
        self.assertEqual(result, "Test response")

        # Check token tracking
        usage = self.provider.get_last_token_usage()
        self.assertIsNotNone(usage)
        self.assertEqual(usage["input_tokens"], 80)
        self.assertEqual(usage["output_tokens"], 40)
        self.assertEqual(usage["total_tokens"], 120)


class TestGroqProviderTokenTracking(unittest.TestCase):
    """Test Groq provider token tracking."""

    @patch.dict("os.environ", {"GROQ_API_KEY": "test_key"})
    @patch("llm.groq_provider.OpenAI")
    def setUp(self, mock_openai):
        """Set up test fixtures."""
        self.mock_client = Mock()
        mock_openai.return_value = self.mock_client

        self.provider = GroqProvider(
            config={"groq": {"api_key_env": "GROQ_API_KEY"}}, model_name="llama-3.3-70b-versatile"
        )

    def test_initial_token_usage_is_none(self):
        """Test that initial token usage is None."""
        self.assertIsNone(self.provider.get_last_token_usage())

    def test_raw_call_tracks_tokens(self):
        """Test that raw calls track token usage."""
        # Mock response with usage data (Groq uses OpenAI-compatible API)
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(prompt_tokens=70, completion_tokens=35, total_tokens=105)
        self.mock_client.chat.completions.create.return_value = mock_response

        # Make a raw call
        result = self.provider.raw(system="Test system", user="Test user")

        # Check result
        self.assertEqual(result, "Test response")

        # Check token tracking
        usage = self.provider.get_last_token_usage()
        self.assertIsNotNone(usage)
        self.assertEqual(usage["input_tokens"], 70)
        self.assertEqual(usage["output_tokens"], 35)
        self.assertEqual(usage["total_tokens"], 105)


if __name__ == "__main__":
    unittest.main()
