"""Tests for the Atlas Cloud provider."""

import os
import unittest
from unittest.mock import patch

from llm.atlascloud_provider import AtlasCloudProvider


class DummyOpenAIClient:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs


class TestAtlasCloudProvider(unittest.TestCase):
    @patch("llm.openai_provider.OpenAI", DummyOpenAIClient)
    def test_uses_atlas_credentials_and_endpoint(self):
        with patch.dict(
            os.environ,
            {
                "ATLASCLOUD_API_KEY": "atlas-test-key",
                "OPENAI_BASE_URL": "https://ignored.example/v1",
            },
        ):
            provider = AtlasCloudProvider({}, "deepseek-ai/deepseek-v4-pro")

        self.assertEqual(provider.provider_name, "Atlas Cloud")
        self.assertEqual(provider.client.init_kwargs["api_key"], "atlas-test-key")
        self.assertEqual(
            str(provider.client.init_kwargs["base_url"]),
            "https://api.atlascloud.ai/v1",
        )

    def test_missing_atlas_api_key_is_reported(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ATLASCLOUD_API_KEY"):
                AtlasCloudProvider({}, "deepseek-ai/deepseek-v4-pro")

    def test_unified_client_selects_atlascloud(self):
        cfg = {
            "models": {
                "graph": {
                    "provider": "atlascloud",
                    "model": "deepseek-ai/deepseek-v4-pro",
                }
            }
        }

        class DummyAtlasCloud:
            provider_name = "Atlas Cloud"
            supports_thinking = False

            def __init__(self, **kwargs):
                self.init_kwargs = kwargs

        with patch("llm.unified_client.AtlasCloudProvider", DummyAtlasCloud):
            from llm.unified_client import UnifiedLLMClient

            client = UnifiedLLMClient(cfg, profile="graph")

        self.assertEqual(client.provider.provider_name, "Atlas Cloud")
        self.assertEqual(client.provider.init_kwargs["model_name"], "deepseek-ai/deepseek-v4-pro")


if __name__ == "__main__":
    unittest.main()
