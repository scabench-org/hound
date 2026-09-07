"""Atlas Cloud provider implementation."""
from __future__ import annotations

from typing import Any

from .openai_provider import OpenAIProvider


class AtlasCloudProvider(OpenAIProvider):
    """Atlas Cloud provider using its OpenAI-compatible chat API."""

    def __init__(self, config: dict[str, Any], model_name: str, **kwargs):
        atlas_config = config.get("atlascloud", {})
        super().__init__(
            config=config,
            model_name=model_name,
            api_key_env=atlas_config.get("api_key_env", "ATLASCLOUD_API_KEY"),
            base_url=atlas_config.get("base_url", "https://api.atlascloud.ai/v1"),
            **kwargs,
        )

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "Atlas Cloud"
