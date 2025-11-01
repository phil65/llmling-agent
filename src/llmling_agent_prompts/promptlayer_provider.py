"""PromptLayer prompt provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from promptlayer import PromptLayer  # pyright: ignore

from llmling_agent.prompts.base import BasePromptProvider


if TYPE_CHECKING:
    from llmling_agent_config.prompt_hubs import PromptLayerConfig


class PromptLayerProvider(BasePromptProvider):
    """PromptLayer provider."""

    name = "promptlayer"
    supports_versions = True

    def __init__(self, config: PromptLayerConfig):
        self.client = PromptLayer(
            api_key=config.api_key.get_secret_value() if config.api_key else None
        )

    async def get_prompt(
        self,
        name: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        # PromptLayer primarily tracks prompts used with their API
        # But also allows template management
        return self.client.templates.get(name)


if __name__ == "__main__":
    from llmling_agent_config.prompt_hubs import PromptLayerConfig

    async def main():
        config = PromptLayerConfig(api_key="pl_480ead79b098fc25c63cdb4c95115deb")
        provider = PromptLayerProvider(config)
        prompt = await provider.get_prompt("example_prompt")
        print(prompt)

    import asyncio

    asyncio.run(main())
