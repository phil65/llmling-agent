"""Langfuse prompt provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langfuse import Langfuse  # pyright: ignore

from llmling_agent.prompts.base import BasePromptProvider


if TYPE_CHECKING:
    from llmling_agent_config.prompt_hubs import LangfuseConfig


class LangfusePromptHub(BasePromptProvider):
    """Langfuse prompt provider implementation."""

    name = "langfuse"
    supports_versions = True
    supports_variables = True

    def __init__(self, config: LangfuseConfig) -> None:
        self.config = config
        secret = config.secret_key.get_secret_value()
        pub = config.public_key.get_secret_value()
        self._client = Langfuse(secret_key=secret, public_key=pub, host=str(config.host))

        from langfuse.api.client import FernLangfuse

        self._api_client = FernLangfuse(
            base_url=str(config.host),
            x_langfuse_public_key=pub,
            username=pub,
            password=secret,
        )

    async def get_prompt(
        self,
        name: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Get and optionally compile a prompt from Langfuse."""
        prompt = self._client.get_prompt(
            name,
            type="text",
            version=int(version) if version else None,
            cache_ttl_seconds=self.config.cache_ttl_seconds,
            max_retries=self.config.max_retries,
            fetch_timeout_seconds=self.config.fetch_timeout_seconds,
        )
        if variables:
            return prompt.compile(**variables)
        return prompt.prompt

    async def list_prompts(self) -> list[str]:
        """List available prompts from Langfuse."""
        try:
            response = self._api_client.prompts.list()
            return [prompt_meta.name for prompt_meta in response.data]
        except Exception:  # noqa: BLE001
            # Fallback to empty list if listing fails
            return []


if __name__ == "__main__":
    import asyncio

    from pydantic import SecretStr

    from llmling_agent_config.prompt_hubs import LangfuseConfig

    config = LangfuseConfig(secret_key=SecretStr("test"), public_key=SecretStr(""))
    prompt_hub = LangfusePromptHub(config)

    async def main() -> None:
        print(await prompt_hub.list_prompts())

    asyncio.run(main())
