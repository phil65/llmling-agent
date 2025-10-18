from __future__ import annotations

from pydantic import BaseModel

from llmling_agent import AgentsManifest


class Result(BaseModel):
    """Structured response result."""

    is_positive: bool


AGENT_CONFIG = """
agents:
    summarizer:
        model: {default_model}
        system_prompts:
            - Summarize text in a structured way.
"""


async def test_structured_response(default_model: str):
    manifest = AgentsManifest.from_yaml(AGENT_CONFIG.format(default_model=default_model))
    async with manifest.pool as pool:
        agent = pool.get_agent("summarizer", return_type=Result)
        result = await agent.run("I love this new feature!")
        assert result.data.is_positive
