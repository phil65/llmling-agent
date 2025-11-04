from __future__ import annotations

import pytest

from llmling_agent import Agent
from llmling_agent.models.content import ImageURLContent


@pytest.mark.flaky(reruns=3)
async def test_vision(vision_model: str):
    """Test basic vision capability with a small, public image."""
    agent = Agent(name="test-vision", model=vision_model)
    # Using a small, public image
    msg = "https://python.org/static/community_logos/python-logo-master-v3-TM.png"
    image = ImageURLContent(url=msg, description="Python logo")
    msg = "What does this image show? Answer in one short sentence."
    result = await agent.run(msg, image)

    assert isinstance(result.content, str)
    assert "Python" in result.content
    assert len(result.content) < 120  # noqa: PLR2004


if __name__ == "__main__":
    pytest.main([__file__])
