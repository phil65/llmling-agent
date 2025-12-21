"""Integration tests for prompt commands."""

from __future__ import annotations

import pytest
from slashed import CommandStore

from agentpool import AgentsManifest
from agentpool_commands.prompts import ShowPromptCommand


TEST_CONFIG = """
prompts:
  system_prompts:
    greet:
      content: "Hello {{ name }}!"
      category: role

    analyze:
      content: |
        Analyzing {{ data }}...
        Please check {{ data }}
      category: methodology
"""


async def test_prompt_command():
    """Test prompt command with new prompt system."""
    messages: list[str] = []
    store = CommandStore()
    manifest = AgentsManifest.from_yaml(TEST_CONFIG)
    context = store.create_context(manifest, output_writer=messages.append)
    prompt_cmd = ShowPromptCommand()

    # Test simple prompt
    await prompt_cmd.execute(ctx=context, args=["greet?name=World"], kwargs={})
    assert "Hello World!" in messages[-1]

    # Test prompt with variables
    await prompt_cmd.execute(ctx=context, args=["analyze?data=test.txt"], kwargs={})
    assert "Analyzing test.txt" in messages[-1]
    assert "Please check test.txt" in messages[-1]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
