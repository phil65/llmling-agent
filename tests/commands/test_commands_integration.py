"""Integration tests for prompt commands."""

from __future__ import annotations

from llmling import Config, PromptMessage, PromptParameter, StaticPrompt
import pytest
from slashed import CommandStore, DefaultOutputWriter

from llmling_agent import AgentsManifest
from llmling_agent_commands.prompts import ShowPromptCommand


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


@pytest.fixture
def config() -> Config:
    """Create a Config with test prompts."""
    # Create prompt messages
    greet_message = PromptMessage(role="system", content="Hello {name}!")

    analyze_messages = [
        PromptMessage(role="system", content="Analyzing {data}..."),
        PromptMessage(role="user", content="Please check {data}"),
    ]

    # Create prompt parameters
    name_param = PromptParameter(
        name="name", description="Name to greet", default="World"
    )

    data_param = PromptParameter(
        name="data", description="Data to analyze", required=True
    )

    # Create prompt instances
    greet_prompt = StaticPrompt(
        name="greet",
        description="Simple greeting prompt",
        messages=[greet_message],
        arguments=[name_param],
    )

    analyze_prompt = StaticPrompt(
        name="analyze",
        description="Analysis prompt",
        messages=analyze_messages,
        arguments=[data_param],
    )

    return Config(prompts={"greet": greet_prompt, "analyze": analyze_prompt})


async def test_prompt_command():
    """Test prompt command with new prompt system."""
    messages = []
    store = CommandStore()

    class TestOutput(DefaultOutputWriter):
        async def print(self, message: str):
            messages.append(message)

    # Load test config
    manifest = AgentsManifest.from_yaml(TEST_CONFIG)
    context = store.create_context(manifest, output_writer=TestOutput())
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
