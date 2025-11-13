# /// script
# dependencies = ["llmling-agent"]
# ///

"""Example of structured responses defined both in code and YAML."""

import os

from schemez import Schema

from llmling_agent import Agent, AgentPool, AgentsManifest
from llmling_agent_docs.examples.utils import get_config_path, is_pyodide, run


# set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your_api_key_here")


class PythonResult(Schema):
    """Structured response defined in Python."""

    main_point: str
    is_positive: bool


async def run_example():
    """Show both ways of defining structured responses."""
    # Example 1: Python-defined structure
    agent = Agent(
        model="openai:gpt-5-mini",
        system_prompt="Summarize text in a structured way.",
        output_type=PythonResult,
    )
    async with agent as summarizer:
        result = await summarizer.run("I love this new feature!")
        summary = result.data
        print("\nPython-defined Response:")
        print(f"Main point: {summary.main_point}")
        print(f"Is positive: {summary.is_positive}")

    # Example 2: YAML-defined structure
    # NOTE: this is not recommended for programmatic usage and is just a demo. Use this
    # only for complete YAML workflows, otherwise your linter wont like what you are doin.
    config_path = get_config_path(None if is_pyodide() else __file__)
    manifest = AgentsManifest.from_file(config_path)
    async with AgentPool(manifest) as pool:
        analyzer = pool.get_agent("analyzer")
        result_2 = await analyzer.run("I'm really excited about this project!")
        analysis = result_2.data
        print("\nYAML-defined Response:")
        # Type checkers cant deal with dynamically generated Models, so we have to
        # git-ignore
        print(f"Sentiment: {analysis.sentiment}")  # type: ignore
        print(f"Confidence: {analysis.confidence:.2f}")  # type: ignore
        print(f"Mood: {analysis.mood}")  # type: ignore


if __name__ == "__main__":
    run(run_example())
