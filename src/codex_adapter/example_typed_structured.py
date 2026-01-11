"""Example of using Pydantic types for structured responses with Codex."""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from codex_adapter import CodexClient


class FileInfo(BaseModel):
    """Information about a single file."""

    name: str = Field(description="File name")
    type: str = Field(description="File type/extension")
    size_kb: float | None = Field(default=None, description="File size in KB")


class FileListResponse(BaseModel):
    """Structured response for file listing."""

    files: list[FileInfo] = Field(description="List of files found")
    total_count: int = Field(description="Total number of files")
    summary: str = Field(description="Brief summary of findings")


class LanguageClassification(BaseModel):
    """Classification of project's primary language."""

    answer: Literal["python", "javascript", "rust", "go", "other"] = Field(
        description="Primary programming language"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence level in classification"
    )
    reasoning: str = Field(description="Explanation for the classification")


class CodebaseMetrics(BaseModel):
    """Metrics about a codebase."""

    files: int = Field(description="Number of files")
    lines_of_code: int = Field(description="Approximate lines of code")
    dependencies: int = Field(description="Number of dependencies")


class CodebaseAssessment(BaseModel):
    """Complete codebase complexity assessment."""

    complexity: Literal["simple", "moderate", "complex", "very_complex"] = Field(
        description="Overall complexity classification"
    )
    metrics: CodebaseMetrics = Field(description="Quantitative metrics")
    main_technologies: list[str] = Field(description="Primary technologies used")
    assessment: str = Field(description="Qualitative assessment text")


async def pydantic_file_listing() -> None:
    """Example: Use Pydantic model for file listing."""
    print("=" * 60)
    print("PYDANTIC TYPE EXAMPLE - File Listing")
    print("=" * 60)

    async with CodexClient() as client:
        thread = await client.thread_start(cwd=".")
        print(f"\n✓ Started thread: {thread.id}\n")

        print("Using Pydantic type: FileListResponse")
        print(f"Schema will be auto-generated from: {FileListResponse.__name__}\n")

        structured_content = ""
        async for event in client.turn_stream(
            thread.id,
            "List the Python files in the current directory. "
            "Return as JSON matching the provided schema.",
            output_schema=FileListResponse,  # Pass type directly!
        ):
            if event.event_type == "item/agentMessage/delta":
                delta = event.get_text_delta()
                structured_content += delta
                print(delta, end="", flush=True)

            elif event.event_type == "turn/completed":
                print("\n\n✓ Turn completed")
                break

        # Parse into Pydantic model
        if structured_content:
            try:
                # Parse directly into typed model
                response = FileListResponse.model_validate_json(structured_content)

                print("\n" + "=" * 60)
                print("PARSED PYDANTIC MODEL:")
                print("=" * 60)
                print(f"Total files: {response.total_count}")
                print(f"Summary: {response.summary}")
                print("\nFiles:")
                for file in response.files:
                    size = f"{file.size_kb}KB" if file.size_kb else "unknown"
                    print(f"  - {file.name} ({file.type}, {size})")

                # Access with full type safety!
                print(f"\n✓ Type-safe access: {response.files[0].name}")

            except (ValueError, TypeError) as e:
                print(f"\n✗ Failed to parse: {e}")
                print(f"Raw content:\n{structured_content}")


async def pydantic_classification() -> None:
    """Example: Use Pydantic model with Literal enums."""
    print("\n" + "=" * 60)
    print("PYDANTIC TYPE EXAMPLE - Language Classification")
    print("=" * 60)

    async with CodexClient() as client:
        thread = await client.thread_start(cwd=".")

        print("\nUsing Pydantic type: LanguageClassification")
        print("Schema enforces enum values via Literal types\n")

        response_text = ""
        async for event in client.turn_stream(
            thread.id,
            "Looking at the files in this directory, what programming language "
            "is this project primarily written in?",
            output_schema=LanguageClassification,  # Type with enums!
        ):
            if event.event_type == "item/agentMessage/delta":
                delta = event.get_text_delta()
                response_text += delta
                print(delta, end="", flush=True)
            elif event.event_type == "turn/completed":
                break

        if response_text:
            try:
                result = LanguageClassification.model_validate_json(response_text)
                print("\n\n✓ Structured response:")
                print(f"  Language: {result.answer}")
                print(f"  Confidence: {result.confidence}")
                print(f"  Reasoning: {result.reasoning}")

                # Type checker knows these are Literal types!
                if result.confidence == "high":
                    print("\n✓ High confidence classification!")

            except (ValueError, TypeError) as e:
                print(f"\n✗ Failed to parse: {e}")


async def pydantic_nested() -> None:
    """Example: Use nested Pydantic models."""
    print("\n" + "=" * 60)
    print("PYDANTIC TYPE EXAMPLE - Nested Models")
    print("=" * 60)

    async with CodexClient() as client:
        thread = await client.thread_start(cwd=".")

        print("\nUsing nested Pydantic type: CodebaseAssessment")
        print("Contains nested CodebaseMetrics model\n")

        response_text = ""
        async for event in client.turn_stream(
            thread.id,
            "Analyze this codebase and classify its complexity.",
            output_schema=CodebaseAssessment,  # Nested models!
        ):
            if event.event_type == "item/agentMessage/delta":
                delta = event.get_text_delta()
                response_text += delta
                print(delta, end="", flush=True)
            elif event.event_type == "turn/completed":
                break

        if response_text:
            try:
                assessment = CodebaseAssessment.model_validate_json(response_text)
                print("\n\n✓ Codebase Assessment:")
                print(f"  Complexity: {assessment.complexity}")
                print(f"  Files: {assessment.metrics.files}")  # Nested access!
                print(f"  Lines: {assessment.metrics.lines_of_code}")
                print(f"  Dependencies: {assessment.metrics.dependencies}")
                print(f"  Technologies: {', '.join(assessment.main_technologies)}")
                print(f"  Assessment: {assessment.assessment}")

                # Full type safety with autocomplete!
                if assessment.complexity in ["complex", "very_complex"]:
                    print(f"\n⚠ Complex codebase with {assessment.metrics.files} files")

            except (ValueError, TypeError) as e:
                print(f"\n✗ Failed to parse: {e}")


async def main() -> None:
    """Run all Pydantic type examples."""
    await pydantic_file_listing()
    await pydantic_classification()
    await pydantic_nested()

    print("\n" + "=" * 60)
    print("ALL PYDANTIC TYPE EXAMPLES COMPLETE")
    print("=" * 60)
    print("\n✓ Benefits of using Pydantic types:")
    print("  - Auto-generated JSON Schema from type annotations")
    print("  - Full type safety with mypy/pyright")
    print("  - IDE autocomplete for all fields")
    print("  - Validation with helpful error messages")
    print("  - No manual schema definition needed")


if __name__ == "__main__":
    asyncio.run(main())
