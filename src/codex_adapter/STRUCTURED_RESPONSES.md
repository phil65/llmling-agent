# Structured Responses

The Codex adapter supports structured responses using JSON Schema to constrain the agent's output. This ensures the response matches a specific format and can be parsed into typed objects.

## Method Comparison

**Two methods available:**

1. **`turn_stream_structured()`** - High-level, typed API (recommended)
   - Accepts: Pydantic type only
   - Returns: Typed Pydantic model instance
   - Automatically handles schema generation and parsing
   - Best for: When you want type safety and convenience

2. **`turn_stream()`** - Low-level streaming API
   - Accepts: Dict schema OR Pydantic type
   - Returns: Event stream (you manually collect and parse)
   - Gives you full control over event processing
   - Best for: When you need to process events individually or use dict schemas

## Recommended Approach: PydanticAI-like API

The simplest way to get structured responses is using `turn_stream_structured()`:

```python
from codex_adapter import CodexClient
from pydantic import BaseModel

class FileInfo(BaseModel):
    name: str
    type: str

class FileList(BaseModel):
    files: list[FileInfo]
    total: int

async with CodexClient() as client:
    thread = await client.thread_start(cwd=".")

    # One method call - automatic schema generation and parsing!
    result = await client.turn_stream_structured(
        thread.id,
        "List Python files in current directory",
        FileList,  # Pass the Pydantic type
    )

    # Typed result with full IDE autocomplete!
    print(f"Found {result.total} files: {result.files}")
```

**Benefits:**
- Single method call (no manual event streaming or parsing)
- Automatic JSON Schema generation from Pydantic type
- Automatic response parsing and validation
- Full type safety and IDE autocomplete
- Similar developer experience to PydanticAI

## Alternative Approaches

If you need more control over event streaming, you can use the lower-level APIs:

### Alternative 1: Manual Streaming with Pydantic Types

### 1. Raw JSON Schema (Dict)

Pass a dictionary containing a JSON Schema:

```python
from codex_adapter import CodexClient
import json

schema = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "size_kb": {"type": "number"},
                },
                "required": ["name", "type"],
            },
        },
        "total_count": {"type": "number"},
        "summary": {"type": "string"},
    },
    "required": ["files", "total_count", "summary"],
}

async with CodexClient() as client:
    thread = await client.thread_start(cwd=".")

    structured_content = ""
    async for event in client.turn_stream(
        thread.id,
        "List the Python files in the current directory.",
        output_schema=schema,  # Pass dict
    ):
        if event.event_type == "item/agentMessage/delta":
            structured_content += event.get_text_delta()
        elif event.event_type == "turn/completed":
            break

    # Parse JSON response
    data = json.loads(structured_content)
    print(f"Found {data['total_count']} files")
```

### 2. Pydantic Types (Recommended)

Pass a Pydantic model class directly - the schema is auto-generated:

```python
from codex_adapter import CodexClient
from pydantic import BaseModel, Field
from typing import Literal

class FileInfo(BaseModel):
    """Information about a single file."""
    name: str
    type: str
    size_kb: float | None = None

class FileListResponse(BaseModel):
    """Structured response for file listing."""
    files: list[FileInfo]
    total_count: int
    summary: str

async with CodexClient() as client:
    thread = await client.thread_start(cwd=".")

    response_text = ""
    async for event in client.turn_stream(
        thread.id,
        "List the Python files in the current directory.",
        output_schema=FileListResponse,  # Pass Pydantic type!
    ):
        if event.event_type == "item/agentMessage/delta":
            response_text += event.get_text_delta()
        elif event.event_type == "turn/completed":
            break

    # Parse into typed Pydantic model
    response = FileListResponse.model_validate_json(response_text)

    # Full type safety and IDE autocomplete!
    print(f"Found {response.total_count} files")
    for file in response.files:
        print(f"  - {file.name} ({file.type})")
```

## Benefits of Pydantic Types

1. **Auto-generated Schema**: No manual JSON Schema definition needed
2. **Type Safety**: Full mypy/pyright type checking
3. **IDE Support**: Autocomplete for all fields
4. **Validation**: Pydantic validates the response with helpful error messages
5. **Documentation**: Field descriptions become schema descriptions
6. **Enums**: Use `Literal` types for enumerated values

## Advanced Example: Enums and Nested Models

```python
from pydantic import BaseModel, Field
from typing import Literal

class CodebaseMetrics(BaseModel):
    files: int
    lines_of_code: int
    dependencies: int

class CodebaseAssessment(BaseModel):
    complexity: Literal["simple", "moderate", "complex", "very_complex"]
    metrics: CodebaseMetrics  # Nested model
    main_technologies: list[str]
    assessment: str

async with CodexClient() as client:
    thread = await client.thread_start(cwd=".")

    response_text = ""
    async for event in client.turn_stream(
        thread.id,
        "Analyze this codebase and classify its complexity.",
        output_schema=CodebaseAssessment,
    ):
        if event.event_type == "item/agentMessage/delta":
            response_text += event.get_text_delta()
        elif event.event_type == "turn/completed":
            break

    assessment = CodebaseAssessment.model_validate_json(response_text)

    # Type checker knows complexity is one of 4 Literal values
    if assessment.complexity in ["complex", "very_complex"]:
        print(f"⚠ Complex codebase with {assessment.metrics.files} files")
```

## How It Works

1. **With dict**: The schema is passed directly to the `turn/start` request as `outputSchema`
2. **With Pydantic type**:
   - `TypeAdapter` extracts the JSON Schema from the Pydantic model
   - The schema is passed to `turn/start` as `outputSchema`
   - The agent's response is constrained to match this schema
   - You parse the response using `model_validate_json()` for type safety

## Examples

See complete working examples:
- `example_structured.py` - Raw dict schemas
- `example_typed_structured.py` - Pydantic types (recommended)
