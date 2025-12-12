---
title: Response Types
description: Structured response type configuration
icon: material/reply
---

# Response Types

Response types define structured output formats for agents. They can be defined directly in YAML or imported from Python code.

!!! tip "Type Safety"
    While YAML configuration is convenient, defining response types as Pydantic models in Python code provides better type safety, IDE support, and reusability:
    ```python
    from pydantic import BaseModel

    class AnalysisResult(BaseModel):
        success: bool
        issues: list[str]
        severity: str
    ```

## Response Configuration

Each response definition includes:

```yaml
responses:
  MyResponse:
    response_schema:  # Schema definition (required)
      type: "inline"  # or "import"
      # schema details...
    description: "Optional description of the response"
    result_tool_name: "final_result"  # Tool name for result creation
    result_tool_description: "Create the final result"  # Tool description
    output_retries: 3  # Number of validation retries
```

## Inline Responses
Define response structure directly in YAML:

```yaml
--8<-- "docs/configuration/responses.yml"
```

## Imported Responses
Import response types from Python code:

### Python Type Import
```yaml
responses:
  AdvancedAnalysis:
    response_schema:
      type: "import"
      import_path: "myapp.types:AnalysisResult"
  MetricsResult:
    response_schema:
      type: "import"
      import_path: "myapp.analysis:MetricsResponse"
```

### Using Response Types

### Assign to Agent
```yaml
agents:
  analyzer:
    model: "openai:gpt-5"
    output_type: "CodeAnalysis"  # Reference response by name
```

### Inline with Custom Tool Name
```yaml
agents:
  processor:
    output_type:
      response_schema:
        type: "inline"  # Direct inline definition
        fields:
          success:
            type: "bool"
          details:
            type: "str"
      result_tool_name: "create_result"  # Custom tool name
      result_tool_description: "Create the final analysis result"
      output_retries: 2  # Number of validation attempts
```

## Available Field Types
- `str`: Text strings
- `int`: Integer numbers
- `float`: Floating point numbers
- `bool`: Boolean values
- `list[type]`: Lists of values (e.g., `list[str]`, `list[int]`)
- `dict[key_type, value_type]`: Dictionaries
- `datetime`: Date and time values
- Custom types through imports
