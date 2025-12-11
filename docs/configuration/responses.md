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

### Simple Status Response
```yaml
responses:
  StatusResponse:
    response_schema:
      type: "inline"
      description: "Simple operation result with status"
      fields:
        success:
          type: "bool"
          description: "Whether operation succeeded"
        message:
          type: "str"
          description: "Status message or error details"
```

### Analysis Result
```yaml
responses:
  CodeAnalysis:
    response_schema:
      type: "inline"
      description: "Code analysis results with issues"
      fields:
        issues:
          type: "list[str]"
          description: "List of found issues"
        severity:
          type: "str"
          description: "Overall severity level"
        locations:
          type: "list[str]"
          description: "Source code locations"
    result_tool_name: "create_analysis"
    result_tool_description: "Create code analysis result"
```

### Complex Response
```yaml
responses:
  DataProcessingResult:
    response_schema:
      type: "inline"
      description: "Complex data processing result"
      fields:
        success:
          type: "bool"
          description: "Operation success"
        records_processed:
          type: "int"
          description: "Number of processed records"
        errors:
          type: "list[str]"
          description: "List of errors if any"
        metrics:
          type: "dict[str, float]"
          description: "Processing metrics"
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
```

### Package Response Type
```yaml
responses:
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

## Schema Definition

The response schema defines the structure of the return value:

```yaml
response_schema:
  type: "inline"  # Define structure inline
  description: "Result description"
  fields:
    # Field definitions

# OR

response_schema:
  type: "import"  # Import from Python
  import_path: "myapp.types:MyOutputType"
```
