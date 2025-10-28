# Installation

## Basic Installation

Using `pip`:
```bash
pip install llmling-agent
```

Using `uv` (when using in no-code form via CLI and YAMLs)
```bash
uv tool install llmling-agent
```

## Optional Features

LLMling-agent provides several optional features that can be installed with extras:


- **Clipboard Support**:
  ```bash
  pip install llmling-agent[clipboard]
  ```

- **Markdown Processing**:
  ```bash
  pip install llmling-agent[markdown]
  ```


Multiple extras can be combined:
```bash
pip install llmling-agent[ui,markdown]
```

## Requirements

- Python 3.12 or higher
- Core dependencies:
  - llmling-models
  - llmling
  - pydantic-ai
  - promptantic
  - pydantic
  - Other utilities (slashed, watchfiles, tiktoken, etc.)

## Development Installation

For development work, install with additional dependencies:
```bash
pip install llmling-agent[dev,lint,docs]
```

This includes:
- Testing tools (pytest, pytest-cov)
- Linting tools (ruff, mypy)
- Documentation tools (mkdocs)
```
