# MCP Sampling & Elicitation Example

This example demonstrates how to create and use a FastMCP server that combines **sampling** and **elicitation** in a single workflow.

## Overview

The example consists of:
- `server.py`: A compact MCP server with one comprehensive tool
- `demo.py`: Demo script showing agent interaction with the server

## The Code Fixer Tool

### `fix_code(code: str) -> str`
A single tool that demonstrates both MCP capabilities in one workflow:

1. **Sampling** (Server-side LLM): Analyzes the provided code for syntax errors, style issues, and improvements
2. **Elicitation** (Direct user interaction): Asks the user whether to proceed with fixing the identified issues  
3. **Sampling** (Server-side LLM): Generates the corrected code based on the analysis and user approval

**Input**: Code string (e.g., `print("hello world")` with typo)
**Output**: Analysis results and fixed code (if approved)

## Key Patterns

- **Server autonomy**: The server orchestrates a complex multi-step workflow internally
- **Direct user interaction**: Server asks user for decisions without going through the agent
- **Server-side intelligence**: Uses its own LLM for both analysis and code generation
- **Single tool interface**: Agent sees one simple tool, server handles complexity

## Running the Example

```bash
# Run the demo
uv run demo.py

# Or run server standalone
uv run server.py
```

The demo shows a complete workflow: code analysis → user confirmation → code fixing, all within one tool call.