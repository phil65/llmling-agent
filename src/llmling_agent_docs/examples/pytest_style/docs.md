# PyTest-Style Agent Functions

This example demonstrates a pytest-inspired way to work with agents:

- Using agents as function decorators
- Automatic function discovery
- Dependency injection
- Execution order control
- Function result handling


## How It Works

1. Functions are decorated with `@node_function`
2. Type hints specify which agent to inject (`analyzer: Agent`)
3. Dependencies are declared in the decorator (`depends_on="analyze_data"`)
4. Results from one function can be injected into another
5. All functions are discovered and executed in the correct order

Key Features:

- Automatic agent injection based on type hints
- Function dependency resolution
- Parallel execution where possible
- Results passed automatically between functions

This provides a clean, declarative way to orchestrate multi-agent workflows, similar to how pytest fixtures work.
