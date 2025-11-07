# Multi-Agent Documentation System

This example demonstrates a team of three agents working together to scan, document, and validate Python code. It shows different patterns of agent collaboration:

- Async delegation (fire and forget)
- Tool usage with waiting
- Chained tool calls


## How It Works

1. The File Scanner agent scans directories and identifies Python files
2. It passes these files to the Documentation Writer agent asynchronously
3. In parallel, it uses the Error Checker as a tool to validate the files
4. The Documentation Writer processes files as they come in
5. Results are printed to the console as they become available

This demonstrates different ways agents can collaborate:

- Async message passing (scanner to writer)
- Synchronous tool usage (scanner using checker)
- Event-based output handling (writer to console)
