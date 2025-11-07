# Multi-Agent Download System with Cheerleader

This example demonstrates several advanced features of LLMling-agent:

- Continuous repetitive tasks
- Async parallel execution of LLM calls
- YAML configuration with storage providers
- Capability usage (agent listing and task delegation)
- Stateful callback mechanism
- Multiple storage providers (SQLite + pretty-printed logs)


## How It Works

1. We set up a team of downloaders and a cheerleading fan
2. The fan runs continuously in the background, getting updates via callbacks
3. We test downloads in different modes:
   - Sequential (one after another)
   - Parallel (both at once)
   - Overseer-coordinated (using agent capabilities)
4. The fan cheers appropriately for each situation
5. All interactions are logged to both SQLite and pretty-printed text files

This demonstrates:

- Background tasks with continuous prompts
- Agent cloning
- Team operations (sequential vs parallel)
- Capability-based delegation
- Multi-provider storage
