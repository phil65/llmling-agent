---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/durable.py
title: Durable Execution Configuration
description: Workflow orchestration and durable execution
icon: material/repeat
---

# Durable Execution Configuration

Durable execution provides workflow orchestration with automatic retries, state persistence, and recovery capabilities. This allows agents to handle long-running tasks reliably.

## Overview

LLMling-Agent integrates with popular workflow orchestration platforms:

- **Temporal**: Industry-standard workflow engine with strong durability guarantees
- **Prefect**: Modern workflow orchestration with excellent observability
- **DBOS**: Database-oriented workflow system with automatic checkpointing

These integrations wrap agent execution in durable workflows, ensuring that agent operations can survive failures and be resumed from checkpoints.

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.durable.DurableExecutionConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Use Cases

- **Long-running tasks**: Agents that process large datasets or perform extended operations
- **Critical workflows**: Operations that must complete despite transient failures
- **Distributed systems**: Coordinating multiple agents across different services
- **Audit requirements**: Systems that need complete execution history and replay capability
- **Rate-limited APIs**: Automatic retry handling with exponential backoff

## Benefits

- **Automatic retries**: Failed operations are automatically retried with configurable policies
- **State persistence**: Agent state is preserved across failures and restarts
- **Recovery**: Workflows can be resumed from the last successful checkpoint
- **Observability**: Built-in monitoring and tracking of execution progress
- **Scalability**: Distribute agent workload across multiple workers