---
title: Expert Selection
description: Using pick() for expert selection
icon: material/account-group
hide:
  - toc
---

# Expert Selection

/// mknodes
{{ ['docs/examples/pick_experts/main.py', 'docs/examples/pick_experts/pick_teams.py', 'docs/examples/pick_experts/config.yml'] | pydantic_playground }}
///

with pick() and pick_multiple()












This example demonstrates LLMling-agent's type-safe selection methods:

- Using pick() for single expert selection
- Using pick_multiple() for team selection
- Type-safe results with reasoning
- Team-based agent selection


## How It Works

1. Single Selection (pick):
   - Takes a team of agents and a task description
   - Returns a single expert with reasoning
   - Result is type-safe: `Pick[Agent]`

2. Multiple Selection (pick_multiple):
   - Takes same inputs plus min/max picks
   - Returns multiple experts with reasoning
   - Result is type-safe: `MultiPick[Agent]`

Example Output:

This demonstrates:

- Type-safe agent selection
- Reasoned decision-making
- Team-based operations
- Flexible expert allocation
