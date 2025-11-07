# Expert Selection with pick() and pick_multiple()

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
```
Selected: database_expert
Reason: The task specifically involves SQL query optimization, which is the database expert's primary specialty.

Selected: frontend_dev, security_expert
Reason: Creating a secure login page requires both frontend expertise for the user interface and security expertise for proper authentication implementation.
```

This demonstrates:

- Type-safe agent selection
- Reasoned decision-making
- Team-based operations
- Flexible expert allocation
