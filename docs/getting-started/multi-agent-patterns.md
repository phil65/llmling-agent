---
title: Multi-Agent Patterns
description: Orchestration patterns for multiple agents
icon: material/sitemap
---

# Multi-Agent Orchestration Patterns

LLMling-agent offers several patterns for coordinating multiple agents. Each pattern has its strengths and is suited for different use cases.

## Quick Reference

| Pattern | Use Case | Complexity | Type Safety | Visualization | Control Flow |
|---------|----------|------------|-------------|---------------|--------------|
| Direct Connections | Simple A → B flows | Low | ✅ | ❌ | Automatic |
| Controlled Interactions | Interactive/supervised flows | Low | ✅ | ❌ | Manual/Interactive |
| Decision Making (pick) | Agent selection/routing | Low | ✅ | ❌ | Agent-driven |
| Agent as Tool | Hierarchical/expert | Low | ✅ | ❌ | Parent-driven |
| Teams | Parallel/group ops | Medium | ✅ | ❌ | Coordinated |
| Decorator Pattern | Testing/scripted | Medium | ✅ | ❌ | Programmatic |
| Pydantic Graph | Complex workflows | High | ✅ | ✅ | Graph-based |


## 1. Direct Connections (Simple Forwarding)

Best for: Simple linear flows between agents

```python
analyzer = Agent("analyzer")
planner = Agent("planner")
executor = Agent("executor")

# Chain connections
analyzer >> planner >> executor

# Or with configuration
analyzer.connect_to(planner, connection_type="run")
```

## 2. Agent as Tool (Hierarchical)

Best for: Expert consultation patterns where one agent calls others as needed

```python
main_agent = Agent("coordinator")
expert = Agent("expert")

# Register expert as tool
main_agent.register_worker(
    expert,
    name="consult_expert",
    reset_history_on_run=True
)
```

## 3. Teams (Group Operations)

Best for: Parallel execution or group coordination

```python
team = Team([agent1, agent2, agent3])
result = await team.run_parallel("Analyze this data")

# Or chain through team
team.chain("Process this sequentially")
```

## 4. Decorator Pattern (Testing/Scripting)

Best for: Scripted interactions and testing flows

```python
@with_nodes("analyzer", "planner")
async def analysis_flow(analyzer: Agent, planner: Agent):
    result = await analyzer.run("Analyze this")
    return await planner.run(result.content)
```

## 5. Pydantic Graph (Complex Workflows)

Best for: Complex workflows with multiple paths and state management

```python
@dataclass
class AnalyzeNode(BaseNode[WorkflowState]):
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> PlanNode | EndNode:
        result = await ctx.deps.analyzer.run("Analyze")
        if result.content.needs_planning:
            return PlanNode()
        return EndNode()

workflow = Graph(nodes=[AnalyzeNode, PlanNode, EndNode])
```


## 7. Agent Selection (pick)

Best for: Letting agents make informed choices about which other agents to work with. The agent analyzes each potential colleague's capabilities, description, and specialties to make an optimal choice.

```python
from pydantic import BaseModel

# The selection is automatically structured
class Pick[T](BaseModel):
    selection: T
    reason: str

# Let agent pick from team
team = Team([researcher, analyst, writer])
result = await coordinator.talk.pick(
    selections=team,
    task="Choose who should handle this technical documentation task"
)
print(f"Selected {result.selection.name} because {result.reason}")

# Pick from pool
result = await coordinator.talk.pick(
    selections=pool.agents,
    task="Choose a specialist for this task"
)

# Pick multiple agents for collaboration
result = await coordinator.talk.pick_multiple(
    selections=team.agents,
    task="Select a research team",
    min_picks=2,
    max_picks=3
)
```

### How it Works

1. **Agent Bios**: Each agent's capabilities and specialties are automatically formatted into a "bio":
   ```text
   Agent: researcher
   Description: Expert at gathering and analyzing technical information
   Tools: web_search, document_analysis, fact_checking
   System Context: Trained to focus on accuracy and thoroughness
   ```

2. **Structured Decision**: The selecting agent:
   - Receives formatted bios of all options
   - Analyzes the task requirements
   - Makes a structured selection with reasoning
   - Returns a type-safe Pick[Agent] result

3. **Integration with Other Patterns**:
```python
# In Graph Nodes
@dataclass
class AssignTask(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> ExecuteNode:
        result = await ctx.deps.coordinator.talk.pick(
            selections=ctx.deps.team.agents,
            task=f"Choose who should handle: {ctx.state.task}"
        )
        ctx.state.assigned_agent = result.selection
        return ExecuteNode()

# With Teams
team = Team([analyst, planner, executor])
expert = await coordinator.talk.pick(
    selections=team.agents,
    task="Who should lead this analysis?"
)
await team.set_leader(expert.selection)
```


## Choosing the Right Pattern

- **Simple Linear Flows**: Use direct connections
- **Expert Consultation**: Use agent as tool
- **Parallel Processing**: Use teams
- **Testing/Scripting**: Use decorator pattern
- **Complex Workflows**: Use pydantic graph

Multiple patterns can be combined as needed. For example:
- Use teams with graph nodes
- Combine connections with worker tools
- Mix decorators with direct connections
