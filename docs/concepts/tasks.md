# Agent Jobs

## What are Agent Jobs?

In LLMling, tasks are specifications of work that can be executed by agents.
Unlike other frameworks where tasks are tightly coupled with their executors,
LLMling treats tasks as independent definitions that specify requirements and provide resources.

A job defines:

- What needs to be done (prompt)
- What type of agent can do it (required dependencies)
- What result to expect (required return type)
- What tools are needed (equipment)
- What knowledge is required (context)

## The Job Concept

### Traditional Approach (Other Frameworks)

Most frameworks tightly couple tasks with their execution:

```python
agent = Agent()
job = Job(agent, **task_kwargs)
```

### LLMling-Agents's Approach

LLMling separates task definitions from execution:

```python
# Define what needs to be done and what's required
job = Job[AppConfig, AnalysisResult](
    prompt="Analyze this data",
    required_dependency=AppConfig,     # Agent must have these dependencies
    required_return_type=AnalysisResult,  # Agent must produce this type
    requires_vision=True,              # Agent requires vision (runtime-checked)
    tools=[                           # Job provides these tools
        "tools.analyzer",
        "tools.data_loader"
    ]
)

# Any compatible agent can execute the task
agent = pool.get_agent("analyzer", deps=AppConfig(), return_type=AnalysisResult)
result = await agent.run_job(task)  # Deps and return type verified using typing and runtime checks
```

This separation provides:

- Reusability: Same job can be executed by different agents
- Type Safety: Dependencies and results are validated
- Clear Contracts: Requirements are explicitly defined
- Resource Management: Tools and knowledge are job-specific

## Job Fields

```python
class Job[TDeps, TResult]:
    """Definition of work that can be executed by an agent."""

    prompt: str
    """The instruction/prompt for this task"""

    required_dependency: type[TDeps] | None
    """Type of dependencies an agent must have"""

    required_return_type: type[TResult]
    """Type that the agent must return"""

    requires_vision: bool = False
    """Whether the agent requires vision"""

    tools: list[ImportToolConfig | str]
    """Tools provided for this task"""

    knowledge: Knowledge | None
    """Knowledge sources for context"""

    description: str | None
    """Human-readable description"""

    min_context_tokens: int | None
    """Minimum required context size"""
```

## Job Execution

Jobs can be executed by any agent that meets their requirements:

```python
class Agent[TDeps, TResult]:
    async def run_job(
        self,
        job: Job[TDeps, TResult],
        *,
        store_history: bool = True,
        include_agent_tools: bool = True,
    ) -> ChatMessage[TResult]:
        """Execute a pre-defined job.

        1. Validates agent meets task requirements
        2. Loads task knowledge into context
        3. Sets up task-specific tools
        4. Executes task with strategy
        """
```

Jobs provide a clean way to define work requirements and manage resources while maintaining type safety throughout the execution chain.
This separation of concerns allows for better reusability and clearer contracts between task definitions and their executors.


## YAML Definition

Jobs can be defined directly in the agent manifest YAML file:

```yaml
jobs:
  analyze_code:
    prompt: "Analyze the code in the provided files"
    description: "Static code analysis task"

    # Job equipment
    tools:
      - import_path: "myapp.tools.code_analyzer"
        name: "analyze_code"
        description: "Run static analysis"
      - "myapp.tools.metrics"  # Short form

    # Knowledge sources
    knowledge:
      resources:
        - type: cli
          command: "mypy src/"
        - type: text
          content: "Analysis guidelines..."
```

Jobs can then be fetched from the pool and executed by any compatible agent:

```python
# Get job from pool
job = pool.get_job("analyze_code")

# Execute with compatible agent
agent = pool.get_agent("code_analyzer")
result = await agent.run_job(task)
```

Jobs defined in YAML are automatically registered with the pool's task registry during initialization. This allows for:

- Central task management
- Configuration-driven task definitions
- Easy sharing of tasks between agents
- Version control of task definitions
