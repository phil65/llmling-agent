---
title: System Prompts & Prompt Library
description: Prompt library configuration
icon: material/book
---

# Prompt Library Configuration

## Overview
LLMling's prompt library allows defining reusable prompts that can be shared across agents.
Prompts are defined in the `prompts` section of your configuration and can be referenced by name.

## Basic Structure

```yaml
prompts:
  # Main system prompts for defining agent behavior
  system_prompts:
    expert_analyst:
      content: |
        You are an expert data analyst.
        Focus on finding patterns and insights.
      category: role

    step_by_step:
      content: |
        Break tasks into sequential steps.
        Explain each step thoroughly.
      category: methodology

# Using prompts in agents
agents:
  analyst:
    system_prompts:
      # Direct string prompts
      - "You help with analysis."

      # Reference library prompts
      - type: library
        reference: expert_analyst
      - type: library
        reference: step_by_step
```

## Prompt Categories

### System Prompts
Define core agent behaviors and methodologies:

```yaml
prompts:
  system_prompts:
    # Role definition
    technical_writer:
      content: |
        You are an expert technical writer.
        Focus on clarity and precision.
        Use proper terminology consistently.
      category: role

    # Methodology definition
    step_by_step:
      content: |
        Follow these steps for each task:
        1. Understand requirements
        2. Plan approach
        3. Execute systematically
        4. Verify results
      category: methodology

    # Tone/style definition
    professional:
      content: |
        Maintain formal, business-appropriate language.
        Be concise but thorough.
      category: tone
```

## Prompt Types

The `type` field categorizes prompts by their purpose:

### Role Types
Define WHO the agent is:
```yaml
system_prompts:
  expert_dev:
    category: role
    content: |
      You are a senior software developer with expertise in:
      - System architecture design
      - Code quality assessment
      - Performance optimization

  data_scientist:
    category: role
    content: |
      You specialize in data analysis and machine learning.
      Your expertise includes statistical modeling and data visualization.
```

### Methodology Types
Define HOW the agent works:
```yaml
system_prompts:
  analytical:
    category: methodology
    content: |
      Approach problems systematically:
      1. Gather and analyze data
      2. Identify patterns and trends
      3. Form evidence-based conclusions
      4. Present findings clearly

  iterative:
    category: methodology
    content: |
      Work in small iterations:
      - Start with minimal viable approach
      - Test and validate results
      - Refine based on feedback
      - Scale successful patterns
```

### Tone Types
Define communication STYLE:
```yaml
system_prompts:
  formal:
    category: tone
    content: |
      Use professional, business-appropriate language:
      - Maintain formal tone
      - Be precise and clear
      - Avoid colloquialisms

  friendly:
    category: tone
    content: |
      Be approachable and helpful:
      - Use warm, welcoming language
      - Show empathy and understanding
      - Encourage questions and dialogue
```

### Format Types
Define output STRUCTURE:
```yaml
system_prompts:
  markdown:
    category: format
    content: |
      Format responses using Markdown:
      - Use headers for sections (# ## ###)
      - Use bullet points for lists
      - Use code blocks for code examples
      - Use tables for structured data

  structured:
    category: format
    content: |
      Structure responses with:
      1. **Summary** - Brief overview
      2. **Details** - Comprehensive explanation
      3. **Examples** - Practical illustrations
      4. **Next Steps** - Recommended actions
```

## Using Library Prompts

Reference prompts in agent configuration:

```yaml
agents:
  technical_assistant:
    model: gpt-4
    system_prompts:
      # Direct string prompts
      - "You are a technical assistant."
      - "Focus on helping with code and documentation."

      # Library reference prompts
      - type: library
        reference: technical_writer    # Role
      - type: library
        reference: step_by_step       # Methodology
      - type: library
        reference: professional       # Tone
      - type: library
        reference: markdown          # Format

  data_analyst:
    system_prompts:
      - type: library
        reference: expert_analyst
      - type: library
        reference: analytical
      - type: library
        reference: formal
```

## Advanced Prompt Management

### Dynamic Prompt Generation

Use functions to generate context-aware prompts:

```yaml
prompts:
  system_prompts:
    context_aware:
      category: role
      content: |
        You adapt to user context and preferences.

agents:
  adaptive_agent:
    system_prompts:
      - type: library
        reference: context_aware
      - type: function
        function: "prompts:generate_user_context"
        arguments:
          user_type: "developer"
          experience_level: "intermediate"
```

## Complete Example

```yaml
prompts:
  system_prompts:
    # Roles
    technical_expert:
      category: role
      content: |
        You are a technical expert specializing in:
        - Software development best practices
        - System architecture and design
        - Code review and quality assurance

    code_reviewer:
      category: role
      content: |
        You are an experienced code reviewer focused on:
        - Code quality and maintainability
        - Security best practices
        - Performance optimization

    # Methodologies
    systematic:
      category: methodology
      content: |
        Follow this systematic approach:
        1. Understand requirements fully
        2. Break down complex problems
        3. Apply best practices consistently
        4. Validate results thoroughly

    # Tones
    professional:
      category: tone
      content: |
        Maintain professional communication:
        - Use formal, precise language
        - Be respectful and constructive
        - Provide clear explanations

    # Formats
    structured:
      category: format
      content: |
        Structure responses with:
        1. **Overview** - Brief summary
        2. **Analysis** - Detailed examination
        3. **Recommendations** - Actionable advice
        4. **Examples** - Practical illustrations

agents:
  senior_dev:
    model: gpt-4
    description: "Senior developer specialized in code review and optimization"
    system_prompts:
      - "Specialize in Python and TypeScript development."
      - type: library
        reference: technical_expert
      - type: library
        reference: code_reviewer
      - type: library
        reference: systematic
      - type: library
        reference: professional
      - type: library
        reference: structured

  mentor:
    model: gpt-4
    description: "Programming teacher and mentor"
    system_prompts:
      - type: library
        reference: technical_expert
      - type: file
        path: "prompts/teaching_style.j2"
        variables:
          approach: "socratic"
          patience_level: "high"
      - type: function
        function: "education:generate_learning_context"
        arguments:
          student_level: "beginner"
          topic: "programming_fundamentals"
```

## Organization Best Practices

### File Structure
Keep prompts organized in separate files:

```yaml
# prompts/roles.yml
prompts:
  system_prompts:
    technical_expert:
      category: role
      content: ...

# prompts/styles.yml
prompts:
  system_prompts:
    professional:
      category: tone
      content: ...

# agents.yml
INHERIT:
  - prompts/roles.yml
  - prompts/styles.yml

agents:
  my_agent:
    system_prompts:
      - type: library
        reference: technical_expert
      - type: library
        reference: professional
```

### Naming Conventions
Use clear, descriptive names:

- **Roles**: `expert_analyst`, `code_reviewer`, `technical_writer`
- **Methodologies**: `step_by_step`, `analytical`, `iterative`
- **Tones**: `professional`, `friendly`, `formal`, `casual`
- **Formats**: `markdown`, `structured`, `bullet_points`

### Documentation
Document your prompts:

```yaml
prompts:
  system_prompts:
    expert_analyst:
      category: role
      content: |
        You are an expert data analyst with 10+ years experience.

        Core competencies:
        - Statistical analysis and modeling
        - Data visualization and reporting
        - Business intelligence and insights
      # Internal documentation (not sent to agent)
      description: "Primary role for data analysis agents"
      tags: ["data", "analysis", "expert"]
      version: "1.2"
```

## System Prompts Format

The system prompts format allows you to mix different prompt types:

```yaml
agents:
  my_agent:
    system_prompts:
      - "Direct prompt"
      - type: library
        reference: expert_role
      - type: library
        reference: professional_tone
```

This format provides:
- **Type safety**: Clear discrimination between prompt types
- **Flexibility**: Mix different prompt sources in one list
- **Extensibility**: Support for multiple prompt types
- **Consistency**: Unified approach across all prompt types
