---
title: AI Models
description: Language model setup and configuration
icon: material/cpu-64-bit
---

## Overview

AgentPool supports a wide range of model types thanks to `Pydantic-AI`. In the simplest form, models are defined by their "identifier", which is defined as `PROVIDER_NAME:MODEL_NAME` (example: `"openai:gpt-5-nano"`).

For more advanced scenarios, it is also possible to assign a more detailed model config including model settings like `temperature` etc.

In addition, some more experimental (meta-)Models are supported using [LLMling-models](https://github.com/phil65/LLMling-models).

These include models which let the user get into the role of an Agent, as well as fallback models and lot more.

```yaml
agents:
  my_agent:
    model: openai:gpt-5-nano  # simple model identifier
  my_agent2:
    model:  # extended model config
      provider: openai
      model: gpt-5-nano
      temperature: 0.5
```

## Configuration Reference

/// mknodes
{{ "llmling_models.configs.AnyModelConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///
