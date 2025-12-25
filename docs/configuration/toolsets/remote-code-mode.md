---
title: Remote Code Mode Toolset
description: Wrap toolsets for remote code-based interaction
icon: material/cloud-outline
---

# Remote Code Mode Toolset

Similar to Code Mode, but executes code in a remote environment.

## Basic Usage

```yaml
agents:
  remote_coder:
    toolsets:
      - type: remote_code_mode
        toolsets:
          - type: file_access
            fs: "file:///workspace"
```

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.RemoteCodeModeToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets", header_level=3) }}
///
