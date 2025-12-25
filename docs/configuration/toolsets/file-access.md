---
title: File Access Toolset
description: Read, write, and edit files on any filesystem
icon: material/file-document
---

# File Access Toolset

The File Access toolset provides tools for reading, writing, and editing files on any fsspec-compatible filesystem. This includes local files, S3, GitHub repositories, and more.

## Basic Usage

```yaml
agents:
  my_agent:
    toolsets:
      - type: file_access
        fs: "file:///workspace"
```

## Filesystem Options

The `fs` field accepts either a URI string or a full filesystem configuration:

### URI String

```yaml
toolsets:
  - type: file_access
    fs: "file:///home/user/project"
```

### Filesystem Config

```yaml
toolsets:
  - type: file_access
    fs:
      type: github
      org: sveltejs
      repo: svelte
      sha: main
```

### Composed Filesystems

Mount multiple filesystems together using the `mounts` type:

```yaml
toolsets:
  - type: file_access
    fs:
      type: mounts
      mounts:
        docs: "github://sveltejs:svelte@main"
        src: "file:///workspace/src"
        data:
          type: s3
          bucket: my-bucket
```

## Available Tools

```python exec="true"
from agentpool_toolsets.fsspec_toolset import FSSpecTools
from agentpool.docs.utils import generate_tool_docs

toolset = FSSpecTools()
print(generate_tool_docs(toolset))
```

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.FSSpecToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///

## Examples

### Local Development

```yaml
toolsets:
  - type: file_access
    fs: "file:///home/user/project"
    max_file_size_kb: 128
```

### GitHub Repository Access

```yaml
toolsets:
  - type: file_access
    fs:
      type: github
      org: fastapi
      repo: fastapi
      cached: true
```

### Multi-Source Documentation

```yaml
toolsets:
  - type: file_access
    fs:
      type: mounts
      mounts:
        svelte: "github://sveltejs:svelte@main"
        react: "github://facebook:react@main"
        local: "file:///docs"
```
