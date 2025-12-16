---
title: VFS Toolset
description: Access manifest-defined resources
icon: material/folder-network
---

# VFS Toolset

The VFS (Virtual File System) toolset provides access to resources defined in your manifest's `resources` section. It offers a unified interface to query and read from multiple configured filesystems.

## Basic Usage

First, define resources in your manifest:

```yaml
resources:
  docs:
    type: uri
    uri: "memory://"
  data: "s3://my-bucket/data"
  code: "github://myorg:myrepo@main"
```

Then enable the VFS toolset:

```yaml
agents:
  my_agent:
    toolsets:
      - type: vfs
```

## Available Tools

| Tool | Description |
|------|-------------|
| `vfs_list` | List contents of a resource path |
| `vfs_read` | Read content from a resource |
| `vfs_info` | Get information about configured resources |

## Usage Examples

Once configured, agents can access resources like:

```
# List all available resources
vfs_list()

# List files in a specific resource
vfs_list("docs")

# Read a file
vfs_read("docs/guide.md")

# Read a directory recursively
vfs_read("code/src", recursive=True)
```

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.toolsets.VFSToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///

## When to Use VFS vs File Access

| Use Case | Recommended |
|----------|-------------|
| Shared resources across agents | VFS |
| Agent-specific filesystem | File Access |
| Simple resource access | VFS |
| Full file editing capabilities | File Access |
| Composed/mounted filesystems | File Access with `mounts` |
