# OpenCode Tool Metadata Format

Tool results can include a `metadata` dict for rich UI display in OpenCode TUI.

## Write

```python
{
    "filePath": str,
    "content": str
}
```

## Edit

```python
{
    "diff": str,  # Unified diff format
    "filediff": {
        "file": str,
        "before": str,
        "after": str,
        "additions": int,
        "deletions": int
    }
}
```

## Read

```python
{
    "filePath": str,
    "content": str,
    "numLines": int,
    "startLine": int,
    "totalLines": int
}
```

## Bash

```python
{
    "output": str,
    "exit": int | None,
    "description": str
}
```

## TodoWrite

```python
{
    "todos": [
        {
            "id": str,
            "content": str,
            "status": str,  # "pending", "in_progress", "completed"
            "priority": str  # "high", "medium", "low"
        }
    ]
}
```
