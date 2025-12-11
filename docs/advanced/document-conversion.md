---
title: Document Conversion
description: Document conversion features
icon: material/file-document
---

# Document Conversion

LLMling provides document conversion capabilities to transform various document formats (PDFs, Office documents, HTML, etc.)
into markdown text that can be processed by language models.

## Overview

The conversion system:

- Supports multiple document formats through different converters
- Can handle both files and raw content
- Uses async I/O for file operations
- Includes a fallback plain text converter

## Usage

### Through Conversation Manager

Documents can be added to an agent's context:

```python
# Add document from path
await agent.conversation.add_context_from_path(
    "document.pdf",
    convert_to_md=True  # Enable markdown conversion
)

# Add raw content
await agent.conversation.add_context_message(
    html_content,
    mime_type="text/html"  # Helps converter selection
)
```

### Automatic Conversion

When passing paths to `Agent.run()`, documents are automatically converted if needed:

```python
# Will convert PDF if model doesn't support it directly
await agent.run(Path("document.pdf"))

# Multiple inputs
await agent.run(
    "Analyze this document:",
    Path("document.pdf"),
    "And compare it with:",
    Path("other.docx")
)
```

## Configuration

Document conversion is configured globally in your agent manifest:

```yaml
conversion:
  providers:
    - type: markitdown  # Uses MarkItDown for document conversion
      enabled: true
      llm_model: gpt-5  # Optional, for image descriptions
    # More providers will be added soon
```
