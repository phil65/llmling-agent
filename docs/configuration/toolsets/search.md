---
title: Search Toolset
description: Web and news search capabilities
icon: material/magnify
---

# Search Toolset

The Search toolset provides web and news search capabilities using various search providers.

## Basic Usage

```yaml
agents:
  researcher:
    toolsets:
      - type: search
        provider: tavily
```

## Supported Providers

### Web Search

- `tavily` - Tavily AI search
- `brave` - Brave Search
- `searxng` - SearXNG meta search
- `serper` - Serper.dev Google search
- `you` - You.com search

### News Search

- `tavily` - Tavily news search
- `brave` - Brave News
- `serper` - Serper news
- `newsapi` - NewsAPI.org

## Configuration

### Web Search Only

```yaml
toolsets:
  - type: search
    provider: tavily
    api_key: ${TAVILY_API_KEY}
```

### News Search Only

```yaml
toolsets:
  - type: search
    news_provider: newsapi
    news_api_key: ${NEWSAPI_KEY}
```

### Both Web and News

```yaml
toolsets:
  - type: search
    provider: brave
    api_key: ${BRAVE_API_KEY}
    news_provider: newsapi
    news_api_key: ${NEWSAPI_KEY}
```

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.toolsets.SearchToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///

## Environment Variables

Most providers require API keys. You can provide them directly or via environment variables:

| Provider | Environment Variable |
|----------|---------------------|
| Tavily | `TAVILY_API_KEY` |
| Brave | `BRAVE_API_KEY` |
| Serper | `SERPER_API_KEY` |
| NewsAPI | `NEWSAPI_KEY` |
