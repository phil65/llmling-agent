# Tool Configuration

Tools provide agents with specific capabilities. The tool configuration supports three main types: import-based, CrewAI tools, and LangChain tools.

## Import Tools

Tools imported directly from Python modules or packages.

```yaml
agents:
  my-agent:
    tools:
      - type: "import"  # tool discriminator
        import_path: "webbrowser.open"  # module path to import
        name: "open_url"  # optional override for tool name
        description: "Open URL in default browser"  # optional description
        enabled: true  # whether tool is initially available
        requires_confirmation: false  # whether to ask before executing
        metadata:  # additional tool metadata
          category: "browser"
          version: "1.0"
```

## CrewAI Tools

Tools from the CrewAI ecosystem.

```yaml
agents:
  research-agent:
    tools:
      - type: "crewai"
        import_path: "crewai_tools.BrowserTools.BraveSearch"
        params:  # tool-specific parameters
          api_key: "${BRAVE_API_KEY}"
        name: "search"  # optional overrides
        description: "Search the web using Brave"  # override description
        enabled: true
        requires_confirmation: true
```

## LangChain Tools

Tools from the LangChain ecosystem.

```yaml
agents:
  langchain-agent:
    tools:
      - type: "langchain"
        tool_name: "serpapi"  # name of LangChain tool
        params:  # tool-specific parameters
          api_key: "${SERPAPI_KEY}"
        name: "web_search"  # optional overrides
        description: "Search using SerpAPI"
        enabled: true
        requires_confirmation: false
```

## Shorthand Syntax

For simple tools, you can use string shortcuts:

```yaml
tools:
  - "webbrowser.open"  # Import tool with default settings
  - "crewai_tools.BraveSearchTool"  # CrewAI tool with defaults
  - "langchain.tools.serpapi.SerpAPIWrapper"  # LangChain tool
```

## Configuration Notes

- The `type` field identifies the tool type (import/crewai/langchain)
- Tool names must be unique within an agent
- Tools can require specific capabilities for access control
- Tool confirmation can be:
  - Never required (`requires_confirmation: false`)
  - Always required (`requires_confirmation: true`)
  - Based on agent settings (`requires_confirmation` not set)
- Priority affects execution order (lower values = higher priority)
- Environment variables are supported in configuration values
- Metadata can be used for custom tool categorization and filtering
