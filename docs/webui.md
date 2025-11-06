## Web Interface Integration

The web interface provides a graphical way to manage and interact with agents.

### Starting the Interface

```bash
# Basic launch
llmling-agent launch

# Custom server settings
llmling-agent launch --host 0.0.0.0 --port 8080 --share
```

### Features

1. **Configuration Management**
   - Upload agent configurations
   - Select active configuration
   - Choose agents to interact with

2. **Agent Interaction**
   - Real-time chat with streaming responses
   - Visual tool state management
   - Model overrides
   - Conversation history

3. **Debug Features**
   - Token usage monitoring
   - Debug log display
   - Tool execution tracking

### Example Usage

1. Launch the interface:
```bash
llmling-agent launch
```

2. Upload or select your agent configuration:
```yaml
# agents.yml
agents:
  assistant:
    model: openai:gpt-5-mini
    system_prompts:
      - "You are a helpful assistant."
```

3. Use the web interface to:
   - Chat with the agent
   - Toggle available tools
   - Monitor token usage
   - View conversation history

Key features visible in the UI:
- Real-time response streaming
- Tool availability indicators
- Token usage statistics
- Debug log viewer
- Chat history display
```
