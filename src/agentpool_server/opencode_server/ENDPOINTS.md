# OpenCode API Compatibility Checklist

This document tracks the implementation status of OpenCode-compatible API endpoints.

## Status Legend
- [ ] Not implemented
- [x] Implemented
- [~] Partial / Stub
- [-] Skipped (not needed)

---

## Global

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [x] | GET | `/global/health` | Get server health and version |
| [x] | GET | `/global/event` | Get global events (SSE stream) |

---

## Project & Path

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [x] | GET | `/project` | List all projects |
| [x] | GET | `/project/current` | Get the current project |
| [x] | GET | `/path` | Get the current path |
| [x] | GET | `/vcs` | Get VCS info for current project |

---

## Instance

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [ ] | POST | `/instance/dispose` | Dispose the current instance |

---

## Config

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [x] | GET | `/config` | Get config info |
| [ ] | PATCH | `/config` | Update config |
| [~] | GET | `/config/providers` | List providers and default models |

---

## Provider

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [~] | GET | `/provider` | List all providers |
| [x] | GET | `/provider/auth` | Get provider authentication methods |
| [x] | POST | `/provider/{id}/oauth/authorize` | Authorize provider via OAuth |
| [x] | POST | `/provider/{id}/oauth/callback` | Handle OAuth callback |

---

## Sessions

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [x] | GET | `/session` | List all sessions |
| [x] | POST | `/session` | Create a new session |
| [x] | GET | `/session/status` | Get session status for all sessions |
| [x] | GET | `/session/{id}` | Get session details |
| [x] | DELETE | `/session/{id}` | Delete a session |
| [x] | PATCH | `/session/{id}` | Update session properties |
| [ ] | GET | `/session/{id}/children` | Get child sessions |
| [x] | GET | `/session/{id}/todo` | Get todo list for session |
| [x] | POST | `/session/{id}/init` | Analyze app, create AGENTS.md |
| [x] | POST | `/session/{id}/fork` | Fork session at message |
| [x] | POST | `/session/{id}/abort` | Abort running session |
| [x] | POST | `/session/{id}/share` | Share a session |
| [x] | DELETE | `/session/{id}/share` | Unshare a session |
| [x] | GET | `/session/{id}/diff` | Get diff for session |
| [x] | POST | `/session/{id}/summarize` | Summarize the session |
| [x] | POST | `/session/{id}/revert` | Revert a message |
| [x] | POST | `/session/{id}/unrevert` | Restore reverted messages |
| [x] | GET | `/session/{id}/permissions` | Get pending permission requests |
| [x] | POST | `/session/{id}/permissions/{permissionID}` | Respond to permission request |

---

## Messages

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [x] | GET | `/session/{id}/message` | List messages in session |
| [~] | POST | `/session/{id}/message` | Send message (wait for response) |
| [x] | GET | `/session/{id}/message/{messageID}` | Get message details |
| [ ] | POST | `/session/{id}/prompt_async` | Send message async (no wait) |
| [x] | POST | `/session/{id}/command` | Execute slash command (MCP prompts) |
| [x] | POST | `/session/{id}/shell` | Run shell command |

---

## Commands

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [x] | GET | `/command` | List all commands (MCP prompts) |

---

## Files

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [x] | GET | `/find?pattern=` | Search for text in files |
| [x] | GET | `/find/file?query=` | Find files by name |
| [~] | GET | `/find/symbol?query=` | Find workspace symbols |
| [x] | GET | `/file?path=` | List files and directories |
| [x] | GET | `/file/content?path=` | Read a file |
| [~] | GET | `/file/status` | Get status for tracked files |

---

## Tools (Experimental)

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [x] | GET | `/experimental/tool/ids` | List all tool IDs |
| [x] | GET | `/experimental/tool?provider=&model=` | List tools with schemas |

---

## LSP, Formatters & MCP

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [~] | GET | `/lsp` | Get LSP server status |
| [~] | GET | `/formatter` | Get formatter status |
| [~] | GET | `/mcp` | Get MCP server status |
| [x] | POST | `/mcp` | Add MCP server dynamically |

---

## Agents

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [~] | GET | `/agent` | List all available agents |

---

## Logging

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [x] | POST | `/log` | Write log entry |

---

## Modes

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [~] | GET | `/mode` | List all modes |

---

## PTY (Pseudo-Terminal)

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [ ] | GET | `/pty` | List all PTY sessions |
| [ ] | POST | `/pty` | Create a new PTY session |
| [ ] | GET | `/pty/{ptyID}` | Get PTY session details |
| [ ] | PATCH | `/pty/{ptyID}` | Update PTY session (resize, etc.) |
| [ ] | DELETE | `/pty/{ptyID}` | Remove/kill PTY session |
| [ ] | GET | `/pty/{ptyID}/connect` | Connect to PTY (WebSocket) |

### PTY SSE Event Types

| Status | Event Type | Description |
|--------|------------|-------------|
| [ ] | `pty.created` | PTY session created |
| [ ] | `pty.updated` | PTY session updated |
| [ ] | `pty.exited` | PTY process exited |
| [ ] | `pty.deleted` | PTY session deleted |

---

## TUI (Skipped)

These endpoints are for driving the TUI and are not needed for programmatic access.

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [-] | POST | `/tui/append-prompt` | Append text to prompt |
| [-] | POST | `/tui/open-help` | Open help dialog |
| [-] | POST | `/tui/open-sessions` | Open session selector |
| [-] | POST | `/tui/open-themes` | Open theme selector |
| [-] | POST | `/tui/open-models` | Open model selector |
| [-] | POST | `/tui/submit-prompt` | Submit current prompt |
| [-] | POST | `/tui/clear-prompt` | Clear the prompt |
| [-] | POST | `/tui/execute-command` | Execute a command |
| [-] | POST | `/tui/show-toast` | Show toast notification |
| [-] | GET | `/tui/control/next` | Wait for next control request |
| [-] | POST | `/tui/control/response` | Respond to control request |

---

## Auth

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [ ] | PUT | `/auth/{id}` | Set authentication credentials |

---

## Events

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [x] | GET | `/event` | SSE event stream |

### SSE Event Types

| Status | Event Type | Description |
|--------|------------|-------------|
| [x] | `server.connected` | Server connected |
| [x] | `session.created` | Session created |
| [x] | `session.updated` | Session updated |
| [x] | `session.deleted` | Session deleted |
| [x] | `session.status` | Session status changed |
| [x] | `session.idle` | Session became idle (deprecated) |
| [x] | `message.updated` | Message updated |
| [x] | `message.part.updated` | Message part updated |
| [x] | `permission.request` | Tool permission requested |
| [x] | `permission.resolved` | Permission request resolved |

---

## Docs

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [x] | GET | `/doc` | OpenAPI 3.1 specification |

---

## Implementation Summary

### Completed (TUI can connect!)
- Health check and SSE events
- Session CRUD operations
- File listing and reading
- Path/Project/VCS info
- Config endpoint
- All stubs needed for TUI to render

### Next Steps
1. **Agent Integration** - Wire up actual LLM calls for `/session/{id}/message`
2. **Provider Discovery** - Populate `/config/providers` with real models
3. **File Search** - Implement `/find` endpoints

---

## Testing

**Terminal 1:** Start server
```bash
duty opencode-server
```

**Terminal 2:** Attach TUI
```bash
duty opencode-tui
```

Or combined (less reliable for interactive use):
```bash
duty opencode
```
