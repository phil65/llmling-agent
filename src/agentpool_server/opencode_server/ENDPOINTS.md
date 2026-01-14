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
| [x] | PATCH | `/config` | Update config |
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
| [x] | POST | `/session/{id}/message` | Send message (wait for response) |
| [x] | GET | `/session/{id}/message/{messageID}` | Get message details |
| [x] | POST | `/session/{id}/prompt_async` | Send message async (no wait) |
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
| [x] | GET | `/lsp` | Get LSP server status |
| [x] | POST | `/lsp/start` | Start an LSP server |
| [x] | POST | `/lsp/stop` | Stop an LSP server |
| [x] | GET | `/lsp/servers` | List available LSP servers |
| [x] | GET | `/lsp/diagnostics` | Get LSP diagnostics (CLI-based) |
| [x] | GET | `/formatter` | Get formatter status (stub) |
| [~] | GET | `/mcp` | Get MCP server status |
| [x] | POST | `/mcp` | Add MCP server dynamically |
| [x] | GET | `/experimental/resource` | List MCP resources from connected servers |

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
| [x] | GET | `/pty` | List all PTY sessions |
| [x] | POST | `/pty` | Create a new PTY session |
| [x] | GET | `/pty/{ptyID}` | Get PTY session details |
| [x] | PATCH | `/pty/{ptyID}` | Update PTY session (resize, etc.) |
| [x] | DELETE | `/pty/{ptyID}` | Remove/kill PTY session |
| [x] | WS | `/pty/{ptyID}/connect` | Connect to PTY (WebSocket) |

### PTY SSE Event Types

| Status | Event Type | Description |
|--------|------------|-------------|
| [x] | `pty.created` | PTY session created |
| [x] | `pty.updated` | PTY session updated |
| [x] | `pty.exited` | PTY process exited |
| [x] | `pty.deleted` | PTY session deleted |

---

## TUI (External Control)

These endpoints allow external integrations (e.g., VSCode extension) to control the TUI
by broadcasting events via SSE.

| Status | Method | Path | Description |
|--------|--------|------|-------------|
| [x] | POST | `/tui/append-prompt` | Append text to prompt |
| [x] | POST | `/tui/open-help` | Open help dialog |
| [x] | POST | `/tui/open-sessions` | Open session selector |
| [x] | POST | `/tui/open-themes` | Open theme selector |
| [x] | POST | `/tui/open-models` | Open model selector |
| [x] | POST | `/tui/submit-prompt` | Submit current prompt |
| [x] | POST | `/tui/clear-prompt` | Clear the prompt |
| [x] | POST | `/tui/execute-command` | Execute a command |
| [x] | POST | `/tui/show-toast` | Show toast notification |
| [-] | GET | `/tui/control/next` | Wait for next control request (not needed) |
| [-] | POST | `/tui/control/response` | Respond to control request (not needed) |

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

All event types supported by the OpenCode protocol:

| Status | Event Type | Description |
|--------|------------|-------------|
| [x] | `server.connected` | Server connected (sent on SSE connect) |
| [-] | `global.disposed` | Global instance disposed (multi-project, not needed) |
| [-] | `installation.updated` | Installation updated (auto-upgrade complete, not needed) |
| [x] | `installation.update-available` | Update available (via `tui.toast.show` workaround) |
| [x] | `project.updated` | Project metadata updated |
| [-] | `server.instance.disposed` | Server instance disposed (multi-project, not needed) |
| [x] | `lsp.updated` | LSP server status updated |
| [~] | `lsp.client.diagnostics` | LSP client diagnostics received |
| [x] | `session.created` | Session created |
| [x] | `session.updated` | Session updated |
| [x] | `session.deleted` | Session deleted |
| [x] | `session.status` | Session status changed (running/idle/error) |
| [x] | `session.idle` | Session became idle (deprecated but used by TUI) |
| [x] | `session.compacted` | Session context was compacted/summarized |
| [ ] | `session.diff` | Session file diff updated |
| [x] | `session.error` | Session encountered an error |
| [x] | `message.updated` | Message created or updated |
| [x] | `message.removed` | Message removed (during revert) |
| [x] | `message.part.updated` | Message part (text, tool, etc.) updated |
| [x] | `message.part.removed` | Message part removed (during revert) |
| [x] | `permission.asked` | Tool permission requested (awaiting user response) |
| [x] | `permission.replied` | Permission request resolved (user responded) |
| [x] | `todo.updated` | Todo list item updated |
| [ ] | `file.edited` | File was edited |
| [x] | `file.watcher.updated` | File watcher detects project file changes |
| [x] | `vcs.branch.updated` | VCS branch changed (polling-based) |
| [ ] | `mcp.tools.changed` | MCP server tools changed |
| [ ] | `command.executed` | Slash command executed |
| [x] | `tui.prompt.append` | Append text to TUI prompt input |
| [x] | `tui.command.execute` | Execute a TUI command |
| [x] | `tui.toast.show` | Show toast notification in TUI |
| [x] | `pty.created` | PTY session created |
| [x] | `pty.updated` | PTY session updated |
| [x] | `pty.exited` | PTY process exited |
| [x] | `pty.deleted` | PTY session deleted |

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

---

## Tool Metadata Support - Complete OpenCode UI Coverage

**These are ALL 11 tools registered in OpenCode's UI that use metadata for enhanced rendering:**

| # | Tool | AgentPool | Metadata | UI Feature |
|---|------|-----------|----------|------------|
| 1 | `read` | ✅ **DONE** | `preview`, `truncated` | File preview, truncation badge |
| 2 | `list` | ✅ **DONE** | `count`, `truncated` | File count, directory tree |
| 3 | `glob` | ❌ **MISSING** | `count`, `truncated` | File count, pattern display |
| 4 | `grep` | ✅ **DONE** | `matches`, `truncated` | Match count badge |
| 5 | `webfetch` | ❌ **MISSING** | `url`, `format` | URL display |
| 6 | `task` | ❌ **MISSING** | `summary`, `sessionId` | **Sub-agent tool list** |
| 7 | `bash` | ✅ **DONE** | `output`, `exit`, `description` | Live output, exit code |
| 8 | `edit` | ✅ **DONE** | `diff`, `filediff`, `diagnostics` | **Diff viewer**, LSP errors |
| 9 | `write` | ⚠️ **PARTIAL** | `filePath`, `content`, (TODO: `diagnostics`) | Code viewer, LSP errors |
| 10 | `todowrite` | ✅ **DONE** | `todos` | **Interactive checkboxes** |
| 11 | `question` | ✅ **DONE** | `answers` | **Q&A display** |

### Summary
- **Total:** 11 OpenCode UI tools
- **Implemented:** 6/11 (✅)
- **Partial:** 1/11 (⚠️)
- **Missing:** 4/11 (❌)

### Missing Tools for 100% Coverage
1. **`glob`** - File pattern matching (HIGH priority - complements grep)
2. **`task`** - Sub-agent execution tracking (HIGH priority - delegation)
3. **`webfetch`** - Web content fetching (LOW priority - external)
4. **`write` diagnostics** - LSP integration (MEDIUM priority)

### Other OpenCode Tools (Not UI-Rendered)
These exist in OpenCode but don't have special UI treatment:
- `plan`, `batch`, `multiedit`, `patch`, `lsp`, `skill`, `codesearch`, `websearch`

See [`OPENCODE_UI_TOOLS_COMPLETE.md`](file:///home/phil65/dev/oss/agentpool/OPENCODE_UI_TOOLS_COMPLETE.md) for detailed metadata specs.

---

## Tool UI Rendering

The OpenCode TUI has special rendering for certain tool names. Tools must use these exact names
and parameter formats (after snake_case → camelCase conversion) to get custom UI treatment.

Parameter conversion is handled in `converters.py` via `_PARAM_NAME_MAP`.

| Tool Name | Expected Parameters (camelCase) | UI Treatment |
|-----------|--------------------------------|--------------|
| `read` | `filePath`, `offset`, `limit` | Glasses icon, shows filename |
| `list` | `path` | Bullet-list icon, shows directory |
| `glob` | `path`, `pattern` | Magnifying-glass icon, shows pattern |
| `grep` | `path`, `pattern`, `include` | Magnifying-glass icon, shows pattern |
| `webfetch` | `url`, `format` | Window icon, shows URL |
| `task` | `subagent_type`, `description` | Task icon, shows agent summary |
| `bash` | `command`, `description` | Console icon, shows command + output |
| `edit` | `filePath`, `oldString`, `newString` | Code icon, **diff view** |
| `write` | `filePath`, `content` | Code icon, **syntax-highlighted content** |
| `todowrite` | `todos` (array with `status`, `content`) | Checklist icon, checkbox list |
| `todoread` | - | Filtered out (not displayed) |

### Metadata

Some tools also use `props.metadata` for additional UI data:

| Tool | Metadata Fields | Description |
|------|-----------------|-------------|
| `edit` | `filediff`, `diagnostics` | Diff data and LSP diagnostics |
| `write` | `filePath`, `content` | File path and content for UI display (diagnostics TODO) |
| `bash` | `command` | Fallback if `input.command` missing |
| `task` | `summary`, `sessionId` | Child tool summary and session ID |

### Parameter Name Mapping

The `_PARAM_NAME_MAP` in `converters.py` converts our snake_case to TUI's camelCase:

```python
_PARAM_NAME_MAP = {
    "path": "filePath",
    "file_path": "filePath",
    "old_string": "oldString",
    "new_string": "newString",
    "replace_all": "replaceAll",
    "line_hint": "lineHint",
}
```


