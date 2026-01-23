# Claude Code SDK Message Metadata

This document describes the structure of messages and metadata returned by the Claude Code SDK (`clawd_code_sdk`).

> **Note**: The `convert_to_opencode_metadata()` function in
> `src/agentpool/agents/claude_code_agent/converters.py` converts these shapes
> to the OpenCode TUI metadata format for rich display of file operations.

## Message Types

The SDK streams these message types via `query()`:

- `SystemMessage` - Session initialization
- `AssistantMessage` - Claude's responses (text and tool calls)
- `UserMessage` - Tool results and user input
- `ResultMessage` - Final session result
- `StreamEvent` - Raw Anthropic API streaming events (when streaming enabled)

---

## SystemMessage

Sent at session start with initialization data.

```typescript
interface SystemMessage {
  subtype: "init";
  data: {
    type: "system";
    subtype: "init";
    cwd: string;                    // Working directory
    session_id: string;             // UUID for this session
    tools: string[];                // Available tool names
    mcp_servers: string[];          // Connected MCP servers
    model: string;                  // Model ID (e.g., "claude-opus-4-5-20251101")
    permissionMode: string;         // "default" | "acceptEdits" | "bypassPermissions" | "plan"
    slash_commands: string[];       // Available slash commands
    apiKeySource: string;           // "none" | "env" | "config" | etc.
    claude_code_version: string;    // CLI version
    output_style: string;           // "default" | "json" | etc.
    agents: string[];               // Available subagents
    skills: string[];               // Registered skills
    plugins: string[];              // Active plugins
    uuid: string;                   // Message UUID
  };
}
```

---

## AssistantMessage

Claude's responses containing text and/or tool calls.

```typescript
interface AssistantMessage {
  content: ContentBlock[];          // Array of content blocks
  model: string;                    // Model that generated this
  parent_tool_use_id: string | null; // If from a subagent
  error: string | null;             // Error if message failed
}

type ContentBlock = TextBlock | ToolUseBlock | ThinkingBlock;

interface TextBlock {
  type: "text";
  text: string;
}

interface ToolUseBlock {
  type: "tool_use";
  id: string;                       // Tool use ID (e.g., "toolu_01ABC...")
  name: string;                     // Tool name
  input: Record<string, any>;       // Tool arguments
}

interface ThinkingBlock {
  type: "thinking";
  thinking: string;
}
```

---

## UserMessage

Contains tool results after tool execution.

```typescript
interface UserMessage {
  content: ToolResultBlock[];
  uuid: string;                     // Message UUID
  parent_tool_use_id: string | null;
  tool_use_result: ToolResult;      // Structured result data (see below)
}

interface ToolResultBlock {
  tool_use_id: string;              // Matches ToolUseBlock.id
  content: string;                  // Human-readable result
  is_error: boolean | null;
}
```

### Tool Result Shapes

The `tool_use_result` field contains structured data specific to each tool.

#### Write Tool Result

```typescript
interface WriteToolResult {
  type: "create";
  filePath: string;                 // Absolute path to created file
  content: string;                  // Full file content
  structuredPatch: PatchHunk[];     // Empty for new files
  originalFile: string | null;      // null for new files
}
```

#### Edit Tool Result

```typescript
interface EditToolResult {
  filePath: string;                 // Absolute path to edited file
  oldString: string;                // Original text that was replaced
  newString: string;                // New text that replaced it
  originalFile: string;             // Full original file content before edit
  structuredPatch: PatchHunk[];     // Unified diff hunks
  userModified: boolean;            // Whether user modified the edit
  replaceAll: boolean;              // Whether replace_all was used
}

interface PatchHunk {
  oldStart: number;                 // 1-based start line in original
  oldLines: number;                 // Number of lines in original
  newStart: number;                 // 1-based start line in new
  newLines: number;                 // Number of lines in new
  lines: string[];                  // Diff lines with prefixes:
                                    //   " " = context (unchanged)
                                    //   "+" = added
                                    //   "-" = removed
}
```

#### Read Tool Result

```typescript
interface ReadToolResult {
  type: "text";
  file: {
    filePath: string;               // Absolute path
    content: string;                // File content (possibly truncated)
    numLines: number;               // Lines returned
    startLine: number;              // 1-based start line
    totalLines: number;             // Total lines in file
  };
}
```

#### Bash Tool Result

Successful command:
```typescript
interface BashToolResult {
  stdout: string;                   // Command stdout output
  stderr: string;                   // Command stderr output
  interrupted: boolean;             // Whether command was interrupted (e.g., timeout)
  isImage: boolean;                 // Whether output is base64 image data
}
```

Example successful result:
```json
{
  "stdout": "Hello from bash",
  "stderr": "",
  "interrupted": false,
  "isImage": false
}
```

Example with output:
```json
{
  "stdout": "total 675396\ndrwxrwxrwt 50 root root 6640 Jan 20 19:12 .\n...",
  "stderr": "",
  "interrupted": false,
  "isImage": false
}
```

**Converted to OpenCode format:**
```typescript
interface BashMetadata {
  output: string;        // Combined stdout + stderr
  exit: number | null;   // Exit code (0 for success, null if interrupted)
  description: string;   // Command description from tool input
}
```

#### Bash Tool Input

The input sent to the Bash tool:
```typescript
interface BashToolInput {
  command: string;                  // Shell command to execute
  description: string;              // Human-readable description of what the command does
}
```

Example:
```json
{
  "command": "ls -la /tmp",
  "description": "List files in /tmp directory"
}
```

#### TodoWrite Tool Result

```typescript
interface TodoWriteToolResult {
  oldTodos: TodoItem[];             // Previous todo list
  newTodos: TodoItem[];             // Updated todo list
}

interface TodoItem {
  content: string;                  // Task description
  status: string;                   // "pending" | "in_progress" | "completed"
  activeForm?: string;              // Present tense description (optional)
}
```

Example:
```json
{
  "oldTodos": [],
  "newTodos": [
    {
      "content": "Fix critical bug in auth system",
      "status": "pending",
      "activeForm": "Fixing critical bug in auth system"
    },
    {
      "content": "Review pull requests",
      "status": "in_progress",
      "activeForm": "Reviewing pull requests"
    }
  ]
}
```

**Converted to OpenCode format:**
```typescript
interface TodoWriteMetadata {
  todos: Array<{
    id: string;          // Generated UUID
    content: string;     // Task description
    status: string;      // Same as SDK
    priority: string;    // Inferred: "high" | "medium" | "low"
  }>;
}
```

Priority is inferred from:
1. Keywords in content ("critical", "urgent" → high; "later", "nice to have" → low)
2. Position in list (first third → high, last third → low)

---

#### Error Results

When a command fails (`is_error: true`), `tool_use_result` is a plain string:

```typescript
// For failed bash commands:
tool_use_result: string;  // e.g., "Error: Exit code 1\ncat: /nonexistent/file: No such file or directory (os error 2)"

// For permission/system errors:
tool_use_result: string;  // e.g., "Error: EACCES: permission denied, mkdir '/Users'"
```

---

## ResultMessage

Final message with session statistics.

```typescript
interface ResultMessage {
  subtype: "success" | "error";
  duration_ms: number;              // Total wall-clock time
  duration_api_ms: number;          // Time spent in API calls
  is_error: boolean;
  num_turns: number;                // Number of conversation turns
  session_id: string;
  total_cost_usd: number;           // Estimated cost
  usage: UsageStats;
  result: string;                   // Final text response
  structured_output: any | null;    // If structured output was requested
}

interface UsageStats {
  input_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
  output_tokens: number;
  server_tool_use: {
    web_search_requests: number;
    web_fetch_requests: number;
  };
  service_tier: string;             // "standard" | "priority"
  cache_creation: {
    ephemeral_1h_input_tokens: number;
    ephemeral_5m_input_tokens: number;
  };
}
```

---

## StreamEvent

Raw Anthropic API streaming events (when streaming is enabled).

```typescript
interface StreamEvent {
  uuid: string;
  session_id: string;
  event: AnthropicStreamEvent;      // Raw API event
  parent_tool_use_id: string | null;
}
```

Common `event.type` values:
- `message_start` - Start of a message
- `content_block_start` - Start of a content block (text or tool_use)
- `content_block_delta` - Incremental content (text_delta or input_json_delta)
- `content_block_stop` - End of a content block
- `message_delta` - Message-level updates (stop_reason, usage)
- `message_stop` - End of message

---

## Example Flow

1. **SystemMessage** (init) - Session starts
2. **AssistantMessage** (TextBlock) - Claude explains what it will do
3. **AssistantMessage** (ToolUseBlock) - Claude calls Write tool
4. **UserMessage** (ToolResultBlock) - Write result with `tool_use_result`
5. **AssistantMessage** (ToolUseBlock) - Claude calls Edit tool
6. **UserMessage** (ToolResultBlock) - Edit result with `tool_use_result` containing `structuredPatch`
7. **AssistantMessage** (TextBlock) - Claude summarizes what was done
8. **ResultMessage** - Session complete with usage stats
