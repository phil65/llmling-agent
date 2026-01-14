# OpenCode Tools with UI Metadata Support - Complete List

## ALL OpenCode Tools That Get Special UI Treatment

These are **all** tools registered in OpenCode's UI (`message-part.tsx`) that use metadata for enhanced rendering:

| # | Tool | AgentPool Status | Metadata Fields | UI Feature |
|---|------|------------------|-----------------|------------|
| 1 | `read` | ✅ **DONE** | `preview`, `truncated` | Shows file preview, truncation badge |
| 2 | `list` | ✅ **DONE** | `count`, `truncated` | File count, directory tree |
| 3 | `glob` | ❌ **MISSING** | `count`, `truncated` | File count, pattern display |
| 4 | `grep` | ✅ **DONE** | `matches`, `truncated` | Match count badge |
| 5 | `webfetch` | ❌ **MISSING** | `url`, `format` | URL display, format indicator |
| 6 | `task` | ❌ **MISSING** | `summary`, `sessionId` | Sub-agent tool list, navigation |
| 7 | `bash` | ✅ **DONE** | `output`, `exit`, `description` | Live output, exit code, command |
| 8 | `edit` | ✅ **DONE** | `diff`, `filediff`, `diagnostics` | **Diff viewer**, LSP errors |
| 9 | `write` | ⚠️ **PARTIAL** | `diagnostics`, `filepath`, `exists` | Code viewer, LSP errors |
| 10 | `todowrite` | ✅ **DONE** | `todos` | **Interactive checkbox list** |
| 11 | `question` | ✅ **DONE** | `answers` | **Q&A display** |

---

## Summary

**Total OpenCode UI Tools:** 11  
**Implemented:** 6 (✅)  
**Partial:** 1 (⚠️)  
**Missing:** 4 (❌)

### ✅ Implemented (6/11)
- `read`, `list`, `grep`, `bash`, `edit`, `todowrite`, `question`

### ⚠️ Partial (1/11)
- `write` - exists but missing diagnostics integration

### ❌ Missing (4/11)
1. **`glob`** - File pattern matching (like grep but for filenames)
2. **`webfetch`** - Fetch web content
3. **`task`** - Sub-agent execution with tool summary
4. **`write` diagnostics** - Need LSP integration

---

## Detailed Metadata Specifications

### 1. read ✅
```typescript
metadata: {
  preview: string      // First 20 lines
  truncated: boolean   // Was content cut off?
}
```

### 2. list ✅
```typescript
metadata: {
  count: number        // Total files/dirs
  truncated: boolean   // Hit max limit?
}
```

### 3. glob ❌ NOT IMPLEMENTED
```typescript
metadata: {
  count: number        // Files matched
  truncated: boolean   // Results limited?
}
```

### 4. grep ✅
```typescript
metadata: {
  matches: number      // Match count
  truncated: boolean   // Results limited?
}
```

### 5. webfetch ❌ NOT IMPLEMENTED
```typescript
metadata: {
  url: string          // Target URL
  format: string       // "markdown" | "html" | "text"
  timeout?: number     // Request timeout
}
```

### 6. task ❌ NOT IMPLEMENTED
```typescript
metadata: {
  summary: Array<{     // Sub-agent's tool calls
    id: string
    tool: string
    state: {
      status: string
      title?: string
    }
  }>
  sessionId: string    // For navigation to sub-agent session
}
```

### 7. bash ✅
```typescript
metadata: {
  output: string       // Combined stdout+stderr
  exit: number | null  // Exit code
  description: string  // Command description
}
```

### 8. edit ✅
```typescript
metadata: {
  diff: string         // Unified diff format
  filediff: {
    file: string
    before: string
    after: string
    additions: number
    deletions: number
  }
  diagnostics: Record<string, Diagnostic[]>
}
```

### 9. write ⚠️ PARTIAL
```typescript
metadata: {
  diagnostics: Record<string, Diagnostic[]>  // ⚠️ Not implemented
  filepath: string
  exists: boolean      // Did file exist before?
}
```

### 10. todowrite ✅
```typescript
metadata: {
  todos: Array<{
    content: string
    status: "completed" | "pending" | "in_progress"
  }>
}
```

### 11. question ✅
```typescript
metadata: {
  answers: Array<Array<string>>  // One array per question
}
```

---

## Implementation Priority

### Must Have (P0) - Core Functionality
- ✅ All filesystem tools: read, list, grep
- ✅ Basic execution: bash
- ✅ File editing: edit
- ✅ Planning: todowrite
- ✅ Interaction: question

### Should Have (P1) - Enhanced Features
- ❌ **`glob`** - File pattern search (HIGH - complements grep)
- ❌ **`task`** - Sub-agent tracking (HIGH - important for delegation)
- ⚠️ **`write` diagnostics** - LSP integration (MEDIUM)

### Nice to Have (P2) - Optional
- ❌ **`webfetch`** - Web content (LOW - external dependency)

---

## What's Missing vs OpenCode

OpenCode has these additional tools that we don't implement:

| OpenCode Tool | Purpose | Why Not in AgentPool? |
|---------------|---------|----------------------|
| `plan` | Plan management | We use `get_plan`/`set_plan` instead |
| `todoread` | Read todos | Handled by `get_plan` |
| `batch` | Parallel tool execution | Generic utility, not core |
| `multiedit` | Multi-file edits | Advanced feature |
| `patch` | Apply git patches | Git-specific |
| `lsp` | LSP queries | We have LSP but not as a tool |
| `skill` | Execute skills | OpenCode-specific |
| `codesearch` | Semantic search | Advanced feature |
| `websearch` | Web search | External service |

**Note:** These are all valid tools but not part of the core UI metadata rendering system.

---

## Next Steps

To achieve 100% OpenCode UI compatibility:

1. **Implement `glob` tool** - File pattern matching with metadata
2. **Implement `task` tool** - Sub-agent execution tracking
3. **Add diagnostics to `write`** - LSP integration
4. **Implement `webfetch`** - Web content fetching (optional)

After these 4 implementations, we'll have **complete** OpenCode UI metadata support!
