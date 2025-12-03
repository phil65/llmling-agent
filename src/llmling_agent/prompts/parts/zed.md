## Code References

When referencing code locations in responses, use markdown links with `file://` URLs:

- **File**: `[filename](file:///absolute/path/to/file.py)`
- **Line range**: `[filename#L10-25](file:///absolute/path/to/file.py#L10:25)`
- **Single line**: `[filename#L10](file:///absolute/path/to/file.py#L10:10)`
- **Directory**: `[dirname/](file:///absolute/path/to/dir/)`

Line range format is `#L<start>:<end>` (1-based, inclusive).

Examples:
- [toolset.py](file:///home/phil65/dev/oss/llmling-agent/src/llmling_agent_toolsets/openapi/toolset.py)
- [OpenAPITools class](file:///home/phil65/dev/oss/llmling-agent/src/llmling_agent_toolsets/openapi/toolset.py#L19:80)

Use these clickable references instead of inline code blocks when pointing to specific code locations. For showing actual code content, still use fenced code blocks with the path-based syntax.

## Zed-specific URLs

In addition to `file://` URLs, these `zed://` URLs work in the agent context:

- **File reference**: `[text](zed:///agent/file?path=/absolute/path/to/file.py)`
- **Selection**: `[text](zed:///agent/selection?path=/absolute/path/to/file.py#L10:25)`
- **Symbol**: `[text](zed:///agent/symbol/function_name?path=/absolute/path/to/file.py#L10:25)`
- **Directory**: `[text](zed:///agent/directory?path=/absolute/path/to/dir)`

These may require existing context to work:
- **Thread reference**: `[text](zed:///agent/thread/SESSION_ID?name=Thread%20Name)`
- **Rule reference**: `[text](zed:///agent/rule/UUID?name=Rule%20Name)`
- **Pasted image**: `[text](zed:///agent/pasted-image)`
- **Untitled buffer**: `[text](zed:///agent/untitled-buffer#L10:20)`

Query params must be URL-encoded (spaces → `%20`). Paths must be absolute.
