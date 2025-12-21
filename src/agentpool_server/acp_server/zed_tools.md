File System Operations

### `copy_path`
Copies a file or directory in the project
- **source_path** (required): The source path of the file or directory to copy
- **destination_path** (required): The destination path where the file or directory should be copied to

### `create_directory`
Creates a new directory at the specified path
- **path** (required): The path of the new directory

### `delete_path`
Deletes the file or directory at the specified path
- **path** (required): The path of the file or directory to delete

### `move_path`
Moves or renames a file or directory
- **source_path** (required): The source path of the file or directory to move/rename
- **destination_path** (required): The destination path where the file or directory should be moved/renamed to

## File Content Operations

### `edit_file`
Creates a new file or edits an existing file
- **display_description** (required): A one-line, user-friendly markdown description of the edit
- **path** (required): The full path of the file to create or modify
- **mode** (required): The mode of operation - 'edit', 'create', or 'overwrite'

### `read_file`
Reads the content of a file
- **path** (required): The relative path of the file to read
- **start_line** (optional): Line number to start reading on (1-based index)
- **end_line** (optional): Line number to end reading on (1-based index, inclusive)

## Search and Discovery

### `find_path`
Fast file path pattern matching with glob patterns
- **glob** (required): The glob pattern to match against every path in the project
- **offset** (optional): Starting position for paginated results (default: 0)

### `grep`
Searches file contents with regular expressions
- **regex** (required): A regex pattern to search for in the project
- **include_pattern** (optional): A glob pattern for the paths of files to include in the search
- **case_sensitive** (optional): Whether the regex is case-sensitive (default: false)
- **offset** (optional): Starting position for paginated results (default: 0)

### `list_directory`
Lists files and directories in a given path
- **path** (required): The fully-qualified path of the directory to list

## Code Quality and Diagnostics

### `diagnostics`
Get errors and warnings for the project or a specific file
- **path** (optional): The path to get diagnostics for. If not provided, returns a project-wide summary

## External Resources

### `fetch`
Fetches a URL and returns the content as Markdown
- **url** (required): The URL to fetch

### `resolve-library-id`
Resolves a package/product name to a Context7-compatible library ID
- **libraryName** (required): Library name to search for and retrieve a Context7-compatible library ID

### `get-library-docs`
Fetches up-to-date documentation for a library
- **context7CompatibleLibraryID** (required): Exact Context7-compatible library ID
- **topic** (optional): Topic to focus documentation on
- **tokens** (optional): Maximum number of tokens of documentation to retrieve (default: 5000)

## System Operations

### `terminal`
Executes a shell one-liner and returns the combined output
- **command** (required): The one-liner command to execute
- **cd** (required): Working directory for the command (must be one of the root directories)

### `now`
Returns the current datetime in RFC 3339 format
- **timezone** (required): The timezone to use - 'utc' or 'local'

## Analysis and Planning

### `thinking`
A tool for thinking through problems, brainstorming ideas, or planning without executing actions
- **content** (required): Content to think about or a problem to solve
