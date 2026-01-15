"""Context generation for files and directories.

This module provides functionality to generate rich context previews for files and
directories, used when resources are referenced in prompts (e.g., via @ mentions in IDEs).

The goal is to provide agents with useful, token-efficient summaries:
- For directories: Repository maps showing file structure and symbols
- For large files: Outlines/structure maps instead of full content
- For small files: Full content
- For binary files: Metadata only

Future enhancements could include:
- Semantic search to find relevant sections
- Dependency graphs
- Cross-reference analysis
- Custom preview strategies per file type
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from fsspec.asyn import AsyncFileSystem

from agentpool.log import get_logger


logger = get_logger(__name__)

# Token estimation: ~4 chars per token
LARGE_FILE_THRESHOLD = 8192  # ~2000 tokens


async def get_resource_context(
    path: Path,
    fs: AsyncFileSystem | None = None,
    max_files_to_read: int | None = None,
) -> str | None:
    """Get context for a resource (file or directory).

    This is the main entry point for generating context. It routes to the
    appropriate handler based on whether the path is a file or directory.

    Args:
        path: File or directory path
        fs: Optional AsyncFileSystem to use (defaults to LocalFileSystem)
        max_files_to_read: Maximum files to read for directories (None = unlimited)

    Returns:
        Context string if applicable, None if generation fails or path doesn't exist
    """
    if not path.exists():
        logger.debug("Path does not exist", path=str(path))
        return None

    try:
        if path.is_dir():
            return await generate_directory_context(
                path, fs=fs, max_files_to_read=max_files_to_read
            )
        return await generate_file_context(path, fs=fs)
    except OSError as e:
        logger.warning("Failed to generate context", path=str(path), error=str(e))
        return None


async def generate_directory_context(
    path: Path,
    fs: AsyncFileSystem | None = None,
    max_files_to_read: int | None = None,
) -> str | None:
    """Generate a repository map for a directory.

    Creates a hierarchical view of the directory's code structure, showing:
    - File organization
    - Classes, functions, and other symbols
    - Dependencies between files (when available)

    Currently supports: Python, JavaScript, TypeScript, JSX, TSX

    Args:
        path: Directory path
        fs: Optional AsyncFileSystem to use (defaults to LocalFileSystem)
        max_files_to_read: Maximum number of files to read and analyze (None = unlimited)

    Returns:
        Repository map string or None if generation fails
    """
    logger.info("Generating directory context", path=str(path))

    try:
        from fsspec.implementations.local import LocalFileSystem

        from agentpool.repomap import RepoMap

        # Use provided filesystem or fall back to local
        if fs is None:
            fs = LocalFileSystem()
        repo_map = RepoMap(fs, root_path=str(path), max_tokens=4096)

        # Find all source files in the directory (non-recursive for now)
        # TODO: Add recursive option with depth control
        from agentpool.repomap import is_language_supported

        all_files = [
            item for item in path.iterdir() if item.is_file() and is_language_supported(item.name)
        ]

        if not all_files:
            logger.debug("No source files found in directory", path=str(path))
            return f"Directory: {path}\n(No source files found)"

        # Prioritize important files (config files, __init__.py, etc.)
        from agentpool.repomap import is_important

        priority_files = [f for f in all_files if is_important(f.name) or f.name == "__init__.py"]
        other_files = [f for f in all_files if f not in priority_files]

        # Select top N files for detailed analysis
        if max_files_to_read is not None:
            selected_files = (priority_files + other_files)[:max_files_to_read]
            remaining_files = all_files[max_files_to_read:]
        else:
            selected_files = priority_files + other_files
            remaining_files = []

        logger.info(
            "Generating repomap",
            total_files=len(all_files),
            selected=len(selected_files),
            remaining=len(remaining_files),
            path=str(path),
        )

        # Generate the map for selected files
        files_to_analyze = [str(f) for f in selected_files]
        map_content = await repo_map.get_map(files_to_analyze)

        if not map_content:
            logger.warning("Repomap generation returned empty", path=str(path))
            return f"Directory: {path}"

        logger.info(
            "Successfully generated repomap",
            content_length=len(map_content),
            path=str(path),
        )

        # Build output with header
        header = f"# Repository map for {path.name}\n\n"
        if remaining_files:
            header += f"## Detailed structure (analyzed {len(selected_files)} of {len(all_files)} files):\n\n"

        result = header + map_content

        # Append listing of remaining files
        if remaining_files:
            result += "\n\n## Additional files (not analyzed):\n"
            for file in remaining_files:
                try:
                    size_kb = file.stat().st_size / 1024
                    result += f"- {file.name} ({size_kb:.1f} KB)\n"
                except OSError:
                    # If stat fails, just show name
                    result += f"- {file.name}\n"

        return result

    except OSError as e:
        logger.warning("Failed to generate directory context", path=str(path), error=str(e))
        return None


async def generate_file_context(path: Path, fs: AsyncFileSystem | None = None) -> str | None:
    """Generate context for a file (outline or content based on size).

    Strategy:
    1. For binary files: Return metadata only (not implemented yet)
    2. For large text files (>8KB): Generate outline/structure map
    3. For small text files: Return full content
    4. Fallback: Truncated content with notice

    Args:
        path: File path
        fs: Optional AsyncFileSystem to use (defaults to local file reading)

    Returns:
        File context string or None if generation fails
    """
    logger.info("Generating file context", path=str(path))

    try:
        # TODO: Handle binary files with metadata
        # from mimetypes import guess_type
        # mime_type = guess_type(str(path))
        # if is_binary_mime(mime_type):
        #     return generate_binary_file_metadata(path)

        # Read file content
        content = path.read_text(encoding="utf-8", errors="ignore")

        # For large files, try to generate outline
        if len(content) > LARGE_FILE_THRESHOLD:
            logger.info("File is large, generating outline", path=str(path), size=len(content))
            outline = await _generate_file_outline(path, content)
            if outline:
                return outline

            # No outline available, truncate content
            logger.debug("No outline available, truncating", path=str(path))
            from agentpool.repomap import truncate_with_notice

            return truncate_with_notice(str(path), content)

        # Small file, return full content
        logger.debug("File is small, returning full content", path=str(path), size=len(content))
        return content

    except OSError as e:
        logger.warning("Failed to generate file context", path=str(path), error=str(e))
        return None


async def _generate_file_outline(path: Path, content: str) -> str | None:
    """Generate an outline/structure map for a file.

    Uses tree-sitter to parse the file and extract structural information
    like classes, functions, methods, etc.

    Args:
        path: File path (used for language detection)
        content: File content

    Returns:
        Outline string or None if generation fails or language not supported
    """
    try:
        from agentpool.repomap import get_file_map_from_content

        file_map = get_file_map_from_content(str(path), content)
        if file_map:
            logger.info("Generated file outline", path=str(path), outline_length=len(file_map))
            return f"# File outline for {path}\n\n{file_map}"

        return None

    except OSError as e:
        logger.debug("Failed to generate file outline", path=str(path), error=str(e))
        return None


# Future enhancements:

# async def generate_binary_file_metadata(path: Path) -> str:
#     """Generate metadata summary for binary files (images, PDFs, etc.)."""
#     ...

# async def generate_semantic_preview(path: Path, query: str) -> str:
#     """Generate preview focused on sections relevant to a query."""
#     ...

# async def generate_dependency_graph(path: Path) -> str:
#     """Generate dependency/import graph for a file or directory."""
#     ...

# async def generate_cross_references(path: Path, symbol: str) -> str:
#     """Find all references to a symbol across the codebase."""
#     ...
