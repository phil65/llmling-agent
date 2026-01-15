"""Language support utilities for tree-sitter."""

from __future__ import annotations

from pathlib import Path


def get_scm_fname(lang: str) -> Path | None:
    """Get path to tree-sitter query file for a language.

    Args:
        lang: Language identifier (e.g. "python", "javascript")

    Returns:
        Path to .scm query file, or None if not found
    """
    from importlib import resources

    fname = f"{lang}-tags.scm"
    ref = resources.files("agentpool") / "queries" / "tree-sitter-language-pack" / fname

    # Try to get the path
    if hasattr(ref, "__fspath__"):
        path = Path(ref.__fspath__())
        return path if path.exists() else None

    # Fallback for older Python or packaged resources
    try:
        with resources.as_file(ref) as path:
            return path if path.exists() else None
    except (FileNotFoundError, AttributeError):
        return None


def get_supported_languages() -> set[str]:
    """Get set of languages that have tree-sitter tag support.

    Returns:
        Set of language identifiers that have query files
    """
    from grep_ast.parsers import PARSERS  # type: ignore[import-untyped]

    supported = set()
    for lang in set(PARSERS.values()):
        scm = get_scm_fname(lang)
        if scm and scm.exists():
            supported.add(lang)
    return supported


def is_language_supported(fname: str) -> bool:
    """Check if a file's language supports tree-sitter tags.

    Args:
        fname: Filename (e.g. "foo.py", "main.rs")

    Returns:
        True if language is supported for tag extraction
    """
    from grep_ast import filename_to_lang  # type: ignore[import-untyped]

    lang = filename_to_lang(fname)
    if not lang:
        return False
    scm = get_scm_fname(lang)
    return scm is not None and scm.exists()


def get_supported_languages_md() -> str:
    """Generate markdown table of supported languages."""
    from grep_ast.parsers import PARSERS

    supported = get_supported_languages()

    # Group by language
    lang_exts: dict[str, list[str]] = {}
    for ext, lang in PARSERS.items():
        if lang in supported:
            lang_exts.setdefault(lang, []).append(ext)

    # Build markdown table
    lines = ["| Language | Extensions |", "|----------|------------|"]
    for lang in sorted(lang_exts):
        exts = ", ".join(f"`{e}`" for e in sorted(lang_exts[lang]))
        lines.append(f"| {lang} | {exts} |")

    return "\n".join(lines)
