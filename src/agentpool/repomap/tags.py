"""Tag extraction using tree-sitter."""

from __future__ import annotations

from typing import Any, NamedTuple, cast


class Tag(NamedTuple):
    """Represents a code tag (definition or reference)."""

    rel_fname: str
    fname: str
    line: int
    name: str
    kind: str
    end_line: int = -1
    signature_end_line: int = -1


type RankedTag = Tag | tuple[str]


def get_tags_from_content(content: str, filename: str) -> list[Tag]:
    """Extract tags from content without filesystem IO.

    Uses tree-sitter to parse the file and extract definitions and references.
    Falls back to Pygments lexer for reference tokens if tree-sitter queries
    don't capture them.

    Args:
        content: File content as string
        filename: Filename for language detection (e.g. "foo.py")

    Returns:
        List of Tag objects (definitions and references)
    """
    from grep_ast import filename_to_lang  # type: ignore[import-untyped]
    from grep_ast.tsl import get_language, get_parser  # type: ignore[import-untyped]
    from pygments.lexers import guess_lexer_for_filename
    from pygments.token import Token
    from tree_sitter import Query, QueryCursor
    from tree_sitter_language_pack import SupportedLanguage

    from agentpool.repomap.languages import get_scm_fname

    lang = cast(SupportedLanguage, filename_to_lang(filename))
    if not lang:
        return []

    try:
        language = get_language(lang)
        parser = get_parser(lang)
    except Exception:  # noqa: BLE001
        return []

    query_scm = get_scm_fname(lang)
    if not query_scm or not query_scm.exists():
        return []
    query_scm_text = query_scm.read_text("utf-8")

    tree = parser.parse(bytes(content, "utf-8"))
    query = Query(language, query_scm_text)
    cursor = QueryCursor(query)

    tags: list[Tag] = []
    saw: set[str] = set()
    all_nodes: list[tuple[Any, str]] = []

    for _pattern_index, captures_dict in cursor.matches(tree.root_node):
        for tag, nodes in captures_dict.items():
            all_nodes.extend((node, tag) for node in nodes)

    for node, tag in all_nodes:
        if tag.startswith("name.definition."):
            kind = "def"
        elif tag.startswith("name.reference."):
            kind = "ref"
        else:
            continue

        saw.add(kind)
        name = node.text.decode("utf-8")
        line = node.start_point[0]

        end_line = -1
        signature_end_line = -1
        if kind == "def" and node.parent is not None:
            end_line = node.parent.end_point[0]
            for child in node.parent.children:
                if child.type in ("block", "body", "compound_statement"):
                    signature_end_line = child.start_point[0] - 1
                    break
            signature_end_line = max(signature_end_line, line)

        tags.append(
            Tag(
                rel_fname=filename,
                fname=filename,
                name=name,
                kind=kind,
                line=line,
                end_line=end_line,
                signature_end_line=signature_end_line,
            )
        )

    if "ref" in saw or "def" in saw:
        return tags

    # Fallback to pygments lexer for references
    try:
        lexer = guess_lexer_for_filename(filename, content)
    except Exception:  # noqa: BLE001
        return tags

    tokens = list(lexer.get_tokens(content))
    name_tokens = [token[1] for token in tokens if token[0] in Token.Name]  # type: ignore[comparison-overlap]

    for token in name_tokens:
        tags.append(Tag(rel_fname=filename, fname=filename, name=token, kind="ref", line=-1))  # noqa: PERF401

    return tags
