"""Repository map generation using tree-sitter for code analysis.

Adapted from aider's repomap module with full type annotations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
import colorsys
from dataclasses import dataclass
from importlib import resources
import math
import os
import random
import shutil
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

from upath import UPath


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from diskcache import Cache  # type: ignore[import-untyped]
    import rustworkx as rx


# Important files that should be prioritized in repo map
ROOT_IMPORTANT_FILES: list[str] = [
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "build.gradle",
    "pom.xml",
    "Makefile",
    "CMakeLists.txt",
    "Gemfile",
    "composer.json",
    ".env.example",
    "Dockerfile",
    "docker-compose.yml",
    "README.md",
    "README.rst",
    "README",
]

NORMALIZED_ROOT_IMPORTANT_FILES: set[str] = {str(UPath(path)) for path in ROOT_IMPORTANT_FILES}

# Type aliases
type TokenCounter = Callable[[str], int]


class Tag(NamedTuple):
    """Represents a code tag (definition or reference)."""

    rel_fname: UPath
    fname: UPath
    line: int
    name: str
    kind: str
    end_line: int = -1  # End line for definitions (-1 if unknown)
    signature_end_line: int = -1  # End line for signature (-1 if unknown)


type RankedTag = Tag | tuple[UPath]  # Either full Tag or just (filename,)


@dataclass
class RepoMapResult:
    """Result of repository map generation with metadata."""

    content: str
    total_files_processed: int
    total_tags_found: int
    total_files_with_tags: int
    included_files: int
    included_tags: int
    truncated: bool
    coverage_ratio: float
    io_budget_used: int = 0
    strategy_used: str = "default"


CACHE_VERSION = 4

# Thresholds
MIN_TOKEN_SAMPLE_SIZE: int = 256
LARGE_CACHE_DIFF_THRESHOLD: int = 25
MIN_IDENT_LENGTH: int = 4
MAX_DEFINERS_THRESHOLD: int = 5


def _get_mtime(path: UPath) -> float | None:
    """Get file modification time."""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def is_important(fname: UPath) -> bool:
    """Check if a file is considered important (like config files)."""
    normalized = str(fname)
    if normalized in NORMALIZED_ROOT_IMPORTANT_FILES:
        return True
    # Check basename
    return fname.name in NORMALIZED_ROOT_IMPORTANT_FILES


class RepoMap:
    """Generates a map of a repository's code structure using tree-sitter."""

    TAGS_CACHE_DIR: ClassVar[str] = f".llmling-agent.tags.cache.v{CACHE_VERSION}"

    warned_files: ClassVar[set[UPath]] = set()

    def __init__(
        self,
        root: str | UPath,
        *,
        max_tokens: int = 1024,
        max_line_length: int = 250,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize RepoMap.

        Args:
            root: Root directory of the repository.
            max_tokens: Maximum tokens for the generated map.
            max_line_length: Maximum character length for output lines.
            token_counter: Callable to count tokens. Defaults to len(text) / 4.
        """
        self.root = UPath(root) if not isinstance(root, UPath) else root
        self.max_tokens = max_tokens
        self.max_line_length = max_line_length
        self._token_counter = token_counter

        self._load_tags_cache()

        self.tree_cache: dict[tuple[str, tuple[int, ...], float | None], str] = {}
        self.tree_context_cache: dict[str, dict[str, Any]] = {}
        self.map_cache: dict[tuple[Any, ...], str | None] = {}
        self.map_processing_time: float = 0
        self.last_map: str | None = None
        self.TAGS_CACHE: Cache | dict[str, Any] = {}

    def token_count(self, text: str) -> float:
        """Estimate token count for text."""
        if self._token_counter:
            len_text = len(text)
            if len_text < MIN_TOKEN_SAMPLE_SIZE:
                return self._token_counter(text)

            # Sample for large texts
            lines = text.splitlines(keepends=True)
            num_lines = len(lines)
            step = num_lines // 100 or 1
            sampled_lines = lines[::step]
            sample_text = "".join(sampled_lines)
            sample_tokens = self._token_counter(sample_text)
            return sample_tokens / len(sample_text) * len_text

        # Rough estimate: ~4 chars per token
        return len(text) / 4

    def get_map(
        self,
        files: Sequence[UPath],
        *,
        exclude: set[UPath] | None = None,
        boost_files: set[UPath] | None = None,
        boost_idents: set[str] | None = None,
    ) -> str | None:
        """Generate a repository map for the given files.

        Args:
            files: Files to include in the map.
            exclude: Files to exclude from the map output (but still used for ranking).
            boost_files: Files to boost in ranking.
            boost_idents: Identifiers to boost in ranking.

        Returns:
            The generated repository map as a string, or None if no map could be generated.
        """
        if not files:
            return None

        exclude = exclude or set()
        boost_files = boost_files or set()
        boost_idents = boost_idents or set()

        return self._get_ranked_tags_map(
            files=files,
            exclude=exclude,
            boost_files=boost_files,
            boost_idents=boost_idents,
        )

    def get_map_with_metadata(
        self,
        files: Sequence[UPath],
        *,
        exclude: set[UPath] | None = None,
        boost_files: set[UPath] | None = None,
        boost_idents: set[str] | None = None,
    ) -> RepoMapResult:
        """Generate a repository map with detailed metadata about what was included/excluded.

        Args:
            files: Files to include in the map.
            exclude: Files to exclude from the map output (but still used for ranking).
            boost_files: Files to boost in ranking.
            boost_idents: Identifiers to boost in ranking.

        Returns:
            RepoMapResult with the map content and metadata about truncation.
        """
        import re

        if not files:
            return RepoMapResult(
                content="",
                total_files_processed=0,
                total_tags_found=0,
                total_files_with_tags=0,
                included_files=0,
                included_tags=0,
                truncated=False,
                coverage_ratio=0.0,
            )

        exclude = exclude or set()
        boost_files = boost_files or set()
        boost_idents = boost_idents or set()

        # Get all ranked tags to calculate totals
        ranked_tags = self._get_ranked_tags(
            files=files,
            exclude=exclude,
            boost_files=boost_files,
            boost_idents=boost_idents,
        )

        total_tags = len([tag for tag in ranked_tags if isinstance(tag, Tag)])
        all_files_with_tags = {tag.fname if isinstance(tag, Tag) else tag[0] for tag in ranked_tags}
        total_files_with_tags = len(all_files_with_tags)

        # Generate the actual map content
        content = self._get_ranked_tags_map(
            files=files,
            exclude=exclude,
            boost_files=boost_files,
            boost_idents=boost_idents,
        )

        # Count what made it into final output
        if content:
            # Count files that have content (lines ending with :)
            included_files = len(set(re.findall(r"^([^:\s]+):", content, re.MULTILINE)))
            # Count function/class definitions
            included_tags = content.count(" def ") + content.count("class ")
        else:
            included_files = included_tags = 0

        coverage_ratio = (
            included_files / total_files_with_tags if total_files_with_tags > 0 else 0.0
        )
        truncated = included_files < total_files_with_tags or included_tags < total_tags

        return RepoMapResult(
            content=content or "",
            total_files_processed=len(files),
            total_tags_found=total_tags,
            total_files_with_tags=total_files_with_tags,
            included_files=included_files,
            included_tags=included_tags,
            truncated=truncated,
            coverage_ratio=coverage_ratio,
        )

    def get_rel_fname(self, fname: UPath) -> UPath:
        """Get relative filename from root."""
        try:
            return fname.relative_to(self.root)
        except ValueError:
            # ValueError: path is on mount 'C:', start on mount 'D:'
            return fname

    def _tags_cache_error(self, original_error: Exception | None = None) -> None:
        """Handle SQLite errors by trying to recreate cache, falling back to dict."""
        import sqlite3

        from diskcache import Cache

        if isinstance(self.TAGS_CACHE, dict):
            return

        path = self.root / self.TAGS_CACHE_DIR

        try:
            if path.exists():
                shutil.rmtree(str(path))

            new_cache = Cache(path)

            # Test that it works
            test_key = "test"
            new_cache[test_key] = "test"
            _ = new_cache[test_key]
            del new_cache[test_key]

            self.TAGS_CACHE = new_cache
        except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError):
            self.TAGS_CACHE = {}

    def _load_tags_cache(self) -> None:
        """Load the tags cache from disk."""
        import sqlite3

        from diskcache import Cache

        path = self.root / self.TAGS_CACHE_DIR
        try:
            self.TAGS_CACHE = Cache(str(path))
        except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
            self._tags_cache_error(e)

    def _get_tags(self, fname: UPath, rel_fname: UPath) -> list[Tag]:
        """Get tags for a file, using cache when possible."""
        import sqlite3

        file_mtime = _get_mtime(fname)
        if file_mtime is None:
            return []

        cache_key = str(fname)
        val: dict[str, Any] | None = None
        try:
            val = self.TAGS_CACHE.get(cache_key)
        except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
            self._tags_cache_error(e)
            val = self.TAGS_CACHE.get(cache_key)

        if val is not None and val.get("mtime") == file_mtime:
            try:
                return cast(list[Tag], self.TAGS_CACHE[cache_key]["data"])
            except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
                self._tags_cache_error(e)
                return cast(list[Tag], self.TAGS_CACHE[cache_key]["data"])

        # Cache miss
        data = list(self._get_tags_raw(fname, rel_fname))

        try:
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}
        except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
            self._tags_cache_error(e)
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}

        return data

    def _get_tags_raw(self, fname: UPath, rel_fname: UPath) -> Iterator[Tag]:  # noqa: PLR0911
        """Extract tags from a file using tree-sitter."""
        from grep_ast import filename_to_lang  # type: ignore[import-untyped]
        from grep_ast.tsl import get_language, get_parser  # type: ignore[import-untyped]
        from pygments.lexers import guess_lexer_for_filename
        from pygments.token import Token
        from tree_sitter import Query, QueryCursor

        lang = filename_to_lang(str(fname))
        if not lang:
            return

        try:
            language = get_language(lang)  # pyright: ignore[reportArgumentType]
            parser = get_parser(lang)  # pyright: ignore[reportArgumentType]
        except Exception:  # noqa: BLE001
            return

        query_scm = get_scm_fname(lang)
        if not query_scm or not query_scm.exists():
            return
        query_scm_text = query_scm.read_text("utf-8")

        try:
            code = fname.read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            return
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))
        query = Query(language, query_scm_text)
        cursor = QueryCursor(query)
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
            # For definitions, get the end line from the parent node (the full definition)
            end_line = -1
            signature_end_line = -1
            if kind == "def" and node.parent is not None:
                end_line = node.parent.end_point[0]
                # Find signature end by looking for block/body child
                for child in node.parent.children:
                    if child.type in ("block", "body", "compound_statement"):
                        # Signature ends on line before block starts
                        signature_end_line = child.start_point[0] - 1
                        break
                # Fallback: use parent's start line if no block found
                signature_end_line = max(signature_end_line, line)
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=name,
                kind=kind,
                line=line,
                end_line=end_line,
                signature_end_line=signature_end_line,
            )

        if "ref" in saw or "def" in saw:
            return

        try:  # We saw defs without refs - use pygments to backfill refs
            lexer = guess_lexer_for_filename(str(fname), code)
        except Exception:  # noqa: BLE001
            return

        tokens = list(lexer.get_tokens(code))
        name_tokens = [token[1] for token in tokens if token[0] in Token.Name]  # type: ignore[comparison-overlap]
        for token in name_tokens:
            yield Tag(rel_fname=rel_fname, fname=fname, name=token, kind="ref", line=-1)

    def _get_ranked_tags(  # noqa: PLR0915
        self,
        files: Sequence[UPath],
        exclude: set[UPath],
        boost_files: set[UPath],
        boost_idents: set[str],
    ) -> list[RankedTag]:
        """Rank tags using PageRank algorithm."""
        import rustworkx as rx

        defines: defaultdict[str, set[str]] = defaultdict(set)
        references: defaultdict[str, list[str]] = defaultdict(list)
        definitions: defaultdict[tuple[str, str], set[Tag]] = defaultdict(set)
        personalization: dict[str, float] = {}
        exclude_rel_fnames: set[str] = set()
        # Map string keys back to original UPath objects (preserves auth, protocol, etc.)
        str_to_rel_path: dict[str, UPath] = {}
        sorted_fnames = sorted(files, key=str)
        # Default personalization for unspecified files
        personalize = 100 / len(sorted_fnames) if sorted_fnames else 0
        fnames_iter: Iterable[UPath] = sorted_fnames
        for fname in fnames_iter:
            try:
                file_ok = fname.is_file()
            except OSError:
                file_ok = False
            if not file_ok:
                if fname not in self.warned_files:
                    self.warned_files.add(fname)
                continue
            rel_fname = self.get_rel_fname(fname)
            rel_fname_str = str(rel_fname)
            str_to_rel_path[rel_fname_str] = rel_fname
            current_pers = 0.0
            if fname in exclude:
                current_pers += personalize
                exclude_rel_fnames.add(rel_fname_str)
            if rel_fname in boost_files:
                current_pers = max(current_pers, personalize)
            # Check path components against boost_idents
            path_components = set(rel_fname.parts)
            basename_with_ext = rel_fname.name
            basename_without_ext = rel_fname.stem
            components_to_check = path_components.union({basename_with_ext, basename_without_ext})

            matched_idents = components_to_check.intersection(boost_idents)
            if matched_idents:
                current_pers += personalize

            if current_pers > 0:
                personalization[rel_fname_str] = current_pers
            tags = list(self._get_tags(fname, rel_fname))
            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname_str)
                    key = (rel_fname_str, tag.name)
                    definitions[key].add(tag)
                elif tag.kind == "ref":
                    references[tag.name].append(rel_fname_str)

        if not references:
            references = defaultdict(list, {k: list(v) for k, v in defines.items()})

        idents = set(defines.keys()).intersection(set(references.keys()))

        # Build graph using rustworkx - map string node names to indices
        graph: rx.PyDiGraph[str, dict[str, Any]] = rx.PyDiGraph(multigraph=True)
        node_to_idx: dict[str, int] = {}
        idx_to_node: dict[int, str] = {}

        def get_or_add_node(name: str) -> int:
            if name not in node_to_idx:
                idx = graph.add_node(name)
                node_to_idx[name] = idx
                idx_to_node[idx] = name
            return node_to_idx[name]

        # Add self-edges for definitions without references
        for ident in defines:
            if ident in references:
                continue
            for definer in defines[ident]:
                idx = get_or_add_node(definer)
                graph.add_edge(idx, idx, {"weight": 0.1, "ident": ident})

        for ident in idents:
            definers = defines[ident]
            mul = 1.0
            is_snake = ("_" in ident) and any(c.isalpha() for c in ident)
            is_kebab = ("-" in ident) and any(c.isalpha() for c in ident)
            is_camel = any(c.isupper() for c in ident) and any(c.islower() for c in ident)
            if ident in boost_idents:
                mul *= 10
            if (is_snake or is_kebab or is_camel) and len(ident) >= MIN_IDENT_LENGTH:
                mul *= 10
            if ident.startswith("_"):
                mul *= 0.1
            if len(defines[ident]) > MAX_DEFINERS_THRESHOLD:
                mul *= 0.1

            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    use_mul = mul
                    if referencer in exclude_rel_fnames:
                        use_mul *= 50
                    # Scale down high frequency mentions
                    scaled_refs = math.sqrt(num_refs)
                    src_idx = get_or_add_node(referencer)
                    dst_idx = get_or_add_node(definer)
                    graph.add_edge(
                        src_idx, dst_idx, {"weight": use_mul * scaled_refs, "ident": ident}
                    )

        if not graph.num_nodes():
            return []

        # Build personalization dict with node indices
        pers_idx: dict[int, float] | None = None
        if personalization:
            pers_idx = {
                node_to_idx[name]: val
                for name, val in personalization.items()
                if name in node_to_idx
            }

        try:
            ranked_idx = rx.pagerank(
                graph,
                weight_fn=lambda e: e["weight"],
                personalization=pers_idx,
                dangling=pers_idx,
            )
        except ZeroDivisionError:
            try:
                ranked_idx = rx.pagerank(graph, weight_fn=lambda e: e["weight"])
            except ZeroDivisionError:
                return []

        # Convert back to string keys
        ranked: dict[str, float] = {idx_to_node[idx]: rank for idx, rank in ranked_idx.items()}

        # Distribute rank across out edges
        ranked_definitions: defaultdict[tuple[str, str], float] = defaultdict(float)
        for src_idx in graph.node_indices():
            src_name = idx_to_node[src_idx]
            src_rank = ranked[src_name]
            out_edges = graph.out_edges(src_idx)  # WeightedEdgeList: list of (src, dst, data)
            total_weight = sum(edge_data["weight"] for _, _, edge_data in out_edges)
            if total_weight == 0:
                continue
            for _, dst_idx, edge_data in out_edges:
                edge_rank = src_rank * edge_data["weight"] / total_weight
                ident = edge_data["ident"]
                dst_name = idx_to_node[dst_idx]
                ranked_definitions[(dst_name, ident)] += edge_rank

        ranked_tags: list[RankedTag] = []
        sorted_definitions = sorted(
            ranked_definitions.items(), reverse=True, key=lambda x: (x[1], x[0])
        )

        for (fname, ident), _rank in sorted_definitions:
            if fname in exclude_rel_fnames:
                continue
            ranked_tags += list(definitions.get((fname, ident), []))

        rel_fnames_without_tags = {str(self.get_rel_fname(fname)) for fname in files}
        for fname in exclude:
            rel = str(self.get_rel_fname(fname))
            rel_fnames_without_tags.discard(rel)

        fnames_already_included = {str(rt[0]) for rt in ranked_tags}

        top_rank = sorted([(rank, node) for (node, rank) in ranked.items()], reverse=True)
        for _rank, fname_str in top_rank:
            if fname_str in rel_fnames_without_tags:
                rel_fnames_without_tags.remove(fname_str)
            if fname_str not in fnames_already_included:
                # Use original UPath from mapping, preserving auth/protocol info
                original_path = str_to_rel_path.get(fname_str)
                if original_path is not None:
                    ranked_tags.append((original_path,))

        for fname_str in rel_fnames_without_tags:
            original_path = str_to_rel_path.get(fname_str)
            if original_path is not None:
                ranked_tags.append((original_path,))

        return ranked_tags

    def _get_ranked_tags_map(
        self,
        files: Sequence[UPath],
        exclude: set[UPath],
        boost_files: set[UPath],
        boost_idents: set[str],
        max_tokens: int | None = None,
    ) -> str | None:
        """Generate a ranked tags map."""
        if not max_tokens:
            max_tokens = self.max_tokens

        ranked_tags = self._get_ranked_tags(
            files=files,
            exclude=exclude,
            boost_files=boost_files,
            boost_idents=boost_idents,
        )

        rel_fnames = sorted({self.get_rel_fname(fname) for fname in files})
        special_fnames = [fname for fname in rel_fnames if is_important(fname)]
        ranked_tags_fnames = {tag[0] for tag in ranked_tags}
        special_fnames = [fn for fn in special_fnames if fn not in ranked_tags_fnames]
        special_tags: list[RankedTag] = [(fn,) for fn in special_fnames]
        ranked_tags = special_tags + ranked_tags
        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree: str | None = None
        best_tree_tokens: float = 0
        exclude_rel_fnames = {str(self.get_rel_fname(fname)) for fname in exclude}
        self.tree_cache = {}
        middle = min(int(max_tokens // 25), num_tags)
        while lower_bound <= upper_bound:
            tree = self._to_tree(ranked_tags[:middle], exclude_rel_fnames)
            num_tokens = self.token_count(tree)
            pct_err = abs(num_tokens - max_tokens) / max_tokens if max_tokens else 0
            ok_err = 0.15
            if (num_tokens <= max_tokens and num_tokens > best_tree_tokens) or pct_err < ok_err:
                best_tree = tree
                best_tree_tokens = num_tokens
                if pct_err < ok_err:
                    break

            if num_tokens < max_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = (lower_bound + upper_bound) // 2

        return best_tree

    def _render_tree(  # noqa: PLR0915
        self,
        abs_fname: UPath,
        rel_fname: UPath,
        lois: list[int],
        line_ranges: dict[int, int] | None = None,
    ) -> str:
        """Render a tree representation of a file with lines of interest.

        The output includes 1-based line number ranges for definitions.

        Args:
            abs_fname: Absolute path to the file.
            rel_fname: Relative path to the file.
            lois: List of lines of interest (0-indexed).
            line_ranges: Optional dict mapping start line to end line (0-indexed).
        """
        import re

        from grep_ast import TreeContext

        if line_ranges is None:
            line_ranges = {}

        mtime = _get_mtime(abs_fname)
        key = (str(rel_fname), tuple(sorted(lois)), mtime)
        if key in self.tree_cache:
            return self.tree_cache[key]
        cached = self.tree_context_cache.get(str(rel_fname))
        if cached is None or cached["mtime"] != mtime:
            try:
                code = abs_fname.read_text("utf-8")
            except (OSError, UnicodeDecodeError):
                code = ""
            if not code.endswith("\n"):
                code += "\n"

            context = TreeContext(
                str(rel_fname),
                code,
                child_context=False,
                last_line=False,
                margin=0,
                mark_lois=False,
                loi_pad=0,
                show_top_of_file_parent_scope=False,
            )
            self.tree_context_cache[str(rel_fname)] = {"context": context, "mtime": mtime}

        context = self.tree_context_cache[str(rel_fname)]["context"]
        context.lines_of_interest = set()
        context.add_lines_of_interest(lois)
        context.add_context()
        res: str = context.format()

        # Add line numbers/ranges to class/function/method definitions
        try:
            code = abs_fname.read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            code = ""
        code_lines = code.splitlines()
        lois_set = set(lois)

        # Pattern to match class/function definitions
        def_pattern = re.compile(r"^(.*?)(class\s+\w+|def\s+\w+|async\s+def\s+\w+)")

        result_lines = []
        for output_line in res.splitlines():
            modified_line = output_line
            # Check if this line contains a class or function definition
            match = def_pattern.search(output_line)
            if match:
                # Find matching line number from lois
                # Strip leading markers (│, etc.) to get the actual code
                stripped = output_line.lstrip("│ \t")
                for line_num in lois_set:
                    if line_num < len(code_lines):
                        orig_line = code_lines[line_num].strip()
                        if orig_line and stripped.startswith(orig_line.split("(")[0].split(":")[0]):
                            # Find the name part and add line range at end of line
                            name_match = re.search(
                                r"(class\s+\w+|def\s+\w+|async\s+def\s+\w+)", output_line
                            )
                            if name_match:
                                # line_num is 0-indexed, display as 1-indexed
                                start_line_display = line_num + 1
                                end_line = line_ranges.get(line_num, -1)
                                if end_line >= 0 and end_line != line_num:
                                    # Display as range
                                    end_line_display = end_line + 1
                                    line_info = f"  # [{start_line_display}-{end_line_display}]"
                                else:
                                    # Single line or unknown end
                                    line_info = f"  # [{start_line_display}]"
                                # Append line info at the end of the line
                                modified_line = f"{output_line}{line_info}"
                            break
            result_lines.append(modified_line)

        res = "\n".join(result_lines)
        if result_lines:
            res += "\n"
        self.tree_cache[key] = res
        return res

    def _to_tree(self, tags: list[RankedTag], exclude_rel_fnames: set[str]) -> str:
        """Convert ranked tags to a tree representation."""
        if not tags:
            return ""

        cur_fname: UPath | None = None
        cur_abs_fname: UPath | None = None
        lois: list[int] | None = None
        line_ranges: dict[int, int] | None = None
        output = ""

        # Add bogus tag to trigger final output
        dummy_tag: tuple[None] = (None,)
        for tag in [*sorted(tags, key=lambda t: str(t[0]) if t[0] else ""), dummy_tag]:
            this_rel_fname = tag[0]
            if str(this_rel_fname) in exclude_rel_fnames:
                continue

            if this_rel_fname != cur_fname:
                if lois is not None and cur_fname and cur_abs_fname:
                    output += "\n"
                    output += str(cur_fname) + ":\n"
                    output += self._render_tree(cur_abs_fname, cur_fname, lois, line_ranges)
                    lois = None
                    line_ranges = None
                elif cur_fname:
                    output += "\n" + str(cur_fname) + "\n"
                if isinstance(tag, Tag):
                    lois = []
                    line_ranges = {}
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None and line_ranges is not None and isinstance(tag, Tag):
                # Add all lines from start to signature end as lines of interest
                if tag.signature_end_line >= tag.line:
                    lois.extend(range(tag.line, tag.signature_end_line + 1))
                else:
                    lois.append(tag.line)
                if tag.end_line >= 0:
                    line_ranges[tag.line] = tag.end_line

        # Truncate long lines
        return "\n".join([line[: self.max_line_length] for line in output.splitlines()]) + "\n"


def find_src_files(directory: str | UPath) -> list[UPath]:
    """Find all source files in a directory."""
    directory_path = UPath(directory)
    if not directory_path.is_dir():
        return [directory_path]

    return [file_path for file_path in directory_path.rglob("*") if file_path.is_file()]


def get_random_color() -> str:
    """Generate a random color in hex format."""
    hue = random.random()
    r, g, b = (int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75))
    return f"#{r:02x}{g:02x}{b:02x}"


def get_scm_fname(lang: str) -> UPath | None:
    """Get the path to the SCM query file for a language."""
    package = __package__ or "llmling_agent"
    subdir = "tree-sitter-language-pack"
    try:
        path = resources.files(package).joinpath("queries", subdir, f"{lang}-tags.scm")
        if path.is_file():
            return UPath(str(path))
    except KeyError:
        pass

    # Fall back to tree-sitter-languages
    subdir = "tree-sitter-languages"
    try:
        path = resources.files(package).joinpath("queries", subdir, f"{lang}-tags.scm")
        return UPath(str(path))
    except KeyError:
        return None


def get_supported_languages_md() -> str:
    """Generate markdown table of supported languages."""
    from grep_ast.parsers import PARSERS  # type: ignore[import-untyped]

    res = """
| Language | File extension | Repo map |
|:--------:|:--------------:|:--------:|
"""
    data = sorted((lang, ext) for ext, lang in PARSERS.items())
    for lang, ext in data:
        fn = get_scm_fname(lang)
        repo_map = "✓" if fn and fn.exists() else ""
        res += f"| {lang:20} | {ext:20} | {repo_map:^8} |\n"

    res += "\n"
    return res


if __name__ == "__main__":
    # Test with GitHub UPath using GH_TOKEN for authentication
    print("=" * 80)
    print("GitHub UPath Test - epregistry repository")
    print("=" * 80)

    gh_token = os.environ.get("GH_TOKEN")
    if not gh_token:
        print("Warning: GH_TOKEN not set, may hit rate limits")

    github_repo = UPath(
        "github://phil65:epregistry@main/src/epregistry",
        username=gh_token,
        token=gh_token,
    )
    repo_root = UPath(
        "github://phil65:epregistry@main",
        username=gh_token,
        token=gh_token,
    )

    print(f"Repository: {github_repo}")

    try:
        all_py_files = [f for f in find_src_files(github_repo) if f.suffix == ".py"]
        print(f"Found {len(all_py_files)} Python files")

        rm = RepoMap(root=repo_root, max_tokens=4000)
        result = rm.get_map_with_metadata(all_py_files)

        print("\n" + "=" * 80)
        print("REPOSITORY MAP METADATA")
        print("=" * 80)
        print(f"Total files processed: {result.total_files_processed}")
        print(f"Total tags found: {result.total_tags_found}")
        print(f"Files with tags: {result.total_files_with_tags}")
        print(f"Included files: {result.included_files}")
        print(f"Included tags: {result.included_tags}")
        print(f"Truncated: {result.truncated}")
        print(f"Coverage ratio: {result.coverage_ratio:.2%}")

        if result.content:
            print("\n" + "=" * 80)
            print("REPOSITORY MAP")
            print("=" * 80)
            print(result.content)
        else:
            print("No repository map generated")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
