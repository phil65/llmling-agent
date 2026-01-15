"""Core RepoMap class for generating repository structure maps.

This module contains the main RepoMap class that uses tree-sitter and PageRank
to generate intelligent code structure maps.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, cast

from agentpool.repomap.tags import Tag, get_tags_from_content
from agentpool.repomap.utils import (
    MIN_TOKEN_SAMPLE_SIZE,
    get_rel_path,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from fsspec import AbstractFileSystem

    from agentpool.repomap.types import FileInfo, RepoMapResult, TokenCounter

# Constants
CACHE_VERSION = 3
MIN_IDENT_LENGTH = 5
MAX_DEFINERS_THRESHOLD = 6

RankedTag = Tag | tuple[str] | tuple[None]


class RepoMap:
    """Generates a map of a repository's code structure using tree-sitter.

    Uses async fsspec filesystem for non-blocking IO operations.
    """

    TAGS_CACHE_DIR: ClassVar[str] = f".agentpool.tags.cache.v{CACHE_VERSION}"
    warned_files: ClassVar[set[str]] = set()

    def __init__(
        self,
        fs: AbstractFileSystem,
        root_path: str | None = None,
        *,
        max_tokens: int = 1024,
        max_line_length: int = 250,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize RepoMap.

        Args:
            fs: Async fsspec filesystem instance.
            root_path: Root directory path in the filesystem.
            max_tokens: Maximum tokens for the generated map.
            max_line_length: Maximum character length for output lines.
            token_counter: Callable to count tokens. Defaults to len(text) / 4.
        """
        from fsspec.asyn import AsyncFileSystem
        from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

        self.fs = fs if isinstance(fs, AsyncFileSystem) else AsyncFileSystemWrapper(fs)
        self.root_path = root_path.rstrip("/") if root_path else self.fs.root_marker
        self.max_tokens = max_tokens
        self.max_line_length = max_line_length
        self._token_counter = token_counter

        self.tree_cache: dict[tuple[str, tuple[int, ...], float | None], str] = {}
        self.tree_context_cache: dict[str, dict[str, Any]] = {}
        self.TAGS_CACHE: dict[str, Any] = {}

    def token_count(self, text: str) -> float:
        """Estimate token count for text."""
        if self._token_counter:
            len_text = len(text)
            if len_text < MIN_TOKEN_SAMPLE_SIZE:
                return self._token_counter(text)

            lines = text.splitlines(keepends=True)
            num_lines = len(lines)
            step = num_lines // 100 or 1
            sampled_lines = lines[::step]
            sample_text = "".join(sampled_lines)
            sample_tokens = self._token_counter(sample_text)
            return sample_tokens / len(sample_text) * len_text

        return len(text) / 4

    async def _cat_file(self, path: str) -> str | None:
        """Read file content as text."""
        try:
            content = await self.fs._cat_file(path)
            if isinstance(content, bytes):
                return content.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        else:
            return content  # type: ignore[no-any-return]

    async def _info(self, path: str) -> FileInfo | None:
        """Get file info."""
        from agentpool.repomap.types import FileInfo

        try:
            info = await self.fs._info(path)
            return FileInfo(
                path=info.get("name", path),
                size=info.get("size", 0),
                mtime=info.get("mtime"),
                type=info.get("type", "file"),
            )
        except (OSError, FileNotFoundError):
            return None

    async def _ls(self, path: str, detail: bool = True) -> list[dict[str, Any]]:
        """List directory contents."""
        try:
            return await self.fs._ls(path, detail=detail)  # type: ignore[no-any-return]
        except (OSError, FileNotFoundError):
            return []

    async def find_files(self, path: str, pattern: str = "**/*.py") -> list[str]:
        """Find files matching pattern recursively."""
        from agentpool.repomap.rendering import is_directory

        results: list[str] = []

        async def _recurse(current_path: str) -> None:
            entries = await self._ls(current_path, detail=True)
            for entry in entries:
                entry_path = entry.get("name", "")
                entry_type = entry.get("type", "")

                if await is_directory(self.fs, entry_path, entry_type=entry_type):
                    await _recurse(entry_path)
                # It's a file - process it
                elif pattern == "**/*.py":
                    if entry_path.endswith(".py"):
                        results.append(entry_path)
                else:
                    results.append(entry_path)

        await _recurse(path)
        return results

    async def get_file_map(
        self,
        fname: str,
        max_tokens: int = 2048,
    ) -> str | None:
        """Generate a structure map for a single file.

        Unlike get_map which uses PageRank across multiple files, this method
        shows all definitions in a single file with line numbers.

        Args:
            fname: Absolute path to the file
            max_tokens: Maximum tokens for output (approximate)

        Returns:
            Formatted structure map or None if no tags found
        """
        rel_fname = get_rel_path(fname, self.root_path)

        # Get all definition tags for this file
        tags = await self._get_tags(fname, rel_fname)
        def_tags = [t for t in tags if t.kind == "def"]

        if not def_tags:
            return None

        # Build line ranges for rendering
        lois: list[int] = []
        line_ranges: dict[int, int] = {}

        for tag in def_tags:
            if tag.signature_end_line >= tag.line:
                lois.extend(range(tag.line, tag.signature_end_line + 1))
            else:
                lois.append(tag.line)
            if tag.end_line >= 0:
                line_ranges[tag.line] = tag.end_line

        # Render the tree
        tree_output = await self._render_tree(fname, rel_fname, lois, line_ranges)

        # Add header with file info
        info = await self._info(fname)
        size_info = f", {info.size} bytes" if info else ""
        lines = (await self._cat_file(fname) or "").count("\n") + 1
        tokens = self.token_count(tree_output)

        header = (
            f"# File: {rel_fname} ({lines} lines{size_info})\n"
            f"# Structure map ({tokens} tokens). Use offset/limit to read sections.\n\n"
        )

        result = header + f"{rel_fname}:\n" + tree_output

        # Truncate if needed
        max_chars = max_tokens * 4
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... [truncated]\n"

        return result

    async def get_map(
        self,
        files: Sequence[str],
        *,
        exclude: set[str] | None = None,
        boost_files: set[str] | None = None,
        boost_idents: set[str] | None = None,
    ) -> str | None:
        """Generate a repository map for the given files.

        Args:
            files: File paths to include in the map.
            exclude: Files to exclude from the map output (but still used for ranking).
            boost_files: Files to boost in ranking.
            boost_idents: Identifiers to boost in ranking.
        """
        if not files:
            return None

        exclude = exclude or set()
        boost_files = boost_files or set()
        boost_idents = boost_idents or set()

        return await self._get_ranked_tags_map(
            files=files,
            exclude=exclude,
            boost_files=boost_files,
            boost_idents=boost_idents,
        )

    async def get_map_with_metadata(
        self,
        files: Sequence[str],
        *,
        exclude: set[str] | None = None,
        boost_files: set[str] | None = None,
        boost_idents: set[str] | None = None,
    ) -> RepoMapResult:
        """Generate a repository map with detailed metadata.

        Args:
            files: File paths to include in the map.
            exclude: Files to exclude from the map output (but still used for ranking).
            boost_files: Files to boost in ranking.
            boost_idents: Identifiers to boost in ranking.
        """
        import re

        from agentpool.repomap.types import RepoMapResult

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

        ranked_tags = await self._get_ranked_tags(
            files=files,
            exclude=exclude,
            boost_files=boost_files,
            boost_idents=boost_idents,
        )

        total_tags = len([tag for tag in ranked_tags if isinstance(tag, Tag)])
        all_files_with_tags = {tag.fname if isinstance(tag, Tag) else tag[0] for tag in ranked_tags}
        total_files_with_tags = len(all_files_with_tags)

        content = await self._get_ranked_tags_map(
            files=files,
            exclude=exclude,
            boost_files=boost_files,
            boost_idents=boost_idents,
        )

        if content:
            included_files = len(set(re.findall(r"^([^:\s]+):", content, re.MULTILINE)))
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

    async def _get_tags(self, fname: str, rel_fname: str) -> list[Tag]:
        """Get tags for a file, using cache when possible."""
        info = await self._info(fname)
        if info is None:
            return []

        file_mtime = info.mtime
        cache_key = fname

        cached = self.TAGS_CACHE.get(cache_key)
        if cached is not None and cached.get("mtime") == file_mtime:
            return cast(list[Tag], cached["data"])

        data = await self._get_tags_raw(fname, rel_fname)

        self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}
        return data

    async def _get_tags_raw(self, fname: str, rel_fname: str) -> list[Tag]:
        """Extract tags from a file using tree-sitter."""
        content = await self._cat_file(fname)
        if not content:
            return []

        return get_tags_from_content(content, fname)

    async def _get_ranked_tags(  # noqa: PLR0915
        self,
        files: Sequence[str],
        exclude: set[str],
        boost_files: set[str],
        boost_idents: set[str],
    ) -> list[RankedTag]:
        """Rank tags using PageRank algorithm."""
        import rustworkx as rx

        defines: defaultdict[str, set[str]] = defaultdict(set)
        references: defaultdict[str, list[str]] = defaultdict(list)
        definitions: defaultdict[tuple[str, str], set[Tag]] = defaultdict(set)
        personalization: dict[str, float] = {}
        exclude_rel_fnames: set[str] = set()

        sorted_fnames = sorted(files)
        personalize = 100 / len(sorted_fnames) if sorted_fnames else 0

        for fname in sorted_fnames:
            info = await self._info(fname)
            if info is None or info.type != "file":
                if fname not in self.warned_files:
                    self.warned_files.add(fname)
                continue

            rel_fname = get_rel_path(fname, self.root_path)
            current_pers = 0.0

            if fname in exclude:
                current_pers += personalize
                exclude_rel_fnames.add(rel_fname)

            if fname in boost_files or rel_fname in boost_files:
                current_pers = max(current_pers, personalize)

            path_obj = PurePosixPath(rel_fname)
            path_components = set(path_obj.parts)
            basename_with_ext = path_obj.name
            basename_without_ext = path_obj.stem
            components_to_check = path_components.union({basename_with_ext, basename_without_ext})

            if components_to_check.intersection(boost_idents):
                current_pers += personalize

            if current_pers > 0:
                personalization[rel_fname] = current_pers

            tags = await self._get_tags(fname, rel_fname)
            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)
                elif tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        if not references:
            references = defaultdict(list, {k: list(v) for k, v in defines.items()})

        idents = set(defines.keys()).intersection(set(references.keys()))

        graph: rx.PyDiGraph[str, dict[str, Any]] = rx.PyDiGraph(multigraph=True)
        node_to_idx: dict[str, int] = {}
        idx_to_node: dict[int, str] = {}

        def get_or_add_node(name: str) -> int:
            if name not in node_to_idx:
                idx = graph.add_node(name)
                node_to_idx[name] = idx
                idx_to_node[idx] = name
            return node_to_idx[name]

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
                    scaled_refs = math.sqrt(num_refs)
                    src_idx = get_or_add_node(referencer)
                    dst_idx = get_or_add_node(definer)
                    graph.add_edge(
                        src_idx, dst_idx, {"weight": use_mul * scaled_refs, "ident": ident}
                    )

        if not graph.num_nodes():
            return []

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

        ranked: dict[str, float] = {idx_to_node[idx]: rank for idx, rank in ranked_idx.items()}

        ranked_definitions: defaultdict[tuple[str, str], float] = defaultdict(float)
        for src_idx in graph.node_indices():
            src_name = idx_to_node[src_idx]
            src_rank = ranked[src_name]
            out_edges = graph.out_edges(src_idx)
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

        rel_fnames_without_tags = {get_rel_path(fname, self.root_path) for fname in files}
        for fname in exclude:
            rel = get_rel_path(fname, self.root_path)
            rel_fnames_without_tags.discard(rel)

        fnames_already_included = {
            tag.rel_fname if isinstance(tag, Tag) else tag[0] for tag in ranked_tags
        }

        top_rank = sorted([(rank, node) for (node, rank) in ranked.items()], reverse=True)
        for _rank, fname in top_rank:
            if fname in rel_fnames_without_tags:
                rel_fnames_without_tags.remove(fname)
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))

        for fname in rel_fnames_without_tags:
            ranked_tags.append((fname,))

        return ranked_tags

    async def _get_ranked_tags_map(
        self,
        files: Sequence[str],
        exclude: set[str],
        boost_files: set[str],
        boost_idents: set[str],
        max_tokens: int | None = None,
    ) -> str | None:
        """Generate ranked tags map within token budget."""
        max_tokens = max_tokens or self.max_tokens

        ranked_tags = await self._get_ranked_tags(
            files=files,
            exclude=exclude,
            boost_files=boost_files,
            boost_idents=boost_idents,
        )

        if not ranked_tags:
            return None

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree: str | None = None
        best_tree_tokens: float = 0
        exclude_rel_fnames = {get_rel_path(fname, self.root_path) for fname in exclude}
        self.tree_cache = {}

        middle = min(int(max_tokens // 25), num_tags)

        while lower_bound <= upper_bound:
            tree = await self._to_tree(ranked_tags[:middle], exclude_rel_fnames)
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

    async def _render_tree(
        self,
        abs_fname: str,
        rel_fname: str,
        lois: list[int],
        line_ranges: dict[int, int] | None = None,
    ) -> str:
        """Render a tree representation of a file with lines of interest."""
        import re

        from grep_ast import TreeContext  # type: ignore[import-untyped]

        if line_ranges is None:
            line_ranges = {}

        info = await self._info(abs_fname)
        mtime = info.mtime if info else None

        key = (rel_fname, tuple(sorted(lois)), mtime)
        if key in self.tree_cache:
            return self.tree_cache[key]

        cached = self.tree_context_cache.get(rel_fname)
        if cached is None or cached["mtime"] != mtime:
            code = await self._cat_file(abs_fname) or ""
            if not code.endswith("\n"):
                code += "\n"

            context = TreeContext(
                rel_fname,
                code,
                child_context=False,
                last_line=False,
                margin=0,
                mark_lois=False,
                loi_pad=0,
                show_top_of_file_parent_scope=False,
            )
            self.tree_context_cache[rel_fname] = {"context": context, "mtime": mtime}

        context = self.tree_context_cache[rel_fname]["context"]
        context.lines_of_interest = set()
        context.add_lines_of_interest(lois)
        context.add_context()
        res: str = context.format()

        code = await self._cat_file(abs_fname) or ""
        code_lines = code.splitlines()
        lois_set = set(lois)

        def_pattern = re.compile(r"^(.*?)(class\s+\w+|def\s+\w+|async\s+def\s+\w+)")

        result_lines = []
        for output_line in res.splitlines():
            modified_line = output_line
            match = def_pattern.search(output_line)
            if match:
                stripped = output_line.lstrip("│ \t")
                for line_num in lois_set:
                    if line_num < len(code_lines):
                        orig_line = code_lines[line_num].strip()
                        if orig_line and stripped.startswith(orig_line.split("(")[0].split(":")[0]):
                            name_match = re.search(
                                r"(class\s+\w+|def\s+\w+|async\s+def\s+\w+)", output_line
                            )
                            if name_match:
                                start_line_display = line_num + 1
                                end_line = line_ranges.get(line_num, -1)
                                if end_line >= 0 and end_line != line_num:
                                    end_line_display = end_line + 1
                                    line_info = f"  # [{start_line_display}-{end_line_display}]"
                                else:
                                    line_info = f"  # [{start_line_display}]"
                                modified_line = f"{output_line}{line_info}"
                            break
            result_lines.append(modified_line)

        res = "\n".join(result_lines)
        if result_lines:
            res += "\n"

        self.tree_cache[key] = res
        return res

    async def _to_tree(self, tags: list[RankedTag], exclude_rel_fnames: set[str]) -> str:
        """Convert ranked tags to a tree representation."""
        if not tags:
            return ""

        cur_fname: str | None = None
        cur_abs_fname: str | None = None
        lois: list[int] | None = None
        line_ranges: dict[int, int] | None = None
        output = ""

        dummy_tag: tuple[None] = (None,)
        for tag in [*sorted(tags, key=lambda t: str(t[0]) if t[0] else ""), dummy_tag]:
            this_rel_fname = tag[0] if isinstance(tag, Tag) else (tag[0] if tag[0] else None)

            if isinstance(tag, Tag):
                this_rel_fname = tag.rel_fname
            elif tag[0] is not None:
                this_rel_fname = tag[0]
            else:
                this_rel_fname = None

            if this_rel_fname and this_rel_fname in exclude_rel_fnames:
                continue

            if this_rel_fname != cur_fname:
                if lois is not None and cur_fname and cur_abs_fname:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += await self._render_tree(cur_abs_fname, cur_fname, lois, line_ranges)
                    lois = None
                    line_ranges = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"

                if isinstance(tag, Tag):
                    lois = []
                    line_ranges = {}
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None and line_ranges is not None and isinstance(tag, Tag):
                if tag.signature_end_line >= tag.line:
                    lois.extend(range(tag.line, tag.signature_end_line + 1))
                else:
                    lois.append(tag.line)
                if tag.end_line >= 0:
                    line_ranges[tag.line] = tag.end_line

        return "\n".join([line[: self.max_line_length] for line in output.splitlines()]) + "\n"
