"""Find/grep functionality tests.

Ported from OpenCode's test/tool/grep.test.ts

Tests the /find and /find/file endpoints for text and file searching.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestFindText:
    """Tests for /find endpoint (text/regex search)."""

    @pytest.mark.asyncio
    async def test_basic_search(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Basic pattern search should find matches.

        Ported from: "basic search"
        """
        # Create files with searchable content
        (tmp_project_dir / "file1.py").write_text("export function hello() {}")
        (tmp_project_dir / "file2.py").write_text("import os\nexport const value = 1")
        (tmp_project_dir / "file3.txt").write_text("no matches here")

        response = await async_client.get(
            "/find",
            params={"pattern": "export"},
        )

        assert response.status_code == 200
        matches = response.json()

        assert len(matches) >= 2
        # Verify match structure (API uses snake_case)
        for match in matches:
            assert "path" in match
            assert "lines" in match
            assert "line_number" in match

    @pytest.mark.asyncio
    async def test_no_matches_returns_empty_list(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Search with no matches should return empty list.

        Ported from: "no matches returns correct output"
        """
        (tmp_project_dir / "test.txt").write_text("hello world")

        response = await async_client.get(
            "/find",
            params={"pattern": "xyznonexistentpatternxyz123"},
        )

        assert response.status_code == 200
        matches = response.json()
        assert matches == []

    @pytest.mark.asyncio
    async def test_handles_crlf_line_endings(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Search should handle CRLF line endings.

        Ported from: "handles CRLF line endings in output"
        """
        # Create file with Unix line endings
        (tmp_project_dir / "unix.txt").write_text("line1\nline2\nline3")

        response = await async_client.get(
            "/find",
            params={"pattern": "line"},
        )

        assert response.status_code == 200
        matches = response.json()
        assert len(matches) == 3

    @pytest.mark.asyncio
    async def test_handles_windows_crlf(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Search should handle Windows CRLF line endings."""
        # Create file with Windows line endings
        (tmp_project_dir / "windows.txt").write_text("line1\r\nline2\r\nline3")

        response = await async_client.get(
            "/find",
            params={"pattern": "line"},
        )

        assert response.status_code == 200
        matches = response.json()
        assert len(matches) == 3

    @pytest.mark.asyncio
    async def test_handles_mixed_line_endings(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Search should handle mixed line endings."""
        # Create file with mixed line endings
        (tmp_project_dir / "mixed.txt").write_text("line1\nline2\r\nline3")

        response = await async_client.get(
            "/find",
            params={"pattern": "line"},
        )

        assert response.status_code == 200
        matches = response.json()
        assert len(matches) == 3

    @pytest.mark.asyncio
    async def test_regex_pattern(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Search should support regex patterns."""
        (tmp_project_dir / "code.py").write_text(
            "def hello():\n    pass\ndef world():\n    return 1"
        )

        # Search for function definitions
        response = await async_client.get(
            "/find",
            params={"pattern": r"def \w+\(\)"},
        )

        assert response.status_code == 200
        matches = response.json()
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_invalid_regex_returns_400(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Invalid regex pattern should return 400."""
        response = await async_client.get(
            "/find",
            params={"pattern": "[invalid(regex"},
        )

        assert response.status_code == 400
        assert "regex" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_search_in_subdirectories(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Search should find matches in subdirectories."""
        subdir = tmp_project_dir / "src" / "utils"
        subdir.mkdir(parents=True)
        (subdir / "helper.py").write_text("SEARCH_TARGET = 'found'")
        (tmp_project_dir / "main.py").write_text("SEARCH_TARGET = 'also found'")

        response = await async_client.get(
            "/find",
            params={"pattern": "SEARCH_TARGET"},
        )

        assert response.status_code == 200
        matches = response.json()
        assert len(matches) == 2

        paths = [m["path"]["text"] for m in matches]
        assert any("src/utils/helper.py" in p or "src\\utils\\helper.py" in p for p in paths)
        assert any("main.py" in p for p in paths)

    @pytest.mark.asyncio
    async def test_skips_excluded_directories(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Search should skip .git, node_modules, etc."""
        # Create files in excluded directories
        git_dir = tmp_project_dir / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("FINDME in git")

        node_modules = tmp_project_dir / "node_modules"
        node_modules.mkdir()
        (node_modules / "package.json").write_text("FINDME in node_modules")

        # Create a file that should be found
        (tmp_project_dir / "src.py").write_text("FINDME in source")

        response = await async_client.get(
            "/find",
            params={"pattern": "FINDME"},
        )

        assert response.status_code == 200
        matches = response.json()

        # Should only find the src.py file
        assert len(matches) == 1
        assert "src.py" in matches[0]["path"]["text"]

    @pytest.mark.asyncio
    async def test_returns_line_numbers(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Search results should include correct line numbers."""
        (tmp_project_dir / "lines.txt").write_text("one\ntwo\nthree\nfour\nfive")

        response = await async_client.get(
            "/find",
            params={"pattern": "three"},
        )

        assert response.status_code == 200
        matches = response.json()
        assert len(matches) == 1
        assert matches[0]["line_number"] == 3

    @pytest.mark.asyncio
    async def test_limits_results(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Search should limit results to prevent overwhelming responses."""
        # Create a file with many matches
        content = "\n".join([f"match{i}" for i in range(200)])
        (tmp_project_dir / "many.txt").write_text(content)

        response = await async_client.get(
            "/find",
            params={"pattern": "match"},
        )

        assert response.status_code == 200
        matches = response.json()
        # Should be capped at max_matches (100)
        assert len(matches) <= 100


class TestFindFiles:
    """Tests for /find/file endpoint (file name search)."""

    @pytest.mark.asyncio
    async def test_find_by_extension(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Should find files by extension pattern."""
        (tmp_project_dir / "file1.py").write_text("")
        (tmp_project_dir / "file2.py").write_text("")
        (tmp_project_dir / "file3.txt").write_text("")

        response = await async_client.get(
            "/find/file",
            params={"query": "*.py"},
        )

        assert response.status_code == 200
        files = response.json()
        assert len(files) == 2
        assert all(f.endswith(".py") for f in files)

    @pytest.mark.asyncio
    async def test_find_by_name_pattern(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Should find files by name pattern."""
        (tmp_project_dir / "test_one.py").write_text("")
        (tmp_project_dir / "test_two.py").write_text("")
        (tmp_project_dir / "main.py").write_text("")

        response = await async_client.get(
            "/find/file",
            params={"query": "test_*.py"},
        )

        assert response.status_code == 200
        files = response.json()
        assert len(files) == 2
        assert all("test_" in f for f in files)

    @pytest.mark.asyncio
    async def test_find_in_subdirectories(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Should find files in subdirectories."""
        subdir = tmp_project_dir / "src" / "components"
        subdir.mkdir(parents=True)
        (subdir / "Button.tsx").write_text("")
        (subdir / "Input.tsx").write_text("")
        (tmp_project_dir / "App.tsx").write_text("")

        response = await async_client.get(
            "/find/file",
            params={"query": "*.tsx"},
        )

        assert response.status_code == 200
        files = response.json()
        assert len(files) == 3

    @pytest.mark.asyncio
    async def test_skips_excluded_directories(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Should skip excluded directories like node_modules."""
        node_modules = tmp_project_dir / "node_modules" / "package"
        node_modules.mkdir(parents=True)
        (node_modules / "index.js").write_text("")

        (tmp_project_dir / "src.js").write_text("")

        response = await async_client.get(
            "/find/file",
            params={"query": "*.js"},
        )

        assert response.status_code == 200
        files = response.json()
        assert len(files) == 1
        assert files[0] == "src.js"

    @pytest.mark.asyncio
    async def test_include_directories_option(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Should optionally include directories in results."""
        (tmp_project_dir / "src").mkdir()
        (tmp_project_dir / "src_backup").mkdir()
        (tmp_project_dir / "src.py").write_text("")

        response = await async_client.get(
            "/find/file",
            params={"query": "src*", "dirs": "true"},
        )

        assert response.status_code == 200
        files = response.json()
        # Should include both directories and the file
        assert len(files) == 3

    @pytest.mark.asyncio
    async def test_returns_relative_paths(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """Results should be relative paths, not absolute."""
        subdir = tmp_project_dir / "deep" / "nested"
        subdir.mkdir(parents=True)
        (subdir / "file.py").write_text("")

        response = await async_client.get(
            "/find/file",
            params={"query": "*.py"},
        )

        assert response.status_code == 200
        files = response.json()
        assert len(files) == 1

        # Should be relative path
        path = files[0]
        assert not path.startswith("/")
        assert "deep" in path and "nested" in path

    @pytest.mark.asyncio
    async def test_no_matches_returns_empty(
        self,
        async_client,
        tmp_project_dir: Path,
    ):
        """No matches should return empty list."""
        (tmp_project_dir / "file.txt").write_text("")

        response = await async_client.get(
            "/find/file",
            params={"query": "*.nonexistent"},
        )

        assert response.status_code == 200
        assert response.json() == []
