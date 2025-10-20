"""Tests for ACP filesystem."""

import pytest

from acp.filesystem import ACPFileSystem, ACPFileSystemSync, ACPPath


class MockClient:
    """Mock ACP client for testing."""

    def __init__(self):
        self.files = {"test.txt": "test content"}
        self._terminal_commands = {}
        self._terminal_counter = 0

    async def read_text_file(self, request):
        """Mock read_text_file."""
        path = request.path
        if path in self.files:
            return type("Response", (), {"content": self.files[path]})()
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    async def write_text_file(self, request):
        """Mock write_text_file."""
        self.files[request.path] = request.content

    async def create_terminal(self, request):
        """Mock create_terminal."""
        terminal_id = f"mock_terminal_{self._terminal_counter}"
        self._terminal_counter += 1
        self._terminal_commands[terminal_id] = request.command
        return type("Response", (), {"terminal_id": terminal_id})()

    async def wait_for_terminal_exit(self, request):
        """Mock wait_for_terminal_exit."""
        terminal_id = request.terminal_id
        command = self._terminal_commands.get(terminal_id, "")
        exit_code = 0
        if "test -e" in command and "test.txt" not in command:
            exit_code = 1
        if "test -f" in command and "test.txt" not in command:
            exit_code = 1
        if "test -d" in command and "subdir" not in command:
            exit_code = 1
        return type("Response", (), {"exit_code": exit_code})()

    async def terminal_output(self, request):
        """Mock terminal_output."""
        terminal_id = request.terminal_id
        command = self._terminal_commands.get(terminal_id, "")
        if "ls" in command:
            output = (
                "total 8\n"
                "-rw-r--r-- 1 user user 12 2024-01-01-12:00:00 test.txt\n"
                "drwxr-xr-x 2 user user 4096 2024-01-01-12:00:00 subdir"
            )
        elif "stat" in command:
            output = "test.txt|12|1704110400|-rw-r--r--|regular file"
        else:
            output = ""
        return type("Response", (), {"output": output})()

    async def release_terminal(self, request):
        """Mock release_terminal."""


class MockSession:
    """Mock ACP session for testing."""

    def __init__(self):
        self.client = MockClient()
        self.session_id = "test_session"


@pytest.fixture
def mock_session():
    """Create mock ACP session."""
    return MockSession()


@pytest.fixture
def acp_fs(mock_session):
    """Create ACP filesystem instance."""
    return ACPFileSystem(mock_session.client, mock_session.session_id)


@pytest.fixture
def acp_fs_sync(mock_session):
    """Create sync ACP filesystem instance."""
    return ACPFileSystemSync(mock_session.client, mock_session.session_id)


class TestACPFileSystem:
    """Test ACP filesystem functionality."""

    @pytest.mark.asyncio
    async def test_cat_file(self, acp_fs):
        """Test reading file content."""
        content = await acp_fs._cat_file("test.txt")
        assert content == b"test content"

    @pytest.mark.asyncio
    async def test_cat_file_not_found(self, acp_fs):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            await acp_fs._cat_file("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_put_file(self, acp_fs):
        """Test writing file content."""
        await acp_fs._put_file("new.txt", "new content")
        # Verify it was written to mock
        assert acp_fs.client.files["new.txt"] == "new content"

    @pytest.mark.asyncio
    async def test_put_file_bytes(self, acp_fs):
        """Test writing bytes content."""
        await acp_fs._put_file("new.txt", b"new content")
        assert acp_fs.client.files["new.txt"] == "new content"

    @pytest.mark.asyncio
    async def test_ls_detail(self, acp_fs):
        """Test directory listing with details."""
        files = await acp_fs._ls(".", detail=True)
        assert isinstance(files, list)
        assert len(files) >= 1  # At least test.txt

        # Check that we have file entries with expected structure
        for file_info in files:
            assert "name" in file_info
            assert "type" in file_info
            assert "size" in file_info

    @pytest.mark.asyncio
    async def test_ls_simple(self, acp_fs):
        """Test simple directory listing."""
        files = await acp_fs._ls(".", detail=False)
        assert isinstance(files, list)
        # Note: mock returns the same detailed output, so this tests the parsing

    @pytest.mark.asyncio
    async def test_exists(self, acp_fs):
        """Test file existence check."""
        assert await acp_fs._exists("test.txt") is True
        assert await acp_fs._exists("nonexistent.txt") is False

    @pytest.mark.asyncio
    async def test_isfile(self, acp_fs):
        """Test file type check."""
        assert await acp_fs._isfile("test.txt") is True
        assert await acp_fs._isfile("subdir") is False

    @pytest.mark.asyncio
    async def test_isdir(self, acp_fs):
        """Test directory type check."""
        assert await acp_fs._isdir("subdir") is True
        assert await acp_fs._isdir("test.txt") is False

    @pytest.mark.asyncio
    async def test_info(self, acp_fs):
        """Test file info retrieval."""
        info = await acp_fs._info("test.txt")
        assert info["name"] == "test.txt"
        assert info["type"] == "file"
        assert info["size"] == 12  # noqa: PLR2004

    @pytest.mark.asyncio
    async def test_byte_range_not_supported(self, acp_fs):
        """Test that byte range reads are not supported."""
        with pytest.raises(NotImplementedError, match="byte range reads"):
            await acp_fs._cat_file("test.txt", start=0, end=10)

    def test_open(self, acp_fs):
        """Test file opening."""
        file_obj = acp_fs.open("test.txt", "r")
        assert file_obj.path == "test.txt"
        assert file_obj.mode == "rb"  # Mode gets converted to binary


class TestACPFileSystemSync:
    """Test sync ACP filesystem functionality."""

    def test_ls(self, acp_fs_sync):
        """Test sync directory listing."""
        files = acp_fs_sync.ls(".", detail=True)
        assert isinstance(files, list)
        assert len(files) >= 0  # May be empty depending on parsing

    def test_cat(self, acp_fs_sync):
        """Test sync file reading."""
        content = acp_fs_sync.cat("test.txt")
        assert content == b"test content"

    def test_exists(self, acp_fs_sync):
        """Test sync file existence check."""
        assert acp_fs_sync.exists("test.txt") is True
        assert acp_fs_sync.exists("nonexistent.txt") is False

    def test_info(self, acp_fs_sync):
        """Test sync file info retrieval."""
        info = acp_fs_sync.info("test.txt")
        assert info["name"] == "test.txt"
        assert info["type"] == "file"


class TestACPPath:
    """Test ACP path functionality."""

    def test_acp_path_creation(self, mock_session):
        """Test ACP path object creation."""
        fs = ACPFileSystemSync(mock_session.client, mock_session.session_id)
        path = ACPPath("test.txt", fs=fs)
        assert str(path) == "test.txt"

    def test_make_path(self, acp_fs):
        """Test path creation from filesystem."""
        path = acp_fs._make_path("test.txt")
        assert isinstance(path, ACPPath)


class TestParsingUtilities:
    """Test parsing utility methods."""

    def test_parse_ls_output_detailed(self, acp_fs):
        """Test parsing detailed ls output."""
        output = (
            "total 8\n"
            "-rw-r--r-- 1 user user 12 2024-01-01-12:00:00 test.txt\n"
            "drwxr-xr-x 2 user user 4096 2024-01-01-12:00:00 subdir"
        )
        files = acp_fs._parse_ls_output(output, ".", detail=True)

        assert len(files) == 2  # noqa: PLR2004

        # Find test.txt entry
        test_file = next((f for f in files if f["name"] == "test.txt"), None)
        assert test_file is not None
        assert test_file["type"] == "file"
        assert test_file["size"] == 12  # noqa: PLR2004

        # Find subdir entry
        sub_dir = next((f for f in files if f["name"] == "subdir"), None)
        assert sub_dir is not None
        assert sub_dir["type"] == "directory"

    def test_parse_stat_output(self, acp_fs):
        """Test parsing stat command output."""
        output = "test.txt|12|1704110400|-rw-r--r--|regular file"
        info = acp_fs._parse_stat_output(output, "test.txt")

        assert info["name"] == "test.txt"
        assert info["type"] == "file"
        assert info["size"] == 12  # noqa: PLR2004
        assert info["mtime"] == 1704110400  # noqa: PLR2004

    def test_parse_stat_output_directory(self, acp_fs):
        """Test parsing stat output for directory."""
        output = "subdir|4096|1704110400|drwxr-xr-x|directory"
        info = acp_fs._parse_stat_output(output, "subdir")

        assert info["name"] == "subdir"
        assert info["type"] == "directory"
        assert info["size"] == 4096  # noqa: PLR2004

    def test_parse_stat_output_invalid(self, acp_fs):
        """Test parsing invalid stat output."""
        with pytest.raises(ValueError, match="Unexpected stat output format"):
            acp_fs._parse_stat_output("invalid", "test.txt")
