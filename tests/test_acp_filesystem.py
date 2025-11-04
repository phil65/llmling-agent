"""Tests for ACP filesystem."""

import sys

import pytest

from acp.filesystem import ACPFileSystem


if sys.platform == "win32":
    pytest.skip(allow_module_level=True)


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
        # Store full command with args for proper parsing
        full_command = request.command
        if request.args:
            full_command += " " + " ".join(request.args)
        self._terminal_commands[terminal_id] = full_command
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
        return type("Response", (), {"output": output, "truncated": False})()

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


async def test_cat_file(acp_fs):
    """Test reading file content."""
    content = await acp_fs._cat_file("test.txt")
    assert content == b"test content"


async def test_cat_file_not_found(acp_fs):
    """Test reading non-existent file."""
    with pytest.raises(FileNotFoundError):
        await acp_fs._cat_file("nonexistent.txt")


async def test_put_file(acp_fs):
    """Test writing file content."""
    await acp_fs._put_file("new.txt", "new content")
    # Verify it was written to mock
    assert acp_fs.client.files["new.txt"] == "new content"


async def test_put_file_bytes(acp_fs):
    """Test writing bytes content."""
    await acp_fs._put_file("new.txt", b"new content")
    assert acp_fs.client.files["new.txt"] == "new content"


async def test_ls_detail(acp_fs):
    """Test directory listing with details."""
    files = await acp_fs._ls(".", detail=True)
    assert isinstance(files, list)
    assert len(files) >= 1  # At least test.txt

    # Check that we have file entries with expected structure
    for file_info in files:
        assert "name" in file_info
        assert "type" in file_info
        assert "size" in file_info


async def test_ls_simple(acp_fs):
    """Test simple directory listing."""
    files = await acp_fs._ls(".", detail=False)
    assert isinstance(files, list)
    # Note: mock returns the same detailed output, so this tests the parsing


async def test_exists(acp_fs):
    """Test file existence check."""
    assert await acp_fs._exists("test.txt") is True
    assert await acp_fs._exists("nonexistent.txt") is False


async def test_isfile(acp_fs):
    """Test file type check."""
    assert await acp_fs._isfile("test.txt") is True
    assert await acp_fs._isfile("subdir") is False


async def test_isdir(acp_fs):
    """Test directory type check."""
    assert await acp_fs._isdir("subdir") is True
    assert await acp_fs._isdir("test.txt") is False


async def test_info(acp_fs):
    """Test file info retrieval."""
    info = await acp_fs._info("test.txt")
    assert info["name"] == "test.txt"
    assert info["type"] == "file"
    assert info["size"] == 12  # noqa: PLR2004


async def test_byte_range_not_supported(acp_fs):
    """Test that byte range reads are not supported."""
    with pytest.raises(NotImplementedError, match="byte range reads"):
        await acp_fs._cat_file("test.txt", start=0, end=10)


def test_open(acp_fs):
    """Test file opening."""
    file_obj = acp_fs.open("test.txt", "r")
    assert file_obj.path == "test.txt"
    assert file_obj.mode == "rb"  # Mode gets converted to binary


if __name__ == "__main__":
    pytest.main(["-v", __file__])
