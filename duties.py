"""Project tasks."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Literal

from duty import duty  # pyright: ignore[reportMissingImports]


# Read configuration from copier-answers.yml
_answers_file = Path(".copier-answers.yml")
if _answers_file.exists():
    content = _answers_file.read_text()
    match = re.search(r"^python_package_import_name:\s*(.+)$", content, re.MULTILINE)
    if match:
        PACKAGE_NAME = match.group(1).strip()
    else:
        msg = "python_package_import_name not found in copier-answers.yml"
        raise ValueError(msg)
else:
    msg = "copier-answers.yml not found in project root"
    raise FileNotFoundError(msg)


@duty(capture=False)
def build(ctx, *args: str):
    """Build documentation."""
    import subprocess

    args_str = " " + " ".join(args) if args else ""
    result = subprocess.run(
        f"uv run zensical build{args_str}",
        check=False,
        shell=True,
        capture_output=True,
        text=True,
    )
    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    # Check for errors in output (mkdocs may exit 0 even with template errors)
    error_patterns = ["Error:", "error:", "template not found"]
    combined_output = (result.stdout or "") + (result.stderr or "")
    for pattern in error_patterns:
        if pattern in combined_output:
            msg = f"Build failed: found '{pattern}' in output"
            raise RuntimeError(msg)
    if result.returncode != 0:
        msg = f"Build failed with exit code {result.returncode}"
        raise RuntimeError(msg)
    ctx.run("uv run python scripts/reorder_nav.py")


@duty(capture=False)
def serve(ctx, *args: str):
    """Serve documentation. Pass --reorder to build, reorder nav, and serve static files.

    With --reorder, also accepts --port=XXXX (default: 8000).
    """
    reorder = "--reorder" in args
    args = tuple(a for a in args if a != "--reorder")

    # Extract --port for reorder mode
    port = "8000"
    new_args = []
    for arg in args:
        if arg.startswith("--port="):
            port = arg.split("=", 1)[1]
        else:
            new_args.append(arg)
    args = tuple(new_args)

    if reorder:
        ctx.run("uv run zensical build")
        ctx.run("uv run python scripts/reorder_nav.py")
        # Serve the static site directly (zensical serve would rebuild and undo reorder)
        # Create a temp structure to match zensical's URL scheme: /agentpool/
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "agentpool").symlink_to(Path("site").resolve())
            print(f"Serving at http://localhost:{port}/agentpool/")
            ctx.run(f"uv run python -m http.server {port} -d {tmpdir} -b localhost")
    else:
        args_str = " " + " ".join(args) if args else ""
        ctx.run(f"uv run zensical serve{args_str}")


@duty(capture=False)
def test(ctx, *args: str):
    """Run tests."""
    args_str = " " + " ".join(args) if args else ""
    args_str = " -n auto" + args_str
    ctx.run(f"uv run pytest{args_str}")


@duty(capture=False)
def clean(ctx):
    """Clean all files from the Git directory except checked-in files."""
    ctx.run("git clean -dfX")


@duty(capture=False)
def update(ctx):
    """Update all environment packages using pip directly."""
    ctx.run("uv lock --upgrade")
    ctx.run("uv sync --all-extras")


def _get_lint_targets(filepath: str | None) -> tuple[str, str, str | None]:
    """Get lint targets based on optional filepath.

    Returns:
        Tuple of (ruff_target, mypy_target, jsonschema_files_or_none)
    """
    if filepath is None:
        return (
            ".",
            "src/",
            "src/agentpool/config_resources/*.yml docs/examples/**/config.yml",
        )

    path = Path(filepath)

    # For ruff, use the file directly
    ruff_target = filepath

    # For mypy, if it's under src/, use the file; otherwise skip
    mypy_target = filepath if filepath.startswith("src/") else ""

    # For jsonschema, only run if the file matches the patterns
    jsonschema_files = None
    if path.suffix in {".yml", ".yaml"} and (
        "config_resources" in filepath
        or ("docs/examples" in filepath and path.name == "config.yml")
    ):
        jsonschema_files = filepath

    return ruff_target, mypy_target, jsonschema_files


@duty(capture=False)
def lint(ctx, filepath: str | None = None):  # noqa: D417
    """Lint the code and fix issues if possible.

    Args:
        filepath: Optional path to a specific file to lint.
                  If not provided, lints the entire project.
    """
    ruff_target, mypy_target, jsonschema_files = _get_lint_targets(filepath)

    ctx.run(f"uv run ruff check --fix --unsafe-fixes {ruff_target}")
    ctx.run(f"uv run ruff format {ruff_target}")

    if mypy_target:
        ctx.run(f"uv run mypy {mypy_target} --fixed-format-cache")

    if jsonschema_files:
        ctx.run(
            f"uv run check-jsonschema --schemafile schema/config-schema.json {jsonschema_files}"
        )
    elif filepath is None:
        # Full project lint - run jsonschema on all config files
        ctx.run(
            "uv run check-jsonschema --schemafile schema/config-schema.json "
            "src/agentpool/config_resources/*.yml "
            "docs/examples/**/config.yml"
        )


@duty(capture=False)
def lint_check(ctx, filepath: str | None = None):  # noqa: D417
    """Lint the code (check only, no fixes).

    Args:
        filepath: Optional path to a specific file to lint.
                  If not provided, lints the entire project.
    """
    ruff_target, mypy_target, jsonschema_files = _get_lint_targets(filepath)

    ctx.run(f"uv run ruff check {ruff_target}")
    ctx.run(f"uv run ruff format --check {ruff_target}")

    if mypy_target:
        ctx.run(f"uv run mypy {mypy_target} --fixed-format-cache")

    if jsonschema_files:
        ctx.run(
            f"uv run check-jsonschema --schemafile schema/config-schema.json {jsonschema_files}"
        )
    elif filepath is None:
        # Full project lint - run jsonschema on all config files
        ctx.run(
            "uv run check-jsonschema --schemafile schema/config-schema.json "
            "src/agentpool/config_resources/*.yml "
            "docs/examples/**/config.yml"
        )


@duty(capture=False)
def version(
    ctx,
    *bump_type: Literal["major", "minor", "patch", "stable", "alpha", "beta", "rc"],
):
    """Release a new version with git operations. (major|minor|patch|stable|alpha|beta|rc)."""
    # Check for uncommitted changes
    result = ctx.run("git status --porcelain", capture=True)
    if result.strip():
        msg = "Cannot release with uncommitted changes. Please commit or stash first."
        raise RuntimeError(msg)

    # Read current version
    old_version = ctx.run("uv version --short", capture=True).strip()
    print(f"Current version: {old_version}")
    bump_str = " ".join(f"--bump {i}" for i in bump_type)
    ctx.run(f"uv version {bump_str}")
    new_version = ctx.run("uv version --short", capture=True).strip()
    print(f"New version: {new_version}")
    ctx.run("uv lock")

    # Update extension.toml
    ext_toml = Path("distribution/zed/extension.toml")
    if ext_toml.exists():
        content = ext_toml.read_text()
        content = content.replace(old_version, new_version)
        ext_toml.write_text(content)
        ctx.run("git add distribution/zed/extension.toml")

    ctx.run("git add pyproject.toml uv.lock")
    ctx.run(f'git commit -m "chore: bump version {old_version} -> {new_version}"')

    # Create and push tag
    tag = f"v{new_version}"
    ctx.run(f"git tag {tag}")
    print(f"Created tag: {tag}")


@duty(capture=False)
def ui(ctx, *args: str):
    """Launch Textual UI.

    Usage:
        duty ui                # Direct stdio (toad spawns agentpool)
        duty ui --websocket    # Via WebSocket (native transport)
    """
    use_websocket = "--websocket" in args
    args = tuple(a for a in args if a != "--websocket")
    args_str = " " + " ".join(args) if args else ""

    if use_websocket:
        import signal
        import subprocess
        import time

        port = 8765

        # Start ACP server with WebSocket transport directly (no bridge needed!)
        ws_server_cmd = [
            "uv",
            "run",
            "agentpool",
            "serve-acp",
            "--transport",
            "websocket",
            "--ws-port",
            str(port),
            "--model-provider",
            "openai",
        ]
        if args_str.strip():
            ws_server_cmd.extend(args_str.split())

        print(f"Starting ACP server with WebSocket transport on ws://localhost:{port}...")
        ws_server = subprocess.Popen(ws_server_cmd)

        try:
            # Wait for server to be ready
            time.sleep(1.5)

            # Run toad with mcp-ws client
            toad_cmd = (
                f'uvx --from batrachian-toad@latest toad acp "uvx mcp-ws ws://localhost:{port}"'
            )
            ctx.run(toad_cmd)
        finally:
            # Clean up WebSocket server
            print("Shutting down WebSocket server...")
            ws_server.send_signal(signal.SIGTERM)
            ws_server.wait(timeout=5)
    else:
        # Direct stdio - toad spawns agentpool directly
        cmd_parts = [
            "uvx --from batrachian-toad@latest toad acp",
            f'"uv run --python 3.13 agentpool serve-acp --model-provider openai{args_str}"',
        ]
        ctx.run(" ".join(cmd_parts))


@duty(capture=False)
def schema_html(ctx):
    """Create HTML documentation for JSON schema."""
    ctx.run(
        "generate-schema-doc "
        "--config template_name=js "
        "--config expand_buttons=true "
        "schema/config-schema.json "
        "docs/schema/index.html"
    )


@duty(capture=False)
def opencode_server(ctx, *args: str):
    """Start the OpenCode-compatible API server.

    Usage:
        duty opencode-server                    # Start on default port 4096
        duty opencode-server --port 8080        # Start on custom port
    """
    port = "4096"
    host = "127.0.0.1"

    # Parse arguments
    remaining_args = []
    args_iter = iter(args)
    for arg in args_iter:
        if arg == "--port":
            port = next(args_iter, "4096")
        elif arg.startswith("--port="):
            port = arg.split("=", 1)[1]
        elif arg == "--host":
            host = next(args_iter, "127.0.0.1")
        elif arg.startswith("--host="):
            host = arg.split("=", 1)[1]
        else:
            remaining_args.append(arg)

    print(f"Starting OpenCode server on http://{host}:{port}")
    print("Connect with: opencode attach http://{host}:{port}")
    ctx.run(
        f'uv run python -c "'
        f"from agentpool_server.opencode_server import run_server; "
        f"run_server(host='{host}', port={port})\""
    )


@duty(capture=False)
def opencode_tui(ctx, *args: str):
    """Attach OpenCode TUI to our server.

    Usage:
        duty opencode-tui                       # Connect to default port 4096
        duty opencode-tui --port 8080           # Connect to custom port
    """
    port = "4096"
    host = "127.0.0.1"

    # Parse arguments
    for arg in args:
        if arg.startswith("--port="):
            port = arg.split("=", 1)[1]
        elif arg.startswith("--host="):
            host = arg.split("=", 1)[1]

    url = f"http://{host}:{port}"
    print(f"Connecting OpenCode TUI to {url}")
    ctx.run(f"opencode attach {url}")


@duty(capture=False)
def opencode(ctx, *args: str):
    """Start OpenCode server and attach TUI (runs both in parallel).

    Usage:
        duty opencode                           # Start server + TUI on port 4096
        duty opencode --port 8080               # Use custom port
    """
    import signal
    import subprocess
    import time

    port = "4096"
    host = "127.0.0.1"

    # Parse arguments
    for arg in args:
        if arg.startswith("--port="):
            port = arg.split("=", 1)[1]
        elif arg.startswith("--host="):
            host = arg.split("=", 1)[1]

    url = f"http://{host}:{port}"

    # Start server in background
    server_cmd = [
        "uv",
        "run",
        "python",
        "-c",
        f"from agentpool_server.opencode_server import run_server; "
        f"run_server(host='{host}', port={port})",
    ]
    print(f"Starting OpenCode server on {url}...")
    server = subprocess.Popen(server_cmd)

    try:
        # Wait for server to be ready
        time.sleep(1.5)

        # Attach TUI
        print(f"Attaching OpenCode TUI to {url}...")
        ctx.run(f"opencode attach {url}")
    finally:
        # Clean up server
        print("Shutting down OpenCode server...")
        server.send_signal(signal.SIGTERM)
        server.wait(timeout=5)
