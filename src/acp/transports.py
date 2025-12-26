from __future__ import annotations

import contextlib
from contextlib import asynccontextmanager
import os
import subprocess
from typing import TYPE_CHECKING

import anyio


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

    from anyio.abc import ByteReceiveStream, ByteSendStream, Process


DEFAULT_INHERITED_ENV_VARS = (
    [
        "APPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "USERNAME",
        "USERPROFILE",
    ]
    if os.name == "nt"
    else ["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"]
)


def default_environment() -> dict[str, str]:
    """Return a trimmed environment based on MCP best practices."""
    env: dict[str, str] = {}
    for key in DEFAULT_INHERITED_ENV_VARS:
        value = os.environ.get(key)
        if value is None:
            continue
        # Skip function-style env vars on some shells (see MCP reference)
        if value.startswith("()"):
            continue
        env[key] = value
    return env


@asynccontextmanager
async def spawn_stdio_transport(
    command: str,
    *args: str,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    stderr: int | None = subprocess.PIPE,
    shutdown_timeout: float = 2.0,
) -> AsyncIterator[tuple[ByteReceiveStream, ByteSendStream, Process]]:
    """Launch a subprocess and expose its stdio streams as anyio streams.

    This mirrors the defensive shutdown behaviour used by the MCP Python SDK:
    close stdin first, wait for graceful exit, then escalate to terminate/kill.
    """
    merged_env = default_environment()
    if env:
        merged_env.update(env)

    process = await anyio.open_process(
        [command, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr,
        env=merged_env,
        cwd=str(cwd) if cwd is not None else None,
    )

    if process.stdout is None or process.stdin is None:
        process.kill()
        await process.wait()
        msg = "spawn_stdio_transport requires stdout/stdin pipes"
        raise RuntimeError(msg)

    try:
        yield process.stdout, process.stdin, process
    finally:
        # Attempt graceful stdin shutdown first
        if process.stdin is not None:
            with contextlib.suppress(Exception):
                await process.stdin.aclose()

        try:
            with anyio.fail_after(shutdown_timeout):
                await process.wait()
        except TimeoutError:
            process.terminate()
            try:
                with anyio.fail_after(shutdown_timeout):
                    await process.wait()
            except TimeoutError:
                process.kill()
                await process.wait()
