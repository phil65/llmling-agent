from __future__ import annotations

import asyncio
from asyncio import transports as aio_transports
import contextlib
from contextlib import asynccontextmanager
import logging
import platform
import sys
from typing import TYPE_CHECKING, Any, cast

from acp.agent.connection import AgentSideConnection
from acp.client.connection import ClientSideConnection
from acp.connection import Connection
from acp.transports import spawn_stdio_transport


if TYPE_CHECKING:
    import asyncio.subprocess as aio_subprocess
    from collections.abc import AsyncIterator, Callable, Mapping
    from pathlib import Path

    from acp.agent.protocol import Agent
    from acp.client.protocol import Client
    from acp.connection import MethodHandler, StreamObserver

__all__ = [
    "connect_to_agent",
    "run_agent",
    "spawn_agent_process",
    "spawn_client_process",
    "spawn_stdio_connection",
    "stdio_streams",
]


class _WritePipeProtocol(asyncio.BaseProtocol):
    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._paused = False
        self._drain_waiter: asyncio.Future[None] | None = None

    def pause_writing(self) -> None:
        self._paused = True
        if self._drain_waiter is None:
            self._drain_waiter = self._loop.create_future()

    def resume_writing(self) -> None:
        self._paused = False
        if self._drain_waiter is not None and not self._drain_waiter.done():
            self._drain_waiter.set_result(None)
        self._drain_waiter = None

    async def _drain_helper(self) -> None:
        if self._paused and self._drain_waiter is not None:
            await self._drain_waiter


def _start_stdin_feeder(loop: asyncio.AbstractEventLoop, reader: asyncio.StreamReader) -> None:
    # Feed stdin from a background thread line-by-line
    def blocking_read() -> None:
        try:
            while True:
                data = sys.stdin.buffer.readline()
                if not data:
                    break
                loop.call_soon_threadsafe(reader.feed_data, data)
        finally:
            loop.call_soon_threadsafe(reader.feed_eof)

    import threading

    threading.Thread(target=blocking_read, daemon=True).start()


class _StdoutTransport(asyncio.BaseTransport):
    def __init__(self) -> None:
        self._is_closing = False

    def write(self, data: bytes) -> None:
        if self._is_closing:
            return
        try:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        except Exception:
            logging.exception("Error writing to stdout")

    def can_write_eof(self) -> bool:
        return False

    def is_closing(self) -> bool:
        return self._is_closing

    def close(self) -> None:
        self._is_closing = True
        with contextlib.suppress(Exception):
            sys.stdout.flush()

    def abort(self) -> None:
        self.close()

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return default


async def _windows_stdio_streams(
    loop: asyncio.AbstractEventLoop,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader = asyncio.StreamReader()
    _ = asyncio.StreamReaderProtocol(reader)

    _start_stdin_feeder(loop, reader)

    write_protocol = _WritePipeProtocol()
    transport = _StdoutTransport()
    writer = asyncio.StreamWriter(
        cast(aio_transports.WriteTransport, transport), write_protocol, None, loop
    )
    return reader, writer


async def _posix_stdio_streams(
    loop: asyncio.AbstractEventLoop,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    # Reader from stdin
    reader = asyncio.StreamReader()
    reader_protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: reader_protocol, sys.stdin)

    # Writer to stdout with protocol providing _drain_helper
    write_protocol = _WritePipeProtocol()
    transport, _ = await loop.connect_write_pipe(lambda: write_protocol, sys.stdout)
    writer = asyncio.StreamWriter(transport, write_protocol, None, loop)
    return reader, writer


async def stdio_streams() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Create stdio asyncio streams.

    On Windows use a thread feeder + custom stdout transport.
    """
    loop = asyncio.get_running_loop()
    if platform.system() == "Windows":
        return await _windows_stdio_streams(loop)
    return await _posix_stdio_streams(loop)


@asynccontextmanager
async def spawn_stdio_connection(
    handler: MethodHandler,
    command: str,
    *args: str,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    observers: list[StreamObserver] | None = None,
    **transport_kwargs: Any,
) -> AsyncIterator[tuple[Connection, aio_subprocess.Process]]:
    """Spawn a subprocess and bind its stdio to a low-level Connection."""
    async with spawn_stdio_transport(command, *args, env=env, cwd=cwd, **transport_kwargs) as (
        reader,
        writer,
        process,
    ):
        conn = Connection(handler, writer, reader, observers=observers)
        try:
            yield conn, process
        finally:
            await conn.close()


@asynccontextmanager
async def spawn_agent_process(
    to_client: Callable[[Agent], Client],
    command: str,
    *args: str,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    transport_kwargs: Mapping[str, Any] | None = None,
    **connection_kwargs: Any,
) -> AsyncIterator[tuple[ClientSideConnection, aio_subprocess.Process]]:
    """Spawn an ACP agent subprocess and return a ClientSideConnection to it."""
    async with spawn_stdio_transport(
        command,
        *args,
        env=env,
        cwd=cwd,
        **(dict(transport_kwargs) if transport_kwargs else {}),
    ) as (reader, writer, process):
        conn = ClientSideConnection(to_client, writer, reader, **connection_kwargs)
        try:
            yield conn, process
        finally:
            await conn.close()


@asynccontextmanager
async def spawn_client_process(
    to_agent: Callable[[AgentSideConnection], Agent],
    command: str,
    *args: str,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    transport_kwargs: Mapping[str, Any] | None = None,
    **connection_kwargs: Any,
) -> AsyncIterator[tuple[AgentSideConnection, aio_subprocess.Process]]:
    """Spawn an ACP client subprocess and return an AgentSideConnection to it."""
    async with spawn_stdio_transport(
        command,
        *args,
        env=env,
        cwd=cwd,
        **(dict(transport_kwargs) if transport_kwargs else {}),
    ) as (reader, writer, process):
        conn = AgentSideConnection(to_agent, writer, reader, **connection_kwargs)
        try:
            yield conn, process
        finally:
            await conn.close()


async def run_agent(
    agent: Agent | Callable[[AgentSideConnection], Agent],
    input_stream: asyncio.StreamWriter | None = None,
    output_stream: asyncio.StreamReader | None = None,
    **connection_kwargs: Any,
) -> None:
    """Run an ACP agent over stdio or provided streams.

    This is the recommended entry point for running agents. It handles stream
    setup and connection lifecycle automatically.

    Args:
        agent: An Agent implementation or a factory callable that takes
            an AgentSideConnection and returns an Agent. Using a factory allows
            the agent to access the connection for client communication.
        input_stream: Optional StreamWriter for output (defaults to stdio).
        output_stream: Optional StreamReader for input (defaults to stdio).
        **connection_kwargs: Additional keyword arguments for AgentSideConnection.

    Example with direct agent:
        ```python
        class MyAgent(Agent):
            async def initialize(self, params: InitializeRequest) -> InitializeResponse:
                return InitializeResponse(protocol_version=params.protocol_version)
            # ... implement protocol methods ...

        await run_agent(MyAgent())
        ```

    Example with factory:
        ```python
        class MyAgent(Agent):
            def __init__(self, connection: AgentSideConnection):
                self.connection = connection
            # ... implement protocol methods ...

        def create_agent(conn: AgentSideConnection) -> MyAgent:
            return MyAgent(conn)

        await run_agent(create_agent)
        ```
    """
    if input_stream is None or output_stream is None:
        output_stream, input_stream = await stdio_streams()

    # Wrap agent instance in factory if needed
    if callable(agent):
        agent_factory = agent  # pyright: ignore[reportAssignmentType]
    else:

        def agent_factory(connection: AgentSideConnection) -> Agent:
            return agent

    conn = AgentSideConnection(agent_factory, input_stream, output_stream, **connection_kwargs)
    shutdown_event = asyncio.Event()
    try:
        # Keep the connection alive until cancelled
        await shutdown_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        await conn.close()


def connect_to_agent(
    client: Client,
    input_stream: asyncio.StreamWriter,
    output_stream: asyncio.StreamReader,
    **connection_kwargs: Any,
) -> ClientSideConnection:
    """Create a ClientSideConnection to an ACP agent.

    This is the recommended entry point for client-side connections.

    Args:
        client: The client implementation.
        input_stream: StreamWriter for sending to the agent.
        output_stream: StreamReader for receiving from the agent.
        **connection_kwargs: Additional keyword arguments for ClientSideConnection.

    Returns:
        A ClientSideConnection connected to the agent.
    """

    # Create a factory that ignores the connection parameter
    def client_factory(connection: Agent) -> Client:
        return client

    return ClientSideConnection(client_factory, input_stream, output_stream, **connection_kwargs)
