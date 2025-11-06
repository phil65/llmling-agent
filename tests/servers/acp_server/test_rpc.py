from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import anyenv
import pytest

from acp import (
    AgentSideConnection,
    CancelNotification,
    ClientSideConnection,
    InitializeRequest,
    InitializeResponse,
    NewSessionRequest,
    ReadTextFileRequest,
    SessionNotification,
    SetSessionModeRequest,
    WriteTextFileRequest,
)
from acp.schema import (
    AgentMessageChunk,
    AuthenticateRequest,
    AuthenticateResponse,
    LoadSessionRequest,
    LoadSessionResponse,
    SetSessionModelRequest,
    SetSessionModelResponse,
    SetSessionModeResponse,
    TextContentBlock,
    UserMessageChunk,
)


if TYPE_CHECKING:
    from acp import DefaultACPClient

    from .conftest import TestAgent


class _Server:
    def __init__(self) -> None:
        self._server: asyncio.AbstractServer | None = None
        self.server_reader: asyncio.StreamReader | None = None
        self.server_writer: asyncio.StreamWriter | None = None
        self.client_reader: asyncio.StreamReader | None = None
        self.client_writer: asyncio.StreamWriter | None = None

    async def __aenter__(self):
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            self.server_reader = reader
            self.server_writer = writer

        self._server = await asyncio.start_server(handle, host="127.0.0.1", port=0)
        host, port = self._server.sockets[0].getsockname()[:2]
        self.client_reader, self.client_writer = await asyncio.open_connection(host, port)

        # wait until server side is set
        for _ in range(100):
            if self.server_reader and self.server_writer:
                break
            await asyncio.sleep(0.01)
        assert self.server_reader
        assert self.server_writer
        return self

    async def __aexit__(self, *exc: object):
        if self.client_writer:
            self.client_writer.close()
            with contextlib.suppress(Exception):
                await self.client_writer.wait_closed()
        if self.server_writer:
            self.server_writer.close()
            with contextlib.suppress(Exception):
                await self.server_writer.wait_closed()
        if self._server:
            self._server.close()
            await self._server.wait_closed()


async def test_initialize_and_new_session(
    test_agent: TestAgent, test_client: DefaultACPClient
) -> None:
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        # server side is agent; client side is client
        agent_conn = ClientSideConnection(
            lambda _conn: test_client, s.client_writer, s.client_reader
        )
        _client_conn = AgentSideConnection(
            lambda _conn: test_agent, s.server_writer, s.server_reader
        )

        resp = await agent_conn.initialize(InitializeRequest(protocol_version=1))
        assert isinstance(resp, InitializeResponse)
        assert resp.protocol_version == 1
        request = NewSessionRequest(mcp_servers=[], cwd="/test")
        new_sess = await agent_conn.new_session(request)
        assert new_sess.session_id == "test-session-123"
        load_resp = await agent_conn.load_session(
            LoadSessionRequest(
                session_id=new_sess.session_id, cwd="/test", mcp_servers=[]
            )
        )
        assert isinstance(load_resp, LoadSessionResponse)

        auth_resp = await agent_conn.authenticate(
            AuthenticateRequest(method_id="password")
        )
        assert isinstance(auth_resp, AuthenticateResponse)

        mode_resp = await agent_conn.set_session_mode(
            SetSessionModeRequest(session_id=new_sess.session_id, mode_id="ask")
        )
        assert isinstance(mode_resp, SetSessionModeResponse)

        model_resp = await agent_conn.set_session_model(
            SetSessionModelRequest(session_id=new_sess.session_id, model_id="gpt-4o")
        )
        assert isinstance(model_resp, SetSessionModelResponse)


async def test_bidirectional_file_ops(
    test_agent: TestAgent, test_client: DefaultACPClient
) -> None:
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        agent = test_agent
        client = test_client
        client.files["/test/file.txt"] = "Hello, World!"
        _agent_conn = ClientSideConnection(
            lambda _conn: client, s.client_writer, s.client_reader
        )
        client_conn = AgentSideConnection(
            lambda _conn: agent, s.server_writer, s.server_reader
        )

        # Agent asks client to read
        read_request = ReadTextFileRequest(session_id="sess", path="/test/file.txt")
        res = await client_conn.read_text_file(read_request)
        assert res.content == "Hello, World!"

        # Agent asks client to write
        req = WriteTextFileRequest(session_id="sess", path="/test/file.txt", content="A")
        await client_conn.write_text_file(req)
        assert client.files["/test/file.txt"] == "A"


async def test_cancel_notification_and_capture_wire(
    test_agent: TestAgent, test_client: DefaultACPClient
) -> None:
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        # Build only agent-side (server) connection. Client side: reader to inspect wire
        agent = test_agent
        client = test_client
        agent_conn = ClientSideConnection(
            lambda _conn: client, s.client_writer, s.client_reader
        )
        _client_conn = AgentSideConnection(
            lambda _conn: agent, s.server_writer, s.server_reader
        )

        # Send cancel notification from client-side connection to agent
        await agent_conn.cancel(CancelNotification(session_id="test-123"))

        # Read raw line from server peer (it will be consumed by agent recv loop quickly).
        # Instead, wait a brief moment and assert agent recorded it.
        for _ in range(50):
            if agent.cancellations:
                break
            await asyncio.sleep(0.01)
        assert agent.cancellations == ["test-123"]


async def test_session_notifications_flow(
    test_agent: TestAgent, test_client: DefaultACPClient
) -> None:
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        _agent_conn = ClientSideConnection(
            lambda _conn: test_client, s.client_writer, s.client_reader
        )
        client_conn = AgentSideConnection(
            lambda _conn: test_agent, s.server_writer, s.server_reader
        )

        # Agent -> Client notifications
        content = TextContentBlock(text="Hello")
        agent_chunk = AgentMessageChunk(content=content)
        notification = SessionNotification(session_id="sess", update=agent_chunk)
        await client_conn.session_update(notification)
        content = TextContentBlock(text="World")
        chunk = UserMessageChunk(content=content)
        notification = SessionNotification(session_id="sess", update=chunk)
        await client_conn.session_update(notification)

        # Wait for async dispatch
        for _ in range(50):
            if len(test_client.notifications) >= 2:  # noqa: PLR2004
                break
            await asyncio.sleep(0.01)
        assert len(test_client.notifications) >= 2  # noqa: PLR2004
        assert test_client.notifications[0].session_id == "sess"


async def test_concurrent_reads(
    test_agent: TestAgent, test_client: DefaultACPClient
) -> None:
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        for i in range(5):
            test_client.files[f"/test/file{i}.txt"] = f"Content {i}"
        _agent_conn = ClientSideConnection(
            lambda _conn: test_client, s.client_writer, s.client_reader
        )
        client_conn = AgentSideConnection(
            lambda _conn: test_agent, s.server_writer, s.server_reader
        )

        async def read_one(i: int):
            request = ReadTextFileRequest(session_id="sess", path=f"/test/file{i}.txt")
            return await client_conn.read_text_file(request)

        results = await asyncio.gather(*(read_one(i) for i in range(5)))
        for i, res in enumerate(results):
            assert res.content == f"Content {i}"


async def test_invalid_params_results_in_error_response(test_agent: TestAgent):
    async with _Server() as s:
        # Only start agent-side (server) so we can inject raw request from client socket
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        _server_conn = AgentSideConnection(
            lambda _conn: test_agent, s.server_writer, s.server_reader
        )

        # Send initialize with wrong param type (protocolVersion should be int)
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "oops"},
        }
        s.client_writer.write((anyenv.dump_json(req) + "\n").encode())
        await s.client_writer.drain()

        # Read response
        line = await asyncio.wait_for(s.client_reader.readline(), timeout=1)
        resp = anyenv.load_json(line)
        assert resp["id"] == 1
        assert "error" in resp
        invalid_params_code = -32602
        assert resp["error"]["code"] == invalid_params_code


async def test_method_not_found_results_in_error_response(test_agent: TestAgent):
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        _server_conn = AgentSideConnection(
            lambda _conn: test_agent, s.server_writer, s.server_reader
        )

        req = {"jsonrpc": "2.0", "id": 2, "method": "unknown/method", "params": {}}
        s.client_writer.write((anyenv.dump_json(req) + "\n").encode())
        await s.client_writer.drain()

        line = await asyncio.wait_for(s.client_reader.readline(), timeout=1)
        resp = anyenv.load_json(line)
        assert resp["id"] == 2  # noqa: PLR2004
        method_not_found_code = -32601
        assert resp["error"]["code"] == method_not_found_code


async def test_set_session_mode_and_extensions(
    test_agent: TestAgent, test_client: DefaultACPClient
) -> None:
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        agent = test_agent
        client = test_client
        agent_conn = ClientSideConnection(
            lambda _conn: client, s.client_writer, s.client_reader
        )
        _client_conn = AgentSideConnection(
            lambda _conn: agent, s.server_writer, s.server_reader
        )

        # set_session_mode
        request = SetSessionModeRequest(session_id="sess", mode_id="yolo")
        resp = await agent_conn.set_session_mode(request)
        # Either empty object or typed response depending on implementation
        assert resp is None or resp.__class__.__name__ == "SetSessionModeResponse"

        # ext_method
        res = await agent_conn.ext_method("ping", {"x": 1})
        assert res.get("ok") is True

        # ext_notification
        await agent_conn.ext_notification("note", {"y": 2})
        # allow dispatch
        await asyncio.sleep(0.05)
        assert agent.ext_notes
        assert agent.ext_notes[-1][0] == "note"


async def test_ignore_invalid_messages(test_agent: TestAgent):
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        _server_conn = AgentSideConnection(
            lambda _conn: test_agent, s.server_writer, s.server_reader
        )

        # Message without id and method
        msg1 = {"jsonrpc": "2.0"}
        s.client_writer.write((anyenv.dump_json(msg1) + "\n").encode())
        await s.client_writer.drain()

        # Message without jsonrpc and without id/method
        msg2 = {"foo": "bar"}
        s.client_writer.write((anyenv.dump_json(msg2) + "\n").encode())
        await s.client_writer.drain()

        # Should not receive any response lines
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(s.client_reader.readline(), timeout=0.1)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
