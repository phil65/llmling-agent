"""LSP Proxy - Wraps stdio LSP server, exposes via Unix socket.

This script is executed as a subprocess to proxy between Unix socket clients
and a stdio-based LSP server.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
from pathlib import Path
import sys
from typing import Any


class LSPProxy:
    """Proxies between Unix socket clients and a stdio-based LSP server."""

    def __init__(self, command: list[str], socket_path: str):
        self.command = command
        self.socket_path = socket_path
        self._socket_path = Path(socket_path)
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self._request_id = 0

    async def start(self) -> None:
        """Start the LSP server subprocess."""
        # Remove existing socket if present
        if self._socket_path.exists():
            self._socket_path.unlink()

        # Use asyncio subprocess for native async I/O
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _read_lsp_message(self) -> dict[str, Any] | None:
        """Read a single JSON-RPC message from LSP server stdout."""
        if not self.process or not self.process.stdout:
            return None

        headers = {}
        while True:
            line = await self.process.stdout.readline()
            if not line:
                return None  # EOF
            line_str = line.decode().strip()
            if not line_str:
                break  # Empty line = end of headers
            if ": " in line_str:
                key, value = line_str.split(": ", 1)
                headers[key] = value

        if "Content-Length" not in headers:
            return None

        length = int(headers["Content-Length"])
        body = await self.process.stdout.readexactly(length)
        result: dict[str, Any] = json.loads(body)
        return result

    async def _send_to_lsp(self, message: dict[str, Any]) -> None:
        """Send JSON-RPC message to LSP server."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("LSP server not running")

        payload = json.dumps(message)
        header = f"Content-Length: {len(payload)}\r\n\r\n"
        self.process.stdin.write((header + payload).encode())
        await self.process.stdin.drain()

    async def send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send request to LSP and wait for response."""
        if not self.process:
            return {"error": "LSP server not running"}

        async with self.lock:
            self._request_id += 1
            msg_id = self._request_id
            request = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
            await self._send_to_lsp(request)

            try:
                response = await asyncio.wait_for(self._read_lsp_message(), timeout=30.0)
            except TimeoutError:
                # Check stderr for error info
                if self.process.stderr:
                    try:
                        err = await asyncio.wait_for(self.process.stderr.read(4096), timeout=0.1)
                        if err:
                            return {"error": f"Timeout, stderr: {err.decode()}"}
                    except TimeoutError:
                        pass
                return {"error": "Timeout waiting for LSP response"}

            if response is None:
                # Check stderr for error info
                if self.process.stderr:
                    try:
                        err = await asyncio.wait_for(self.process.stderr.read(4096), timeout=0.1)
                        if err:
                            return {"error": f"LSP error: {err.decode()}"}
                    except TimeoutError:
                        pass
                return {"error": "No response from LSP server"}

            return response

    async def send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send notification to LSP (no response expected)."""
        notification = {"jsonrpc": "2.0", "method": method, "params": params}
        async with self.lock:
            await self._send_to_lsp(notification)

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming client connection on Unix socket."""
        try:
            while True:
                # Read Content-Length header
                headers = b""
                while b"\r\n\r\n" not in headers:
                    chunk = await reader.read(1)
                    if not chunk:
                        return  # Client disconnected
                    headers += chunk

                # Parse Content-Length
                header_str = headers.decode()
                length = None
                for line in header_str.split("\r\n"):
                    if line.startswith("Content-Length:"):
                        length = int(line.split(":")[1].strip())
                        break

                if length is None:
                    return

                # Read body
                body = await reader.read(length)
                request = json.loads(body)

                # Forward to LSP
                if "id" in request:
                    # It's a request, wait for response
                    response = await self.send_request(request["method"], request.get("params", {}))
                else:
                    # It's a notification
                    await self.send_notification(request["method"], request.get("params", {}))
                    continue  # No response to send

                # Send response back to client
                payload = json.dumps(response)
                header = f"Content-Length: {len(payload)}\r\n\r\n"
                writer.write(header.encode() + payload.encode())
                await writer.drain()

        except (ConnectionError, json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Client error: {e}", file=sys.stderr, flush=True)
        finally:
            writer.close()
            await writer.wait_closed()

    async def run(self) -> None:
        """Start proxy server."""
        await self.start()

        # Ensure parent directory exists
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)

        server = await asyncio.start_unix_server(self.handle_client, path=self.socket_path)

        # Signal ready by creating a marker file
        Path(self.socket_path + ".ready").touch()

        print(f"LSP Proxy listening on {self.socket_path}", file=sys.stderr, flush=True)

        async with server:
            await server.serve_forever()

    async def shutdown(self) -> None:
        """Shutdown the proxy and LSP server."""
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                self.process.kill()

        # Cleanup socket
        ready_path = Path(self.socket_path + ".ready")
        if self._socket_path.exists():
            self._socket_path.unlink()
        if ready_path.exists():
            ready_path.unlink()


def main() -> None:
    """Entry point for the LSP proxy script."""
    parser = argparse.ArgumentParser(description="LSP Proxy Server")
    parser.add_argument("--command", required=True, help="LSP server command")
    parser.add_argument("--socket", required=True, help="Unix socket path")
    args = parser.parse_args()

    proxy = LSPProxy(args.command.split(), args.socket)

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(proxy.run())


if __name__ == "__main__":
    main()
