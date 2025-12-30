"""LSP Proxy - Wraps stdio-based LSP servers and exposes them via Unix socket.

This module provides the proxy code that runs inside the execution environment.
It can be deployed as a standalone script or embedded via execute_code().
"""

from __future__ import annotations


# The proxy script that runs inside the execution environment
LSP_PROXY_SCRIPT = '''
"""LSP Proxy - Wraps stdio LSP server, exposes via Unix socket."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path


class LSPProxy:
    """Proxies between Unix socket clients and a stdio-based LSP server."""

    def __init__(self, command: list[str], socket_path: str):
        self.command = command
        self.socket_path = socket_path
        self.process: subprocess.Popen | None = None
        self.lock = asyncio.Lock()
        self._reader_task: asyncio.Task | None = None
        self._pending: dict[int | str, asyncio.Future] = {}
        self._notifications: asyncio.Queue = asyncio.Queue()

    async def start(self) -> None:
        """Start the LSP server subprocess."""
        # Remove existing socket if present
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Start background reader for LSP responses
        self._reader_task = asyncio.create_task(self._read_lsp_responses())

    def _send_to_lsp(self, message: dict) -> None:
        """Send JSON-RPC message to LSP server."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("LSP server not running")

        payload = json.dumps(message)
        header = f"Content-Length: {len(payload)}\\r\\n\\r\\n"
        self.process.stdin.write(header.encode() + payload.encode())
        self.process.stdin.flush()

    def _read_message(self) -> dict | None:
        """Read a single JSON-RPC message from LSP server stdout."""
        if not self.process or not self.process.stdout:
            return None

        # Read headers
        headers = {}
        while True:
            line = self.process.stdout.readline()
            if not line:
                return None  # EOF
            line = line.decode().strip()
            if not line:
                break  # Empty line = end of headers
            if ": " in line:
                key, value = line.split(": ", 1)
                headers[key] = value

        if "Content-Length" not in headers:
            return None

        # Read body
        length = int(headers["Content-Length"])
        body = self.process.stdout.read(length)
        return json.loads(body)

    async def _read_lsp_responses(self) -> None:
        """Background task: read responses from LSP and dispatch."""
        loop = asyncio.get_event_loop()

        while True:
            try:
                # Read in executor to avoid blocking
                message = await loop.run_in_executor(None, self._read_message)
                if message is None:
                    break

                # Dispatch based on message type
                if "id" in message and "method" not in message:
                    # Response to a request
                    msg_id = message["id"]
                    if msg_id in self._pending:
                        self._pending[msg_id].set_result(message)
                elif "method" in message and "id" not in message:
                    # Server notification (e.g., publishDiagnostics)
                    await self._notifications.put(message)
                elif "method" in message and "id" in message:
                    # Server request (rare, but possible)
                    # For now, just log it
                    print(f"Server request: {message.get('method')}", file=sys.stderr)

            except Exception as e:
                print(f"Error reading LSP response: {e}", file=sys.stderr)
                break

    async def send_request(self, method: str, params: dict) -> dict:
        """Send request to LSP and wait for response."""
        async with self.lock:
            # Generate ID
            msg_id = id(params)  # Simple unique ID

            # Create future for response
            future = asyncio.get_event_loop().create_future()
            self._pending[msg_id] = future

            # Send request
            request = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
            self._send_to_lsp(request)

        # Wait for response (outside lock)
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
        finally:
            self._pending.pop(msg_id, None)

        return response

    async def send_notification(self, method: str, params: dict) -> None:
        """Send notification to LSP (no response expected)."""
        notification = {"jsonrpc": "2.0", "method": method, "params": params}
        async with self.lock:
            self._send_to_lsp(notification)

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming client connection on Unix socket."""
        try:
            while True:
                # Read Content-Length header
                headers = b""
                while b"\\r\\n\\r\\n" not in headers:
                    chunk = await reader.read(1)
                    if not chunk:
                        return  # Client disconnected
                    headers += chunk

                # Parse Content-Length
                header_str = headers.decode()
                length = None
                for line in header_str.split("\\r\\n"):
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
                header = f"Content-Length: {len(payload)}\\r\\n\\r\\n"
                writer.write(header.encode() + payload.encode())
                await writer.drain()

        except Exception as e:
            print(f"Client error: {e}", file=sys.stderr)
        finally:
            writer.close()
            await writer.wait_closed()

    async def run(self) -> None:
        """Start proxy server."""
        await self.start()

        # Ensure parent directory exists
        Path(self.socket_path).parent.mkdir(parents=True, exist_ok=True)

        server = await asyncio.start_unix_server(self.handle_client, path=self.socket_path)

        # Signal ready by creating a marker file
        Path(self.socket_path + ".ready").touch()

        print(f"LSP Proxy listening on {self.socket_path}", file=sys.stderr)

        async with server:
            await server.serve_forever()

    async def shutdown(self) -> None:
        """Shutdown the proxy and LSP server."""
        if self._reader_task:
            self._reader_task.cancel()

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        # Cleanup socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        if os.path.exists(self.socket_path + ".ready"):
            os.unlink(self.socket_path + ".ready")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LSP Proxy Server")
    parser.add_argument("--command", required=True, help="LSP server command")
    parser.add_argument("--socket", required=True, help="Unix socket path")
    args = parser.parse_args()

    proxy = LSPProxy(args.command.split(), args.socket)

    try:
        asyncio.run(proxy.run())
    except KeyboardInterrupt:
        pass
'''


class LSPProxy:
    """Helper class for deploying and managing LSP proxy instances."""

    @staticmethod
    def get_script() -> str:
        """Get the proxy script code for embedding via execute_code()."""
        return LSP_PROXY_SCRIPT

    @staticmethod
    def get_start_command(lsp_command: str, socket_path: str) -> list[str]:
        """Get command to start proxy as a background process.

        Args:
            lsp_command: The LSP server command (e.g., "pyright-langserver --stdio")
            socket_path: Path for the Unix socket

        Returns:
            Command and args for process_manager.start_process()
        """
        return [
            "python",
            "-c",
            LSP_PROXY_SCRIPT,
            "--command",
            lsp_command,
            "--socket",
            socket_path,
        ]
