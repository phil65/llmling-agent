"""Example ACP client for testing LLMling-Agent ACP server.

This demonstrates how to interact with the LLMling-Agent ACP server using
the Agent Client Protocol over stdin/stdout.
"""

import json
import subprocess
import sys
from typing import Any


class ACPClient:
    """Simple ACP client for testing."""

    def __init__(self, server_command: list[str]):
        """Initialize the client with server command."""
        self.server_command = server_command
        self.process: subprocess.Popen[bytes] | None = None
        self.request_id = 0

    def start_server(self) -> None:
        """Start the ACP server process."""
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        print("🚀 ACP server started")

    def stop_server(self) -> None:
        """Stop the ACP server process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
        print("🛑 Server stopped")

    def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and return the response."""
        if not self.process or not self.process.stdin or not self.process.stdout:
            msg = "Server not started"
            raise RuntimeError(msg)

        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.request_id,
        }

        # Send request
        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line.encode())
        self.process.stdin.flush()

        # Read response
        response_line = self.process.stdout.readline().decode().strip()
        if not response_line:
            msg = "No response from server"
            raise RuntimeError(msg)

        try:
            response = json.loads(response_line)
            if "error" in response:
                msg = f"Server error: {response['error']}"
                raise RuntimeError(msg)
            return response.get("result", {})
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response: {e}"
            raise RuntimeError(msg)  # noqa: B904

    def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.process or not self.process.stdin:
            msg = "Server not started"
            raise RuntimeError(msg)

        notification = {"jsonrpc": "2.0", "method": method, "params": params or {}}

        # Send notification
        notification_line = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_line.encode())
        self.process.stdin.flush()

    def read_notifications(self, timeout_seconds: int = 5) -> list[dict[str, Any]]:
        """Read streaming notifications from the server."""
        import select
        import time

        if not self.process or not self.process.stdout:
            msg = "Server not started"
            raise RuntimeError(msg)

        notifications = []
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            # Check if data is available to read
            ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
            if ready:
                line = self.process.stdout.readline().decode().strip()
                if line:
                    try:
                        notification = json.loads(line)
                        notifications.append(notification)
                        print(f"📢 Received: {notification}")

                        # Check if this is a stop reason (end of streaming)
                        if notification.get(
                            "method"
                        ) == "session/update" and "stopReason" in notification.get(
                            "params", {}
                        ):
                            break
                    except json.JSONDecodeError:
                        print(f"⚠️ Invalid JSON: {line}")
            else:
                # No data available, short sleep
                time.sleep(0.1)

        return notifications


def main() -> int:  # noqa: PLR0915
    """Main function to demonstrate ACP client usage."""
    # Server command - use the binary if available, otherwise use Python module

    # server_cmd = [sys.executable, "-m", "llmling-agent", "serve-acp", "/tmp/acp_test"]
    server_cmd = [
        "uv",
        "run",
        "llmling-agent",
        "serve-acp",
        "/home/phil65/dev/oss/llmling-agent/src/llmling_agent_examples/pick_experts/config.yml",
    ]

    client = ACPClient(server_cmd)

    try:
        # Start server
        client.start_server()

        print("\n🔧 Step 1: Initialize protocol")
        init_result = client.send_request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": True, "writeTextFile": True},
                    "terminal": True,
                },
            },
        )
        print(f"✅ Initialized: {init_result}")

        print("\n🔐 Step 2: Check authentication")
        auth_methods = init_result.get("authMethods", [])
        if auth_methods:
            # If there are auth methods, authenticate with the first one
            method_id = auth_methods[0]["id"]
            auth_result = client.send_request("authenticate", {"methodId": method_id})
            print(f"✅ Authenticated with {method_id}: {auth_result}")
        else:
            print("✅ No authentication required")

        print("\n📝 Step 3: Create new session")
        session_result = client.send_request(
            "session/new", {"cwd": "/tmp", "mcpServers": []}
        )
        session_id = session_result["sessionId"]
        print(f"✅ Session created: {session_id}")

        print("\n💬 Step 4: Send simple prompt")
        prompt_result = client.send_request(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": [
                    {
                        "type": "text",
                        "text": "Hello, just say hi back!",
                    }
                ],
            },
        )
        print(f"✅ Prompt response: {prompt_result}")

        print("\n📡 Step 5: Read streaming updates")
        notifications = client.read_notifications(timeout_seconds=10)
        print(f"✅ Received {len(notifications)} notifications")

        print("\n📁 Step 6: Test file operation (THIS MIGHT HANG!)")
        file_prompt_result = client.send_request(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": [
                    {
                        "type": "text",
                        "text": "Please read README.md file in the current directory.",
                    }
                ],
            },
        )
        print(f"✅ File prompt response: {file_prompt_result}")

        print("\n📡 Step 7: Read file operation streaming updates")
        file_notifications = client.read_notifications(timeout_seconds=30)
        print(f"✅ Received {len(file_notifications)} file operation notifications")

        # Debug: Check server stderr for any errors
        if client.process and client.process.stderr:
            client.process.stderr.flush()
            import fcntl
            import os

            # Make stderr non-blocking
            fd = client.process.stderr.fileno()
            flag = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)
            try:
                stderr_output = client.process.stderr.read()
                if stderr_output:
                    print(f"🔍 Server stderr: {stderr_output.decode()}")
            except BlockingIOError:
                pass  # No stderr output available

        print("\n🎉 ACP client demo completed successfully!")

    except Exception as e:  # noqa: BLE001
        print(f"❌ Error: {e}")
        return 1

    finally:
        client.stop_server()

    return 0


if __name__ == "__main__":
    sys.exit(main())
