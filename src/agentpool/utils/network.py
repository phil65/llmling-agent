from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import socket


def _create_socket(preferred_port: int) -> tuple[socket.socket, int]:
    """Create a socket bound to a free port.

    Returns:
        A tuple containing the socket and the port it is bound to.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        # Try preferred port first
        sock.bind(("127.0.0.1", preferred_port))
    except OSError:
        # Fall back to any available port
        sock.bind(("127.0.0.1", 0))
        return sock, sock.getsockname()[1]
    else:
        return sock, preferred_port
