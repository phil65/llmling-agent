"""Auto-generated metadata file."""

from typing import Literal


# This file is generated from https://raw.githubusercontent.com/agentclientprotocol/agent-client-protocol/6b904666187549412bab891ff3d8846a4c8cfd87/schema/meta.json.
# Do not edit by hand.

AgentMethod = Literal[
    "authenticate",
    "initialize",
    "session/cancel",
    "session/load",
    "session/new",
    "session/prompt",
    "session/set_mode",
    "session/set_model",
]

ClientMethod = Literal[
    "fs/read_text_file",
    "fs/write_text_file",
    "session/request_permission",
    "session/update",
    "terminal/create",
    "terminal/kill",
    "terminal/output",
    "terminal/release",
    "terminal/wait_for_exit",
]

PROTOCOL_VERSION = 1
