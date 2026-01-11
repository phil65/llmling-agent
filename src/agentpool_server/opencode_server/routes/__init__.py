"""Route handlers for the OpenCode API."""

from agentpool_server.opencode_server.routes.global_routes import router as global_router
from agentpool_server.opencode_server.routes.app_routes import router as app_router
from agentpool_server.opencode_server.routes.config_routes import router as config_router
from agentpool_server.opencode_server.routes.session_routes import router as session_router
from agentpool_server.opencode_server.routes.message_routes import router as message_router
from agentpool_server.opencode_server.routes.file_routes import router as file_router
from agentpool_server.opencode_server.routes.agent_routes import router as agent_router
from agentpool_server.opencode_server.routes.permission_routes import router as permission_router
from agentpool_server.opencode_server.routes.question_routes import router as question_router
from agentpool_server.opencode_server.routes.pty_routes import router as pty_router
from agentpool_server.opencode_server.routes.tui_routes import router as tui_router
from agentpool_server.opencode_server.routes.lsp_routes import router as lsp_router

__all__ = [
    "agent_router",
    "app_router",
    "config_router",
    "file_router",
    "global_router",
    "lsp_router",
    "message_router",
    "permission_router",
    "pty_router",
    "question_router",
    "session_router",
    "tui_router",
]
