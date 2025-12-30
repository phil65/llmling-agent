"""Diagnostics module for code analysis via LSP and CLI tools.

This module provides:
- LSP server lifecycle management via LSPManager
- LSP proxy for stdio-based servers exposed via Unix socket
- Integration with execution environments (local, remote, sandboxed)
"""

from agentpool.diagnostics.lsp_manager import LSPManager
from agentpool.diagnostics.lsp_proxy import LSPProxy
from agentpool.diagnostics.models import (
    COMPLETION_KIND_MAP,
    CODE_ACTION_KIND_MAP,
    SYMBOL_KIND_MAP,
    CallHierarchyCall,
    CallHierarchyItem,
    CodeAction,
    CompletionItem,
    Diagnostic,
    DiagnosticsResult,
    HoverInfo,
    Location,
    LSPServerState,
    Position,
    Range,
    RenameResult,
    SignatureInfo,
    SymbolInfo,
    TypeHierarchyItem,
)

__all__ = [
    "CODE_ACTION_KIND_MAP",
    "COMPLETION_KIND_MAP",
    "SYMBOL_KIND_MAP",
    "CallHierarchyCall",
    "CallHierarchyItem",
    "CodeAction",
    "CompletionItem",
    "Diagnostic",
    "DiagnosticsResult",
    "HoverInfo",
    "LSPManager",
    "LSPProxy",
    "LSPServerState",
    "Location",
    "Position",
    "Range",
    "RenameResult",
    "SignatureInfo",
    "SymbolInfo",
    "TypeHierarchyItem",
]
