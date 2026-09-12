"""MCP server package for supergraph.

Exposes supergraph as agent-callable tools over the Model Context Protocol.
The console script `supergraph-mcp` (installed by the [mcp] extra) launches
the stdio server. See `supergraph.mcp.server` for the implementation.
"""
from supergraph.mcp.server import main

__all__ = ["main"]
