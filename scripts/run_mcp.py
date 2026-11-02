"""
scripts/run_mcp.py
Entrypoint script to start the FastMCP server for the slow loop.
"""

from src.mcp.server import mcp

if __name__ == "__main__":
    print("Starting Constellation Control FastMCP Server...")
    mcp.run()