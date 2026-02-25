"""MCP Streamable HTTP Client"""

import argparse
import asyncio
import httpx
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class MCPHttpClient:
    """MCP Client for interacting with an MCP Streamable HTTP server"""

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.init_connect = False,
        self.mcp_url = None
        self.headers = {}

    async def connect_to_streamable_http_server(
        self, base_url:str, server_url: str, api_key:str
    ):
        self.mcp_url = server_url
        self.api_key = api_key


        transport_client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )

        """Connect to an MCP server running with HTTP Streamable transport"""
        self._streams_context = streamable_http_client(  # pylint: disable=W0201
            url=server_url,
            http_client= transport_client,
        )
        read_stream, write_stream, _ = await self._streams_context.__aenter__()  # pylint: disable=E1101

        self._session_context = ClientSession(read_stream, write_stream)  # pylint: disable=W0201
        self.session: ClientSession = await self._session_context.__aenter__()  # pylint: disable=C2801

        await self.session.initialize()
        self.init_connect = True

    async def call_tool(self, tool_name: str,**tool_args) -> str:
        """Process a query using Claude and available tools"""
        if not self.init_connect:
            await self.connect_to_streamable_http_server(self.mcp_url,self.api_key)
        # Execute tool call
        result = await self.session.call_tool(tool_name, tool_args)
        return result


    async def list_tools(self):
        if not self.init_connect:
            await self.connect_to_streamable_http_server(self.mcp_url,self.api_key)

        response = await self.session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        return available_tools





    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:  # pylint: disable=W0125
            await self._streams_context.__aexit__(None, None, None)  # pylint: disable=E1101

        self.init_connect = False

