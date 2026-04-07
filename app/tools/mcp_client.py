import os
import sys
from typing import Any, Dict

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class MCPClient:
    def __init__(self) -> None:
        self._workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    async def _call_tool_async(self, name: str, arguments: Dict[str, Any]) -> str:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.tools.mcp_server"],
            cwd=self._workspace,
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)

                content = getattr(result, "content", None)
                if content and isinstance(content, list):
                    first = content[0]
                    text = getattr(first, "text", None)
                    if text is not None:
                        return str(text)
                return str(result)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        return anyio.run(self._call_tool_async, name, arguments)


mcp_client = MCPClient()
