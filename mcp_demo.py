"""Repeatable Week 9 MCP discovery and tool-call demonstration.

Usage:
  python mcp_demo.py
  python mcp_demo.py --query "How many paid sick days do employees get?"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


def _json_value(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


async def run(query: str) -> None:
    server = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).with_name("mcp_server.py"))])
    async with Client(server) as client:
        discovered = await client.list_tools()
        print("Discovered MCP tools:")
        for tool in discovered.tools:
            print(f"- {tool.name}: {tool.description}")
            print(json.dumps(_json_value(tool.input_schema), indent=2))

        result = await client.call_tool("search_hr_policy", {"query": query, "top_k": 3})
        if result.is_error:
            raise RuntimeError(f"MCP tool failed: {result.content}")
        print("\nsearch_hr_policy result:")
        print(json.dumps(_json_value(result.structured_content or result.content), indent=2))

        tenure = await client.call_tool("calculate_tenure", {"start_date": "2025-04-02", "as_of_date": "2025-07-11"})
        if tenure.is_error:
            raise RuntimeError(f"MCP tool failed: {tenure.content}")
        print("\ncalculate_tenure result:")
        print(json.dumps(_json_value(tenure.structured_content or tenure.content), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="How many paid sick days do employees get per year?")
    args = parser.parse_args()
    asyncio.run(run(args.query))
