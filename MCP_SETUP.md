# Week 9 MCP Setup

## What is running where

`mcp_agent.py` is the **host**: it runs the LLM and decides which action to take.
It starts `mcp_server.py` as an MCP **server** subprocess over stdio. The server
does not run an LLM; it only publishes the reusable read-only tools
`search_hr_policy` and `calculate_tenure`.

## Install and run

Python 3.10+ is required. Install the project dependencies, start Qdrant, and
ensure the HR documents are already ingested:

```powershell
pip install -r requirements.txt
docker start qdrant-hr
python mcp_demo.py
python mcp_agent.py "How many days of paid sick leave do employees get per year?"
```

`mcp_demo.py` proves the client discovers both tools through `tools/list` before
it invokes either one. Add a third `@mcp.tool()` function to `mcp_server.py` and
rerun the demo: it appears in discovery without any edit to `mcp_agent.py`.

## Raw protocol messages

MCP uses JSON-RPC on stdio. The SDK owns these messages; application code must
never print normal output to server stdout. A simplified handshake looks like:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"<SDK-negotiated-version>","capabilities":{},"clientInfo":{"name":"hr-host"}}}
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"<negotiated-version>","capabilities":{"tools":{}},"serverInfo":{"name":"HR Policy MCP Server"}}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
```

Run `python mcp_demo.py` and preserve its discovery output as the evidence for
your mentor. The SDK negotiates the exact protocol version, so do not hard-code
the placeholder values above as a claimed capture.

## Peer validation

A stdio server is local to the machine that has this repository. For the Week 9
peer requirement, have the other person clone the branch and run:

```powershell
pip install -r requirements.txt
python mcp_demo.py --query "What is the probation period for new hires?"
```

Record their name, date, command, discovered tool names, and returned result in
`WEEKLY_PROGRESS.md` only after they actually run it. Do not claim this check
based on a local self-test.

## Security boundary

The server exposes no write, upload, deletion, filesystem, environment, or
arbitrary Qdrant-query tool. `search_hr_policy` bounds query length and result
count, and reuses Week 8's retrieved-content quarantine before sending passages
to a host. `calculate_tenure` accepts only ISO dates.
