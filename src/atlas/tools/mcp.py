"""MCP support: list tools from MCP server, generate cases/tool defs."""

from __future__ import annotations

import json
import logging

import requests

from atlas.models import TestCase, ToolDefinition

logger = logging.getLogger(__name__)


def list_mcp_tools_http(server_url: str, timeout: int = 30) -> list[ToolDefinition]:
    """List tools via MCP HTTP transport (JSON-RPC)."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    resp = requests.post(server_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    tools = data.get("result", {}).get("tools", []) or data.get("tools", [])
    out: list[ToolDefinition] = []
    for t in tools:
        schema = t.get("inputSchema") or t.get("parameters") or {"type": "object", "properties": {}}
        out.append(ToolDefinition(name=t.get("name", "unknown"), description=t.get("description", ""), parameters=schema))
    return out


def list_mcp_tools_stdio(command: str) -> list[ToolDefinition]:
    """Stub for stdio transport: document that real impl spawns process + JSON-RPC."""
    logger.info("MCP stdio command %s not spawned in this build; use HTTP transport or implement subprocess.", command)
    return []


def to_tool_definitions(mcp_tools) -> list[ToolDefinition]:
    return mcp_tools


def generate_cases_from_tools(tools: list[ToolDefinition], domain: str = "mcp") -> list[TestCase]:
    cases: list[TestCase] = []
    for i, t in enumerate(tools):
        props = t.parameters.get("properties", {}) if isinstance(t.parameters, dict) else {}
        # Build a plausible prompt + expected args with dummy values
        args = {}
        for k, schema in props.items():
            typ = schema.get("type", "string") if isinstance(schema, dict) else "string"
            args[k] = "example" if typ == "string" else 1 if typ in ("integer", "number") else True if typ == "boolean" else []
        cases.append(TestCase(
            id=f"mcp-{domain}-{t.name}",
            prompt=f"Use the {t.name} tool. " + (f"Description: {t.description}" if t.description else ""),
            tools=tools,
            expected_calls=[],  # caller fills
            mock_outputs={tt.name: {"ok": True} for tt in tools},
            category=f"mcp_{domain}",
            difficulty="medium",
        ))
        # Only generate first few to avoid explosion; full suite if needed
        if len(cases) > 20:
            break
    return cases
