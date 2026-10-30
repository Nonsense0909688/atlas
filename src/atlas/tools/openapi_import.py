"""OpenAPI import: turn spec into ToolDefinitions + mock outputs + suite."""

from __future__ import annotations

import json
from pathlib import Path

from atlas.models import TestCase, ToolDefinition


def openapi_to_tools(spec: dict) -> list[ToolDefinition]:
    tools: list[ToolDefinition] = []
    for path, methods in spec.get("paths", {}).items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            name = op.get("operationId") or f"{method}_{path.strip('/').replace('/', '_').replace('{', '').replace('}', '')}"
            if not name:
                continue
            props: dict = {}
            required: list[str] = []
            # path params
            for p in op.get("parameters", []):
                pname = p.get("name")
                if not pname:
                    continue
                schema = p.get("schema", {"type": "string"})
                props[pname] = {"type": schema.get("type", "string")}
                if p.get("required"):
                    required.append(pname)
            # request body
            req = op.get("requestBody", {}).get("content", {}).get("application/json", {})
            body_schema = req.get("schema", {})
            if isinstance(body_schema, dict) and "properties" in body_schema:
                for k, v in body_schema["properties"].items():
                    props[k] = {"type": v.get("type", "string")}
                required.extend(body_schema.get("required", []))
            # query params already covered
            parameters = {"type": "object", "properties": props}
            if required:
                parameters["required"] = list(dict.fromkeys(required))
            tools.append(ToolDefinition(
                name=name,
                description=op.get("summary") or op.get("description") or f"{method.upper()} {path}",
                parameters=parameters,
            ))
    return tools


def openapi_to_suite(spec: dict, spec_name: str = "openapi") -> list[dict]:
    tools = openapi_to_tools(spec)
    suite = []
    for t in tools:
        suite.append({
            "id": f"{spec_name}-{t.name}",
            "prompt": f"Call {t.name}: {t.description}",
            "tools": [{"name": tt.name, "description": tt.description, "parameters": tt.parameters} for tt in tools],
            "expected_calls": [],
            "mock_outputs": {tt.name: {"ok": True} for tt in tools},
            "category": f"openapi_{spec_name}",
            "difficulty": "medium",
        })
    return suite


def load_openapi(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
