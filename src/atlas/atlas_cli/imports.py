"""Commands that import external datasets and tool specifications."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atlas.atlas_cli.constants import EXIT_OK
from atlas.exceptions import ConfigError


def _import_dataset(kind: str, source: str) -> list[dict]:
    """Convert a supported external dataset into Atlas case dictionaries."""
    from atlas.datasets import import_dataset

    return import_dataset(kind, source)


def _import_openapi(source: str) -> list[dict]:
    """Load an OpenAPI document and translate its operations into test cases."""
    from atlas.tools.openapi_import import load_openapi, openapi_to_suite

    specification = load_openapi(source)
    return openapi_to_suite(specification, Path(source).stem)


def _serialize_case(case) -> dict:
    """Convert a generated MCP case into the custom-suite JSON shape."""
    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in case.tools
    ]
    return {
        "id": case.id,
        "prompt": case.prompt,
        "tools": tools,
        "expected_calls": [],
        "mock_outputs": case.mock_outputs,
        "category": case.category,
        "difficulty": case.difficulty,
    }


def _import_mcp(source: str) -> list[dict]:
    """Discover an MCP server's tools and create reviewable starter cases."""
    from atlas.tools.mcp import generate_cases_from_tools, list_mcp_tools_http

    tools = list_mcp_tools_http(source)
    cases = generate_cases_from_tools(
        tools,
        domain=Path(source).stem or "mcp",
    )
    return [_serialize_case(case) for case in cases]


def _import_cmd(args: argparse.Namespace) -> int:
    """Implement ``atlas import`` and save the resulting suite as JSON."""
    if args.kind in ("bfcl", "tau-bench", "toolbench"):
        suite = _import_dataset(args.kind, args.source)
    elif args.kind == "openapi":
        suite = _import_openapi(args.source)
    elif args.kind == "mcp":
        suite = _import_mcp(args.source)
    else:
        raise ConfigError(f"unknown import kind: {args.kind}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(suite, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"imported {len(suite)} cases to {output_path}",
        file=sys.stderr,
    )
    return EXIT_OK
