"""Dataset importers: BFCL, tau-bench, ToolBench."""

from __future__ import annotations

import json
from pathlib import Path


def bfcl_to_suite(path: str) -> list[dict]:
    """BFCL JSONL: {'question': ..., 'function': ..., 'ground_truth': ...}."""
    suite = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        func = rec.get("function") or rec.get("functions") or []
        if isinstance(func, dict):
            func = [func]
        tools = []
        for f in func:
            tools.append({"name": f.get("name", "func"), "description": f.get("description", ""), "parameters": f.get("parameters", {})})
        gt = rec.get("ground_truth") or rec.get("expected_calls") or []
        expected = []
        if isinstance(gt, dict):
            expected = [{"name": gt.get("name", tools[0]["name"] if tools else "func"), "arguments": gt.get("arguments", {})}]
        elif isinstance(gt, list):
            for g in gt:
                if isinstance(g, dict):
                    expected.append({"name": g.get("name", tools[0]["name"] if tools else "func"), "arguments": g.get("arguments", {})})
        suite.append({
            "id": rec.get("id", f"bfcl-{len(suite)}"),
            "prompt": rec.get("question") or rec.get("prompt", ""),
            "tools": tools,
            "expected_calls": expected,
            "mock_outputs": {t["name"]: {"ok": True} for t in tools},
            "category": "bfcl",
            "difficulty": "medium",
        })
    return suite


def tau_bench_to_suite(path: str) -> list[dict]:
    """tau-bench tasks JSON/YAML-ish: list of {user_prompt, tools, ...}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]
    suite = []
    for i, rec in enumerate(data):
        tools = [{"name": t.get("name"), "description": t.get("description", ""), "parameters": t.get("parameters", t.get("input_schema", {}))} for t in rec.get("tools", [])] if isinstance(rec.get("tools"), list) else []
        suite.append({
            "id": rec.get("id", f"tau-{i}"),
            "prompt": rec.get("user_prompt") or rec.get("prompt", ""),
            "tools": tools,
            "expected_calls": rec.get("expected_calls", []),
            "mock_outputs": rec.get("mock_outputs", {}),
            "category": "tau_bench",
            "difficulty": rec.get("difficulty", "hard"),
            "initial_state": rec.get("initial_state", {}),
            "expected_state": rec.get("expected_state", {}),
            "final_answer": rec.get("final_answer"),
        })
    return suite


def toolbench_to_suite(path: str) -> list[dict]:
    """ToolBench: queries with tools."""
    suite = []
    # ToolBench often JSON list of {'query':..., 'tools': [...]}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data", [data])
    for i, rec in enumerate(data):
        tools = []
        for t in rec.get("tools", rec.get("available_tools", [])):
            if isinstance(t, str):
                tools.append({"name": t, "description": "", "parameters": {"type": "object", "properties": {}}})
            else:
                tools.append({"name": t.get("name", f"tool_{len(tools)}"), "description": t.get("description", ""), "parameters": t.get("parameters", {})})
        suite.append({
            "id": rec.get("id", f"toolbench-{i}"),
            "prompt": rec.get("query") or rec.get("prompt", ""),
            "tools": tools,
            "expected_calls": rec.get("expected_calls", []),
            "mock_outputs": {t["name"]: {"ok": True} for t in tools},
            "category": "toolbench",
            "difficulty": "medium",
        })
    return suite


def import_dataset(kind: str, path: str) -> list[dict]:
    if kind == "bfcl":
        return bfcl_to_suite(path)
    if kind == "tau-bench":
        return tau_bench_to_suite(path)
    if kind == "toolbench":
        return toolbench_to_suite(path)
    raise ValueError(f"unknown dataset: {kind} (known: bfcl, tau-bench, toolbench)")
