"""Tool-count scaling: pad case with filler tools."""

from __future__ import annotations

import copy

from atlas.models import ToolDefinition


def scale_tools(case, target_count: int):
    """Return a copy of case with filler tools so total tools == target_count."""
    if len(case.tools) >= target_count:
        return case
    new_case = copy.deepcopy(case)
    existing = {t.name for t in new_case.tools}
    i = 0
    while len(new_case.tools) < target_count:
        i += 1
        name = f"filler_tool_{i}"
        if name in existing:
            continue
        new_case.tools.append(ToolDefinition(
            name=name,
            description=f"Utility tool number {i} for scaling test. Does a generic operation on generic data.",
            parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
        ))
    return new_case
