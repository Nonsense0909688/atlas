"""Mock tool sandbox — fake, safe tool execution for evaluations.

Purpose: real tools (databases, filesystems, APIs) have side effects and
vary between runs. The sandbox gives every tool a deterministic canned
output so benchmarks are reproducible and safe to run even on a laptop.

Usage::

    toolbox = default_registry(case)          # pre-loaded from case.mock_outputs
    result = toolbox.execute("get_weather", {"city": "Tokyo"})
    result  # → {"city": "Tokyo", "temp_c": 22, ...} (the canned output)

Unknown tools return ``{"error": "unknown_tool:…"}`` instead of raising,
so multi-turn research code can surface errors gracefully.

Note: the current scorer only grades first-turn calls, so the sandbox is
not yet on the hot path of ``atlas run``. It's here for multi-turn cases
(dependency chains, conversation) and as the single place to swap in
richer mock behaviors later.
"""

from __future__ import annotations

import logging
from typing import Any

from atlas.models import TestCase

logger = logging.getLogger(__name__)


class MockToolbox:
    """Registry of mock tools. Each tool maps to a canned output value."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, output: Any, description: str = "") -> None:
        """Register (or overwrite) one mock tool's canned output."""
        self._tools[name] = {"output": output, "description": description}
        logger.debug("sandbox: registered tool %r", name)

    def register_case(self, case: TestCase) -> None:
        """Load all of a test case's ``mock_outputs`` into this toolbox.

        Existing registrations win (first registration is kept), so a
        shared toolbox can layer multiple cases without clobbering.
        """
        for name, output in case.mock_outputs.items():
            if name not in self._tools:
                self.register(name, output)

    def execute(self, name: str, arguments: dict | None = None) -> Any:
        """Return the canned output for ``name``.

        Arguments are accepted (for API parity with real tools) but ignored
        — mocks are static. Unknown tools return a structured error dict.
        """
        if name not in self._tools:
            logger.warning("sandbox: unknown tool %r (available: %s)",
                           name, list(self._tools.keys()))
            return {"error": f"unknown_tool:{name}", "available": list(self._tools.keys())}
        return self._tools[name]["output"]

    def list_tools(self) -> list[str]:
        """Names of all registered mock tools."""
        return list(self._tools.keys())


def default_registry(case: TestCase) -> MockToolbox:
    """Convenience: a fresh toolbox pre-loaded with one case's mock outputs."""
    toolbox = MockToolbox()
    toolbox.register_case(case)
    return toolbox


def execute_tool(
    name: str,
    arguments: dict,
    case: TestCase | None = None,
    toolbox: MockToolbox | None = None,
) -> Any:
    """One-shot helper: execute a tool against a case's mocks.

    Uses the provided toolbox, or builds one from ``case`` if omitted.
    """
    if toolbox is None:
        toolbox = default_registry(case) if case else MockToolbox()
    return toolbox.execute(name, arguments)
