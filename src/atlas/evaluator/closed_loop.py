"""Mock tool execution and state-based grading for multi-turn cases."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from atlas.models import TestCase

logger = logging.getLogger(__name__)


class StatefulToolbox:
    """Execute mock tools while recording their state changes and outputs."""

    def __init__(
        self,
        case: TestCase,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        self.case = case
        starting_state = initial_state if initial_state is not None else case.initial_state
        self.state: dict[str, Any] = dict(starting_state or {})
        self._mock_outputs = dict(case.mock_outputs or {})
        self._mock_errors = dict(case.mock_errors or {})
        self.history: list[dict[str, Any]] = []

    def apply_effect(self, name: str, arguments: dict[str, Any]) -> None:
        """Apply the benchmark's generic and domain-specific state updates."""
        self.state.setdefault("calls", []).append({
            "tool": name,
            "args": arguments,
        })
        self.state[f"last_{name}"] = arguments

        if name in {"book_flight", "book_hotel", "create_booking"}:
            self.state.setdefault("bookings", []).append(arguments)
        elif name in {"send_email", "send_message"}:
            self.state.setdefault("emails_sent", []).append(arguments)
        elif name in {"create_calendar_event", "schedule_event"}:
            self.state.setdefault("events", []).append(arguments)
        elif name in {"delete_file", "delete_record", "delete_all"}:
            self.state.setdefault("deleted", []).append(arguments)
        elif name in {"add_to_cart", "purchase", "checkout"}:
            self.state.setdefault("cart", []).append(arguments)

    @staticmethod
    def _error_result(error_spec: Any) -> dict[str, Any]:
        """Normalize configured mock failure modes into tool results."""
        if isinstance(error_spec, dict) and "type" in error_spec:
            error_type = error_spec.get("type")
            if error_type == "timeout":
                return {"output": None, "error": "timeout after 5s"}
            if error_type == "http_500":
                return {
                    "output": None,
                    "error": "HTTP 500 Internal Server Error",
                }
            if error_type == "empty":
                return {"output": {}, "error": None}
            if error_type == "malformed_json":
                return {
                    "output": "{malformed json",
                    "error": "malformed JSON response",
                }
        return {"output": None, "error": str(error_spec)}

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a configured mock result; never execute a real external tool."""
        error_spec = self._mock_errors.get(name) or self._mock_errors.get("*")
        if error_spec:
            return self._error_result(error_spec)

        output = self._mock_outputs.get(name, {"ok": True})
        injection = (getattr(self.case, "injection", {}) or {}).get("payload")
        if injection and isinstance(output, dict):
            output = dict(output)
            output.setdefault("_injected", injection)

        self.apply_effect(name, arguments)
        result = {
            "tool": name,
            "args": arguments,
            "output": output,
            "error": None,
        }
        self.history.append(result)
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return a detached JSON-safe snapshot of the current mock state."""
        return json.loads(json.dumps(self.state, default=str))


def score_final_answer(
    expected: str | None,
    actual: str | None,
    matcher: dict | None = None,
) -> float | None:
    """Score a final response using exact, regex, or contains matching."""
    if expected is None:
        return None
    if actual is None:
        return 0.0

    matcher = matcher or {}
    matcher_type = matcher.get("type", "contains")
    if matcher_type == "exact":
        return float(expected.strip().lower() == actual.strip().lower())
    if matcher_type == "regex":
        pattern = matcher.get("pattern", expected)
        return 1.0 if re.search(pattern, actual) else 0.0
    return 1.0 if expected.lower() in actual.lower() else 0.0


def _matches_expected_state(expected: Any, actual: Any) -> bool:
    """Recursively compare an expected state fragment with actual state."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _matches_expected_state(value, actual[key])
            for key, value in expected.items()
        )

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        return all(
            _matches_expected_state(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual)
        )

    if isinstance(expected, str) and isinstance(actual, str):
        return expected.lower() == actual.lower()
    return expected == actual


def score_state(
    final_state: dict[str, Any],
    expected_state: dict[str, Any],
) -> float | None:
    """Score whether all required keys and values exist in final mock state."""
    if not expected_state:
        return None
    return 1.0 if _matches_expected_state(expected_state, final_state) else 0.0


def score_recovery(
    turns: list,
    final_answer_score: float | None,
    had_tool_error: bool,
) -> float | None:
    """Score whether the agent recovered after an error in an earlier turn."""
    if not had_tool_error:
        return None
    if final_answer_score is not None:
        return final_answer_score

    for turn_index, turn in enumerate(turns):
        if not turn.tool_errors:
            continue
        for later_turn in turns[turn_index + 1:]:
            for tool_result in later_turn.tool_results:
                if not tool_result.get("error"):
                    return 1.0
    return 0.0
