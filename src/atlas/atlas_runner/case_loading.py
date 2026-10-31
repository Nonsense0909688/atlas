"""Loading and filtering built-in or user-defined benchmark cases."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from atlas.cases.builtin import get_builtin_cases
from atlas.exceptions import ConfigError
from atlas.models import ExpectedToolCall, TestCase, ToolDefinition

logger = logging.getLogger(__name__)


def load_cases(suite_path: str | None) -> list[TestCase]:
    """Return the test suite: built-in, or parsed from a JSON file.

    Raises:
        ConfigError: Suite file missing, unreadable, or malformed.
    """
    if not suite_path:
        cases = get_builtin_cases()
        logger.info("loaded %d built-in cases", len(cases))
        return cases
    if suite_path in ("full", "extended"):
        from atlas.cases.builtin import get_builtin_cases as _get_b
        from atlas.cases.domains import get_extended_cases as _get_e
        cases = _get_b() + _get_e()
        logger.info("loaded %d full cases", len(cases))
        return cases
    try:
        data = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"suite file not found: {suite_path}") from e
    except UnicodeDecodeError as e:
        raise ConfigError(f"suite file is not UTF-8 text: {suite_path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"can't read suite file {suite_path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"suite file is not valid JSON: {suite_path}: {e}") from e

    if not isinstance(data, list):
        raise ConfigError(f"suite file must contain a JSON array of cases: {suite_path}")

    cases: list[TestCase] = []
    for index, raw_case in enumerate(data, start=1):
        if not isinstance(raw_case, dict):
            raise ConfigError(f"case #{index} in {suite_path} must be a JSON object")
        try:
            cases.append(_case_from_json(raw_case))
        except (KeyError, TypeError, ValueError) as e:
            raise ConfigError(f"invalid case #{index} in {suite_path}: {e}") from e
    logger.info("loaded %d cases from suite %s", len(cases), suite_path)
    return cases


def filter_cases(
    cases: list[TestCase],
    *,
    case_id: str | None,
    category: str | None,
    difficulty: str | None,
) -> list[TestCase]:
    """Apply --case / --category / --difficulty filters.

    Raises:
        ConfigError: A filter matched zero cases (likely a typo).
    """
    if case_id:
        cases = [c for c in cases if c.id == case_id]
        if not cases:
            raise ConfigError(f"case not found: {case_id}")
    if category:
        cases = [c for c in cases if c.category == category]
        if not cases:
            raise ConfigError(f"no cases in category: {category}")
    if difficulty:
        cases = [c for c in cases if c.difficulty == difficulty]
        if not cases:
            raise ConfigError(f"no cases at difficulty: {difficulty}")
    return cases


def _case_from_json(c: dict) -> TestCase:
    """Convert one JSON suite entry into a ``TestCase``."""
    tools = [
        ToolDefinition(
            name=t["name"],
            description=t.get("description", ""),
            parameters=t.get("parameters", {}),
            output_schema=t.get("output_schema", {}),
            destructive=bool(t.get("destructive", False)),
            requires_confirmation=bool(t.get("requires_confirmation", False)),
        )
        for t in c.get("tools", [])
    ]
    expected = [
        ExpectedToolCall(name=e["name"], arguments=e.get("arguments", {}))
        for e in c.get("expected_calls", [])
    ]
    # alternatives: list of sequences, each list of {name, arguments}
    alts = c.get("alternative_sequences") or c.get("alternative_sequences_json") or []
    return TestCase(
        id=c["id"],
        prompt=c["prompt"],
        tools=tools,
        expected_calls=expected,
        mock_outputs=c.get("mock_outputs", {}),
        category=c.get("category", "custom"),
        difficulty=c.get("difficulty", "easy"),
        rubric=c.get("rubric", []),
        matchers=c.get("matchers", {}),
        alternative_sequences=alts,
        multi_turn=bool(c.get("multi_turn", False)),
        max_turns=int(c.get("max_turns", 5)),
        initial_state=c.get("initial_state", {}),
        expected_state=c.get("expected_state", {}),
        final_answer=c.get("final_answer"),
        final_answer_matcher=c.get("final_answer_matcher", {}),
        allowed_tools=c.get("allowed_tools"),
        forbidden_tools=c.get("forbidden_tools", []),
        mock_errors=c.get("mock_errors", {}),
        injection=c.get("injection", {}),
        language=c.get("language", "en"),
        split=c.get("split", "train"),
        version=c.get("version", "2.0.0"),
        requires_confirmation=bool(c.get("requires_confirmation", False)),
    )
