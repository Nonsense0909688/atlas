"""Schema validation + hallucination detection.

Two jobs, both operating on a single ``ToolCallResult``:

- ``validate_structure`` — does this call conform to the tool's JSON Schema?
  Checks: tool exists → no parse_error → arguments is an object → full JSON
  Schema validation via the ``jsonschema`` library (types, enum, required,
  nested properties/items, oneOf/anyOf, min/max, patterns, …). Returns
  ``(ok, reason)``; every problem found is joined into the reason string
  that goes into the report details.

- ``_hallucination_rate`` — what fraction of calls invented something?
  A call is hallucinated if its tool name isn't registered OR it supplies
  a top-level parameter the schema doesn't declare. Signals "model made
  stuff up" vs "model was just wrong about values" (the latter hits
  argument_accuracy instead).

Top-level schemas default to ``additionalProperties: false`` so invented
params are caught; set it explicitly in a tool schema to allow extras.
"""

from __future__ import annotations

import logging
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

from atlas.exceptions import ConfigError
from atlas.models import ToolCallResult

logger = logging.getLogger(__name__)


def _type_name(value: Any) -> str:
    """Python type name for error messages, e.g. ``str`` for ``"hello"``."""
    return type(value).__name__


def _extra_params(arguments: dict, schema: dict) -> set[str]:
    """Top-level argument names the schema doesn't declare.

    Empty if the schema explicitly allows extras
    (``additionalProperties`` true or a sub-schema).
    """
    if schema.get("additionalProperties", False) is not False:
        return set()
    return set(arguments) - set(schema.get("properties", {}) or {})


def _fmt_path(path) -> str:
    """``deque(["a", 0, "b"])`` → ``a[0].b``."""
    out = ""
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _format_error(err) -> list[str]:
    """Turn one ``jsonschema`` error into Atlas-style one-line messages."""
    path = _fmt_path(err.absolute_path)
    prefix = f"{path}." if path else ""
    if err.validator == "required":
        return [f"missing required arg: {prefix}{r}"
                for r in err.validator_value if r not in err.instance]
    if err.validator == "additionalProperties":
        known = set(err.schema.get("properties", {}) or {})
        return [f"unexpected parameter: {prefix}{k} (not in schema)"
                for k in err.instance if k not in known]
    if err.validator == "type":
        want = err.validator_value
        want = "|".join(want) if isinstance(want, list) else want
        return [f"arg {path} should be {want}, got {_type_name(err.instance)}"]
    if err.validator == "enum":
        return [f"arg {path} must be one of {err.validator_value}, got {err.instance!r}"]
    return [f"arg {path}: {err.message}" if path else err.message]


def _tool_validator(tool: Any):
    """Build a ``jsonschema`` validator for a tool's ``parameters`` schema.

    Tool parameter schemas are strict at the top level (undeclared params are
    errors unless the schema sets ``additionalProperties`` itself); nested
    objects follow normal JSON Schema defaults. A malformed tool schema is a
    bug in the test suite, not the model, so it raises ``ConfigError``.
    """
    schema = dict(getattr(tool, "parameters", {}) or {})
    schema.setdefault("type", "object")
    schema.setdefault("additionalProperties", False)
    cls = validator_for(schema, default=Draft202012Validator)
    try:
        cls.check_schema(schema)
    except SchemaError as e:
        raise ConfigError(f"invalid JSON Schema for tool {tool.name!r}: {e.message}") from e
    return cls(schema)


def validate_structure(call: ToolCallResult, tool: Any) -> tuple[bool, str]:
    """Validate one tool call against its tool definition.

    Args:
        call: The model's emitted call (name + arguments + parse_error).
        tool: Matching ``ToolDefinition``, or ``None`` if unregistered.

    Returns:
        ``(True, "ok")`` if valid, else ``(False, human-readable reason)``.
        The reason is surfaced in ``EvalResult.details`` and error analysis.
    """
    if getattr(call, "parse_error", None):
        return False, f"parse_error: {call.parse_error}"
    if not call.name:
        return False, "missing tool name"
    if tool is None:
        return False, f"unknown tool: {call.name} (not in registered tools)"

    args = call.arguments
    if not isinstance(args, dict):
        return False, f"arguments should be an object, got {_type_name(args)}"

    errors = [
        msg
        for err in _tool_validator(tool).iter_errors(args)
        for msg in _format_error(err)
    ]
    if errors:
        logger.debug("validate %s: %s", call.name, errors)
        return False, "; ".join(errors)
    return True, "ok"


def _hallucination_rate(
    actual_calls: list[ToolCallResult],
    tools_by_name: dict,
) -> tuple[float, list[str]]:
    """Fraction of calls with unknown tools or unexpected parameters.

    Returns:
        ``(rate, details)`` where rate is 0..1 (0.0 when no calls) and
        details are human-readable strings like ``"hallucinated tool: X"``
        or ``"get_weather: unexpected params {'forecast_days'}"``.
    """
    if not actual_calls:
        return 0.0, []

    hallucinated = 0
    details: list[str] = []
    for call in actual_calls:
        if not call.name or call.name not in tools_by_name:
            hallucinated += 1
            details.append(f"hallucinated tool: {call.name}")
            continue
        schema = getattr(tools_by_name[call.name], "parameters", {}) or {}
        args = call.arguments if isinstance(call.arguments, dict) else {}
        extra = _extra_params(args, schema)
        if extra:
            hallucinated += 1
            details.append(f"{call.name}: unexpected params {extra}")

    rate = hallucinated / len(actual_calls)
    if hallucinated:
        logger.debug("hallucination rate=%.2f (%d/%d calls)", rate, hallucinated, len(actual_calls))
    return rate, details
