"""Normalize provider responses and recover structured calls from text."""

from __future__ import annotations

import json
import logging

from atlas.models import ToolCallResult

logger = logging.getLogger(__name__)


def response_to_tool_calls(chat_response) -> list[ToolCallResult]:
    """Normalize a ``ChatResponse`` into ``ToolCallResult`` list.

    Structured calls come first; if the model returned none but emitted
    JSON-looking text, we attempt a salvage parse (small models often
    "say" the call instead of making it properly).
    """
    results = [
        ToolCallResult(
            name=call.get("name"),
            arguments=call.get("arguments", {}),
            parse_error=call.get("parse_error"),
            raw=call.get("raw"),
        )
        for call in chat_response.tool_calls
    ]
    if not results and chat_response.text:
        salvaged = salvage_tool_calls_from_text(chat_response.text)
        if salvaged:
            logger.debug("salvaged %d tool call(s) from free text", len(salvaged))
        results.extend(salvaged)
    return results


def salvage_tool_calls_from_text(text: str) -> list[ToolCallResult]:
    """Extract tool-call-shaped JSON objects from free-form model output.

    A JSON decoder is used instead of regular expressions so nested objects,
    arrays, escaped quotes, and braces inside strings are handled correctly.
    Supported forms include Atlas-style calls and common OpenAI-style wrappers.
    """
    decoder = json.JSONDecoder()
    results: list[ToolCallResult] = []
    index = 0

    while index < len(text):
        if text[index] not in "{[":
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue

        extracted = _tool_calls_from_json(value)
        results.extend(extracted)
        # Skip the complete decoded value. This avoids parsing the same nested
        # tool call twice when its wrapper already contained it.
        index = max(end, index + 1)

    return results


def _tool_calls_from_json(value: object) -> list[ToolCallResult]:
    """Normalize supported JSON call envelopes recursively."""
    if isinstance(value, list):
        return [call for item in value for call in _tool_calls_from_json(item)]
    if not isinstance(value, dict):
        return []

    function = value.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
        raw = function
    elif isinstance(function, str):
        name = function
        arguments = value.get("args", value.get("arguments", {}))
        raw = value
    else:
        name = value.get("name")
        arguments = value.get("arguments", value.get("args", {}))
        raw = value

    if isinstance(name, str) and name.strip():
        parse_error = None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                parse_error = f"invalid JSON arguments: {exc.msg}"
        return [ToolCallResult(
            name=name,
            arguments=arguments,
            raw=raw,
            parse_error=parse_error,
        )]

    # Common wrappers: {"tool_calls": [...]} and response containers.
    for key in ("tool_calls", "calls", "content", "response"):
        nested = value.get(key)
        if isinstance(nested, (dict, list)):
            calls = _tool_calls_from_json(nested)
            if calls:
                return calls
    return []
