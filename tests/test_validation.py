"""Regression tests: structure validation is delegated to ``jsonschema``."""

import pytest

from atlas.evaluator.validation import validate_structure
from atlas.exceptions import ConfigError
from atlas.models import ToolCallResult, ToolDefinition

TOOL = ToolDefinition(
    name="t",
    description="",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "minLength": 2},
            "days": {"type": "integer", "minimum": 1, "maximum": 14},
            "unit": {"type": "string", "enum": ["c", "f"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "geo": {
                "type": "object",
                "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}},
                "required": ["lat"],
            },
            "id": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            "free": {},
        },
        "required": ["city"],
    },
)


def check(args):
    return validate_structure(ToolCallResult(name="t", arguments=args), TOOL)


def test_valid_call_passes():
    assert check({"city": "Tokyo", "days": 3, "unit": "c", "tags": ["a"],
                  "geo": {"lat": 1.5}, "id": 7, "free": object()}) == (True, "ok")


def test_missing_required_and_unexpected_param_messages():
    ok, msg = check({"bogus": 1})
    assert not ok
    assert "missing required arg: city" in msg
    assert "unexpected parameter: bogus" in msg


@pytest.mark.parametrize("days", ["5", True, 2.5])
def test_integer_rejects_strings_bools_and_fractions(days):
    ok, msg = check({"city": "Tokyo", "days": days})
    assert not ok and "arg days should be integer" in msg


def test_integer_accepts_integral_float():
    assert check({"city": "Tokyo", "days": 5.0})[0]


# Constraints the old hand-rolled validator silently ignored:
def test_numeric_bounds_enforced():
    assert not check({"city": "Tokyo", "days": 99})[0]
    assert not check({"city": "Tokyo", "days": 0})[0]


def test_string_constraint_enforced():
    assert not check({"city": "T"})[0]


def test_oneof_enforced():
    assert check({"city": "Tokyo", "id": "abc"})[0]
    assert not check({"city": "Tokyo", "id": 1.5})[0]


def test_enum_and_array_items_with_paths():
    ok, msg = check({"city": "Tokyo", "unit": "k", "tags": ["a", 3]})
    assert not ok
    assert "arg unit must be one of ['c', 'f']" in msg
    assert "arg tags[1] should be string, got int" in msg


def test_nested_object_required_and_type_with_paths():
    ok, msg = check({"city": "Tokyo", "geo": {"lon": "x"}})
    assert not ok
    assert "missing required arg: geo.lat" in msg
    assert "arg geo.lon should be number, got str" in msg


def test_all_errors_reported_not_just_first():
    ok, msg = check({"days": "5", "unit": "k"})
    assert not ok and msg.count(";") >= 2


def test_top_level_is_strict_by_default_but_schema_can_opt_out():
    assert not check({"city": "Tokyo", "extra": 1})[0]
    lax = ToolDefinition("t", "", {"type": "object", "properties": {"a": {"type": "string"}},
                                   "additionalProperties": True})
    assert validate_structure(ToolCallResult(name="t", arguments={"a": "x", "b": 1}), lax)[0]


def test_tool_without_parameters_rejects_any_args_and_accepts_none():
    bare = ToolDefinition("t", "")
    assert validate_structure(ToolCallResult(name="t", arguments={}), bare)[0]
    assert not validate_structure(ToolCallResult(name="t", arguments={"x": 1}), bare)[0]


def test_invalid_tool_schema_is_a_config_error_not_a_model_failure():
    broken = ToolDefinition("t", "", {"type": "object", "properties": {"a": {"type": "strng"}}})
    with pytest.raises(ConfigError, match="invalid JSON Schema for tool 't'"):
        validate_structure(ToolCallResult(name="t", arguments={"a": "x"}), broken)


def test_parse_error_unknown_tool_and_nameless_still_fail():
    assert validate_structure(ToolCallResult(name="t", arguments={}, parse_error="bad"), TOOL)[1] \
        == "parse_error: bad"
    assert "unknown tool" in validate_structure(ToolCallResult(name="x"), None)[1]
    assert validate_structure(ToolCallResult(name=None), TOOL) == (False, "missing tool name")


def test_custom_suite_rejects_non_array_root(tmp_path):
    import json
    import pytest

    from atlas.atlas_runner.case_loading import load_cases
    from atlas.exceptions import ConfigError

    suite_path = tmp_path / "invalid-suite.json"
    suite_path.write_text(json.dumps({"id": "not-an-array"}), encoding="utf-8")

    with pytest.raises(ConfigError, match="JSON array"):
        load_cases(str(suite_path))
