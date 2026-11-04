"""Tests for arguments."""

import pytest
from atlas.evaluator.scoring import score_case
from atlas.models import ToolCallResult


def test_wrong_args(get_case):
    case = get_case("weather-single")
    actual = [ToolCallResult(name="get_weather", arguments={"city": "Berlin"})]
    r = score_case(case, actual, model_id="test")
    assert r.scores.tool_selection == 1.0
    assert r.scores.argument_accuracy == 0.0
    assert r.scores.argument_precision == 0.0
    assert r.scores.argument_recall == 0.0
    assert r.scores.overall < 1.0


def test_case_insensitive_args(get_case):
    case = get_case("weather-with-units")
    actual = [ToolCallResult(name="get_weather", arguments={"city": "paris", "units": "Fahrenheit"})]
    r = score_case(case, actual, model_id="test")
    assert r.scores.argument_accuracy == 1.0
    assert r.scores.argument_precision == 1.0
    assert r.scores.argument_recall == 1.0


def test_missing_required_arg(get_case):
    case = get_case("flight-complex")
    actual = [ToolCallResult(name="book_flight", arguments={"origin": "NYC"})]
    r = score_case(case, actual, model_id="test")
    assert r.scores.structure_validity == 0.0


def test_precision_recall_partial(get_case):
    """Model provides 1 correct arg out of 2 expected, plus 1 hallucinated extra."""
    case = get_case("flight-complex")
    actual = [ToolCallResult(name="book_flight", arguments={
        "origin": "NYC", "destination": "LAX", "extra_param": "bogus"
    })]
    r = score_case(case, actual, model_id="test")
    # 2 out of 3 emitted args are correct → precision 0.67
    assert r.scores.argument_precision == pytest.approx(2 / 3, abs=0.01)
    # 2 out of 4 expected provided → recall 0.5
    assert r.scores.argument_recall == pytest.approx(0.5, abs=0.01)


def test_precision_extra_args_penalized(get_case):
    """More args than expected → precision drops."""
    case = get_case("weather-single")
    actual = [ToolCallResult(name="get_weather", arguments={
        "city": "Tokyo", "units": "celsius", "humidity": "high"
    })]
    r = score_case(case, actual, model_id="test")
    # 2 of 3 emitted are valid (city+units in schema, humidity not)
    assert r.scores.argument_precision < 1.0


def test_type_safety_number(get_case):
    """limit as string instead of number should fail structure."""
    case = get_case("search-single")
    actual = [ToolCallResult(name="search_web", arguments={"query": "quantum computing", "limit": "5"})]
    r = score_case(case, actual, model_id="test")
    assert r.scores.structure_validity == 0.0


def test_type_safety_correct_types(get_case):
    case = get_case("search-single")
    actual = [ToolCallResult(name="search_web", arguments={"query": "quantum computing", "limit": 5})]
    r = score_case(case, actual, model_id="test")
    assert r.scores.structure_validity == 1.0


# ── Flattened-path precision/recall for nested arguments ────────────

from atlas.evaluator.arguments import _flatten_args, _precision_recall


def test_flatten_args_paths_lists_and_empty_dicts_are_leaves():
    flat = _flatten_args({"a": {"x": 1, "y": {"z": 2}}, "b": [1, {"k": 2}], "c": {}, "d.e": 3})
    assert flat == {("a", "x"): 1, ("a", "y", "z"): 2, ("b",): [1, {"k": 2}],
                    ("c",): {}, ("d.e",): 3}


def test_nested_precision_recall_counted_per_leaf():
    expected = {"a": {"x": 1, "y": 2}}
    actual = {"a": {"x": 1, "y": 9, "z": 3}}
    p, r = _precision_recall(expected, actual)
    # 1 of 3 emitted leaves correct; 1 of 2 expected leaves provided.
    # (Top-level counting used to report 0.5 / 0.5 here.)
    assert p == pytest.approx(1 / 3)
    assert r == pytest.approx(1 / 2)


def test_nested_extra_inner_keys_now_lower_precision_not_recall():
    p, r = _precision_recall({"a": {"x": 1}}, {"a": {"x": 1, "junk": 1, "junk2": 2}})
    assert p == pytest.approx(1 / 3) and r == 1.0


def test_nested_case_insensitive_and_deep_nesting():
    p, r = _precision_recall({"a": {"b": {"c": "Tokyo"}}}, {"a": {"b": {"c": "tokyo"}}})
    assert (p, r) == (1.0, 1.0)


def test_nested_dict_vs_scalar_mismatch_hurts_both():
    p, r = _precision_recall({"a": {"x": 1}}, {"a": 5})
    assert (p, r) == (0.0, 0.0)


def test_dotted_key_does_not_collide_with_nested_path():
    assert _precision_recall({"a.b": 1}, {"a": {"b": 1}}) == (0.0, 0.0)


def test_list_leaf_is_compared_whole():
    assert _precision_recall({"l": ["Bob", "Al"]}, {"l": ["bob", "al"]}) == (1.0, 1.0)
    assert _precision_recall({"l": ["Bob", "Al"]}, {"l": ["al", "bob"]}) == (0.0, 0.0)


def test_flat_args_unchanged_and_edge_cases_kept():
    assert _precision_recall({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == (pytest.approx(1 / 3), 0.5)
    assert _precision_recall({}, {}) == (1.0, 1.0)
    assert _precision_recall({}, {"a": 1}) == (0.0, 1.0)
    assert _precision_recall({"a": 1}, {}) == (1.0, 0.0)


def test_score_case_reports_leaf_level_precision_recall(make_case):
    from atlas.models import ExpectedToolCall, ToolDefinition
    tool = ToolDefinition("t", "", {"type": "object", "properties": {"a": {"type": "object"}}})
    case = make_case([tool], [ExpectedToolCall("t", {"a": {"x": 1, "y": 2}})])
    r = score_case(case, [ToolCallResult("t", {"a": {"x": 1, "y": 9, "z": 3}})], "m")
    assert r.scores.argument_precision == pytest.approx(1 / 3)
    assert r.scores.argument_recall == pytest.approx(1 / 2)
