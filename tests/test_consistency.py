"""Tests for consistency."""

from atlas.evaluator.scoring import score_case, compute_consistency
from atlas.models import ToolCallResult


def test_consistency_same_results(get_case):
    case = get_case("weather-single")
    runs = [
        score_case(case, [ToolCallResult(name="get_weather", arguments={"city": "Tokyo"})], "m"),
        score_case(case, [ToolCallResult(name="get_weather", arguments={"city": "Tokyo"})], "m"),
        score_case(case, [ToolCallResult(name="get_weather", arguments={"city": "Tokyo"})], "m"),
    ]
    assert compute_consistency(runs) == 1.0


def test_consistency_varied_results(get_case):
    case = get_case("weather-single")
    runs = [
        score_case(case, [ToolCallResult(name="get_weather", arguments={"city": "Tokyo"})], "m"),
        score_case(case, [ToolCallResult(name="get_weather", arguments={"city": "Berlin"})], "m"),
    ]
    assert compute_consistency(runs) == 0.5


# ── Order handling: sequential = ordered, parallel = unordered ──────

import pytest
from atlas.models import ExpectedToolCall, ToolDefinition

_A = ToolDefinition("a", "", {"type": "object", "properties": {"n": {"type": "integer"}}})
_B = ToolDefinition("b", "", {"type": "object", "properties": {"n": {"type": "integer"}}})


def _runs(make_case, category, orders):
    case = make_case([_A, _B], [ExpectedToolCall("a", {"n": 1}), ExpectedToolCall("b", {"n": 2})],
                     category=category)
    calls = {"a": ToolCallResult("a", {"n": 1}), "b": ToolCallResult("b", {"n": 2})}
    return [score_case(case, [calls[n] for n in order], "m") for order in orders]


@pytest.mark.parametrize("category", ["multi_tool", "dependency_chain", "conversation"])
def test_sequential_order_change_is_inconsistent(make_case, category):
    runs = _runs(make_case, category, [("a", "b"), ("b", "a")])
    assert compute_consistency(runs) == 0.5


def test_parallel_order_change_is_consistent(make_case):
    runs = _runs(make_case, "parallel_calls", [("a", "b"), ("b", "a"), ("a", "b")])
    assert compute_consistency(runs) == 1.0


def test_sequential_identical_order_is_consistent(make_case):
    runs = _runs(make_case, "multi_tool", [("a", "b"), ("a", "b"), ("b", "a")])
    assert compute_consistency(runs) == pytest.approx(2 / 3)


def test_parallel_still_detects_different_calls(make_case):
    case = make_case([_A, _B], [ExpectedToolCall("a", {"n": 1})], category="parallel_calls")
    r1 = score_case(case, [ToolCallResult("a", {"n": 1})], "m")
    r2 = score_case(case, [ToolCallResult("a", {"n": 2})], "m")
    assert compute_consistency([r1, r2]) == 0.5
