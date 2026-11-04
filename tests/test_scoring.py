"""Tests for scoring."""

from atlas.evaluator.scoring import score_case, aggregate_scores
from atlas.models import ToolCallResult


def test_perfect_weather(get_case):
    case = get_case("weather-single")
    actual = [ToolCallResult(name="get_weather", arguments={"city": "Tokyo"})]
    r = score_case(case, actual, model_id="test")
    assert r.scores.tool_selection == 1.0
    assert r.scores.argument_accuracy == 1.0
    assert r.scores.structure_validity == 1.0
    assert r.scores.overall == 1.0
    assert r.passed is True


def test_wrong_tool(get_case):
    case = get_case("weather-single")
    actual = [ToolCallResult(name="calculator", arguments={"expression": "2+2"})]
    r = score_case(case, actual, model_id="test")
    assert r.scores.tool_selection == 0.0
    assert r.passed is False
    assert r.scores.hallucination_rate == 1.0  # calculator not in case tools


def test_multi_parallel(get_case):
    case = get_case("multi-parallel")
    actual = [
        ToolCallResult(name="get_weather", arguments={"city": "London"}),
        ToolCallResult(name="get_weather", arguments={"city": "New York"}),
    ]
    r = score_case(case, actual, model_id="test")
    assert r.scores.tool_selection == 1.0
    assert r.scores.multi_tool_planning == 1.0


def test_multi_missing_call(get_case):
    case = get_case("multi-parallel")
    actual = [ToolCallResult(name="get_weather", arguments={"city": "London"})]
    r = score_case(case, actual, model_id="test")
    assert r.scores.tool_selection < 1.0
    assert r.scores.multi_tool_planning is not None


def test_aggregate(get_case):
    case = get_case("weather-single")
    r1 = score_case(case, [ToolCallResult(name="get_weather", arguments={"city": "Tokyo"})], "m1")
    r2 = score_case(case, [ToolCallResult(name="calculator", arguments={"expression": "1"})], "m1")
    agg = aggregate_scores([r1, r2])
    assert agg["count"] == 2
    assert agg["by_category"]["single_tool"]["count"] == 2
    assert agg["by_difficulty"]["easy"]["count"] == 2


def test_alternative_sequence_selection_does_not_mutate_case(get_case):
    case = get_case("weather-single")
    original_expected = list(case.expected_calls)
    case.alternative_sequences = [
        [{"name": "calculator", "arguments": {"expression": "1+1"}}],
        [{"name": "get_weather", "arguments": {"city": "London"}}],
    ]

    result = score_case(
        case,
        [ToolCallResult(name="get_weather", arguments={"city": "London"})],
    )

    assert case.expected_calls == original_expected
    assert [(call.name, call.arguments) for call in result.expected_calls] == [
        ("get_weather", {"city": "London"})
    ]
    assert any("alternative sequence selected" in item for item in result.details)


def test_safety_violation_makes_case_fail(get_case):
    case = get_case("weather-single")
    case.allowed_tools = ["calculator"]

    result = score_case(
        case,
        [ToolCallResult(name="get_weather", arguments={"city": "Tokyo"})],
    )

    assert result.scores.tool_selection == 1.0
    assert result.scores.safety_score < 1.0
    assert result.passed is False
    assert "Safety violations" in result.error_analysis


def test_extended_alternative_sequence_fixture_selects_valid_order():
    from atlas.cases.domains import get_extended_cases

    case = next(item for item in get_extended_cases() if item.id == "alt-sequence-1")
    actual = [
        ToolCallResult(name="b", arguments={}),
        ToolCallResult(name="a", arguments={}),
    ]

    result = score_case(case, actual)

    assert result.scores.multi_tool_planning == 1.0
    assert [(call.name, call.arguments) for call in result.expected_calls] == [
        ("b", {}),
        ("a", {}),
    ]
