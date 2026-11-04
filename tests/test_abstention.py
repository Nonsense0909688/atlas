"""Tests for abstention."""

from atlas.evaluator.scoring import score_case
from atlas.models import ToolCallResult


def test_abstention_correct(get_case):
    case = get_case("no-tool-greeting")
    r = score_case(case, [], model_id="test")
    assert r.scores.tool_selection == 1.0
    assert r.scores.abstention_accuracy == 1.0
    assert r.scores.hallucination_rate == 0.0


def test_abstention_hallucinate(get_case):
    case = get_case("no-tool-greeting")
    actual = [ToolCallResult(name="get_weather", arguments={"city": "Tokyo"})]
    r = score_case(case, actual, model_id="test")
    assert r.scores.tool_selection == 0.0
    assert r.scores.abstention_accuracy == 0.0


def test_abstention_accuracy_no_tool_case(get_case):
    case = get_case("no-tool-opinion")
    r = score_case(case, [], "m")
    assert r.scores.abstention_accuracy == 1.0


def test_abstention_accuracy_with_tool_case(get_case):
    case = get_case("weather-single")
    r = score_case(case, [ToolCallResult(name="get_weather", arguments={"city": "Tokyo"})], "m")
    assert r.scores.abstention_accuracy is None  # n/a
