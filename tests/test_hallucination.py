"""Tests for hallucination."""

from atlas.evaluator.scoring import score_case
from atlas.models import ToolCallResult


def test_hallucination_unknown_tool(get_case):
    case = get_case("weather-single")
    actual = [ToolCallResult(name="get_forecast_api", arguments={"city": "Tokyo"})]
    r = score_case(case, actual, model_id="test")
    assert r.scores.hallucination_rate == 1.0
    assert "hallucinated tool" in r.details[0].lower() or any("hallucination" in d.lower() for d in r.details)


def test_hallucination_extra_param(get_case):
    case = get_case("weather-single")
    actual = [ToolCallResult(name="get_weather", arguments={"city": "Tokyo", "forecast_days": 5})]
    r = score_case(case, actual, model_id="test")
    assert r.scores.hallucination_rate == 1.0
    assert r.scores.structure_validity == 0.0  # extra param fails validation


def test_no_hallucination(get_case):
    case = get_case("weather-single")
    actual = [ToolCallResult(name="get_weather", arguments={"city": "Tokyo"})]
    r = score_case(case, actual, model_id="test")
    assert r.scores.hallucination_rate == 0.0


def test_passed_requires_low_hallucination(get_case):
    """Even with high overall, hallucination ≥ 0.5 should fail."""
    case = get_case("weather-single")
    actual = [
        ToolCallResult(name="get_weather", arguments={"city": "Tokyo"}),
        ToolCallResult(name="nonexistent_tool", arguments={}),
    ]
    r = score_case(case, actual, model_id="test")
    assert r.scores.hallucination_rate == 0.5
    assert r.passed is False
