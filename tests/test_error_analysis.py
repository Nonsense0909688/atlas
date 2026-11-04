"""Tests for error analysis."""

from atlas.evaluator.scoring import score_case
from atlas.models import ToolCallResult


def test_error_analysis_wrong_tool(get_case):
    case = get_case("weather-single")
    r = score_case(case, [ToolCallResult(name="calculator", arguments={"expression": "1"})], "m")
    assert "Wrong tool" in r.error_analysis
    assert "Hallucinated" in r.error_analysis


def test_error_analysis_clean(get_case):
    case = get_case("weather-single")
    r = score_case(case, [ToolCallResult(name="get_weather", arguments={"city": "Tokyo"})], "m")
    assert r.error_analysis == "No major errors"
