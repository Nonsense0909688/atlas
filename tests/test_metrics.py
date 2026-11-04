"""Tests for metrics."""

from atlas.evaluator.scoring import score_case, aggregate_scores
from atlas.models import ToolCallResult


def test_aggregate_with_metrics(get_case):
    case = get_case("weather-single")
    r = score_case(case, [ToolCallResult(name="get_weather", arguments={"city": "Tokyo"})], "m")
    r.latency_ms = 250.0
    r.prompt_tokens = 100
    r.completion_tokens = 50
    r.total_tokens = 150
    agg = aggregate_scores([r])
    assert agg["avg_latency_ms"] == 250.0
    assert agg["total_tokens"] == 150
    assert agg["total_prompt_tokens"] == 100
    assert agg["total_completion_tokens"] == 50
    assert agg["argument_precision"] == 1.0
    assert agg["argument_recall"] == 1.0
    assert agg["hallucination_rate"] == 0.0
