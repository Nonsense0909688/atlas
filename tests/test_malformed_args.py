"""Regression tests: malformed arguments are preserved, never rewritten to ``{}``."""

import pytest

import atlas.runner as runner
from atlas.evaluator.scoring import aggregate_scores, score_case
from atlas.evaluator.validation import validate_structure
from atlas.models import EvalResult, ExpectedToolCall, ScoreBreakdown, ToolCallResult, ToolDefinition
from atlas.providers.base import ChatResponse
from atlas.providers.ollama import OllamaProvider


def normalize(arguments, *, key_present=True):
    func = {"name": "t"}
    if key_present:
        func["arguments"] = arguments
    data = {"message": {"tool_calls": [{"function": func}]}}
    return OllamaProvider._normalize(data, None).tool_calls[0]


# ── provider: normalization ────────────────────────────────────────

def test_dict_arguments_pass_through():
    call = normalize({"city": "Tokyo"})
    assert call["arguments"] == {"city": "Tokyo"} and call["parse_error"] is None


def test_json_string_object_is_decoded():
    call = normalize('{"city": "Tokyo"}')
    assert call["arguments"] == {"city": "Tokyo"} and call["parse_error"] is None


def test_absent_or_null_arguments_mean_no_arguments():
    assert normalize(None)["arguments"] == {}
    assert normalize(None, key_present=False)["arguments"] == {}
    assert normalize(None)["parse_error"] is None


@pytest.mark.parametrize("bad", ["not json {", "", "[1, 2]", "42", [1, 2], 0, False, 3.5, []])
def test_malformed_arguments_are_preserved_with_parse_error(bad):
    call = normalize(bad)
    assert call["arguments"] == bad            # exactly what the model sent
    assert type(call["arguments"]) is type(bad)
    assert call["arguments"] != {}
    assert call["parse_error"]


def test_runner_passes_parse_error_through():
    resp = ChatResponse(tool_calls=[{"name": "t", "arguments": "oops", "raw": {}, "parse_error": "bad"}])
    (call,) = runner.response_to_tool_calls(resp)
    assert call.arguments == "oops" and call.parse_error == "bad"


# ── validation + scoring ───────────────────────────────────────────

NOARGS = ToolDefinition("noargs", "", {"type": "object", "properties": {}})


@pytest.mark.parametrize("bad", ["garbage", [1], 7])
def test_validation_rejects_non_object_arguments_without_crashing(bad):
    ok, msg = validate_structure(ToolCallResult("noargs", bad), NOARGS)
    assert not ok and "arguments should be an object" in msg


def test_malformed_arguments_do_not_score_as_empty_dict(make_case):
    """Expected no-arg call + garbage args used to look like {} == perfect."""
    case = make_case([NOARGS], [ExpectedToolCall("noargs", {})])
    good = score_case(case, [ToolCallResult("noargs", {})], "m")
    bad = score_case(case, [ToolCallResult("noargs", "garbage", parse_error="bad json")], "m")
    assert good.passed and good.scores.argument_accuracy == 1.0
    assert not bad.passed
    assert bad.scores.argument_accuracy == 0.0
    assert bad.scores.argument_precision == 0.0 and bad.scores.argument_recall == 0.0
    assert bad.scores.structure_validity == 0.0
    assert any("MALFORMED" in d and "'garbage'" in d for d in bad.details)
    assert any("parse_error: bad json" in d for d in bad.details)


def test_malformed_extra_call_is_shown_raw_in_details(make_case):
    case = make_case([NOARGS], [ExpectedToolCall("noargs", {})])
    r = score_case(case, [ToolCallResult("noargs", {}), ToolCallResult("noargs", ["x"])], "m")
    assert any("unexpected extra call" in d and "['x']" in d for d in r.details)


def test_malformed_args_survive_into_result_and_json_report(make_case):
    case = make_case([NOARGS], [ExpectedToolCall("noargs", {})])
    r = score_case(case, [ToolCallResult("noargs", "garbage", parse_error="bad")], "m")
    assert r.actual_calls[0].arguments == "garbage"
    import json
    out = json.loads(runner.format_report_json(aggregate_scores([r]), [r], "m", "ollama"))
    assert out["results"][0]["actual_calls"][0]["arguments"] == "garbage"


def test_hallucination_check_tolerates_malformed_arguments(make_case):
    case = make_case([NOARGS], [ExpectedToolCall("noargs", {})])
    r = score_case(case, [ToolCallResult("noargs", "garbage")], "m")
    assert r.scores.hallucination_rate == 0.0  # malformed != invented tool/param


# ── report formatting stays None-safe (n/a axes, all-error runs) ────

def test_text_and_html_reports_handle_na_axes_and_all_error_runs():
    err = EvalResult("e", "x", "m", ScoreBreakdown(overall=0.0), notes="ERROR")
    summary = aggregate_scores([err])
    assert summary["overall"] is None and summary["errors"] == 1
    text = runner.format_report_text(summary, [err], "m")
    html = runner.format_report_html(summary, [err], "m")
    assert "n/a" in text and "1 errors" in text
    assert "n/a" in html
