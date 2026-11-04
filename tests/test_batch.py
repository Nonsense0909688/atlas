"""Batch helpers: model parsing, ranking, and report formats."""

import json

from atlas.atlas_runner.batch import (
    build_leaderboard,
    format_batch_html,
    format_batch_json,
    format_batch_markdown,
    format_batch_text,
    parse_models,
    rank_entries,
)


def test_parse_models_comma_separated_dedupes_and_strips():
    assert parse_models(" a:1 , b:2 ,a:1,,") == ["a:1", "b:2"]


def test_parse_models_file_ignores_blank_and_comment_lines(tmp_path):
    f = tmp_path / "models.txt"
    f.write_text("# comment\nmodel-a\n\nmodel-b # inline stays part of name? no, line is kept\n", encoding="utf-8")
    # Lines are taken verbatim apart from surrounding whitespace.
    assert parse_models(models_file=str(f)) == ["model-a", "model-b # inline stays part of name? no, line is kept"]


def test_parse_models_combines_flag_and_file(tmp_path):
    f = tmp_path / "models.txt"
    f.write_text("model-b\nmodel-c\n", encoding="utf-8")
    assert parse_models("model-a", str(f)) == ["model-a", "model-b", "model-c"]


def test_rank_entries_puts_errors_last_and_sorts_by_overall():
    entries = [
        {"model": "low", "overall": 0.5, "pass_rate": 0.5},
        {"model": "err", "overall": None, "pass_rate": None},
        {"model": "high", "overall": 0.9, "pass_rate": 0.9},
    ]
    assert [e["model"] for e in rank_entries(entries)] == ["high", "low", "err"]


def test_build_leaderboard_picks_best_and_computes_pass_rate():
    runs = [
        {"model": "m1", "provider": "ollama", "error": None,
         "summary": {"count": 4, "passed": 3, "failed": 1, "errors": 0, "overall": 0.75,
                     "tool_selection": 0.8, "argument_accuracy": 0.7, "hallucination_rate": 0.0}},
        {"model": "m2", "provider": "ollama", "error": None,
         "summary": {"count": 4, "passed": 4, "failed": 0, "errors": 0, "overall": 0.95,
                     "tool_selection": 1.0, "argument_accuracy": 0.9, "hallucination_rate": 0.0}},
    ]
    report = build_leaderboard(runs)
    assert report["best_model"] == "m2"
    assert report["leaderboard"][0]["pass_rate"] == 1.0
    assert report["leaderboard"][1]["pass_rate"] == 0.75


def test_format_batch_text_contains_ranked_models():
    report = build_leaderboard([
        {"model": "m1", "provider": "ollama", "error": None,
         "summary": {"count": 2, "passed": 1, "failed": 1, "overall": 0.5}},
    ])
    text = format_batch_text(report)
    assert "m1" in text and "Best: m1" in text


def test_format_batch_markdown_is_a_table():
    report = build_leaderboard([
        {"model": "m1", "provider": "ollama", "error": None,
         "summary": {"count": 2, "passed": 1, "failed": 1, "overall": 0.5}},
    ])
    md = format_batch_markdown(report)
    assert md.startswith("| Rank | Model |")
    assert "`m1`" in md


def test_format_batch_json_round_trips():
    report = build_leaderboard([
        {"model": "m1", "provider": "ollama", "error": None,
         "summary": {"count": 2, "passed": 1, "failed": 1, "overall": 0.5}},
    ])
    parsed = json.loads(format_batch_json(report))
    assert parsed["best_model"] == "m1"
    assert parsed["leaderboard"][0]["model"] == "m1"


def test_format_batch_html_escapes_model_names():
    report = build_leaderboard([
        {"model": "<script>", "provider": "ollama", "error": None,
         "summary": {"count": 1, "passed": 1, "failed": 0, "overall": 1.0}},
    ])
    html = format_batch_html(report)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_batch_parser_accepts_models_and_format():
    from atlas.atlas_cli.parser import build_parser

    args = build_parser().parse_args(["batch", "--models", "a,b", "--format", "markdown"])
    assert args.command == "batch"
    assert args.models == "a,b"
    assert args.format == "markdown"


def test_batch_dry_run_does_not_call_a_provider(capsys):
    from atlas.atlas_cli.batch import _batch
    from atlas.atlas_cli.parser import build_parser

    args = build_parser().parse_args(["batch", "--models", "a,b", "--dry-run"])
    assert _batch(args) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "a, b" in out
