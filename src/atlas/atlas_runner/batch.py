"""Batch evaluation: run one suite against many models, rank the results.

Pure helpers live here (model-list parsing, leaderboard ranking, report
formatting) so they can be unit-tested without a CLI or a live provider.
The CLI wrapper lives in ``atlas.atlas_cli.batch``.
"""

from __future__ import annotations

import json
from html import escape as html_escape
from pathlib import Path


def parse_models(models: str | None = None, models_file: str | None = None) -> list[str]:
    """Parse ``--models a,b,c`` and/or ``--models-file models.txt`` into a list.

    File format: one model ID per line, blank lines and ``#`` comments ignored.
    Order is preserved; duplicates are dropped (first occurrence wins).
    """
    found: list[str] = []
    if models:
        found.extend(part.strip() for part in models.split(",") if part.strip())
    if models_file:
        for line in Path(models_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                found.append(line)
    seen: set[str] = set()
    unique: list[str] = []
    for name in found:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def rank_entries(entries: list[dict]) -> list[dict]:
    """Sort leaderboard entries by overall score (desc), then pass rate, then name.

    Entries without a score (provider error) sort last.
    """
    def key(entry: dict):
        overall = entry.get("overall")
        return (
            overall is None,
            -(overall or 0.0),
            -(entry.get("pass_rate") or 0.0),
            entry.get("model", ""),
        )

    return sorted(entries, key=key)


def build_leaderboard(runs: list[dict]) -> dict:
    """Build the batch report dict from per-model ``{model, provider, summary, results}`` runs."""
    entries: list[dict] = []
    for run in runs:
        summary = run.get("summary") or {}
        total = summary.get("count", 0) or 0
        passed = summary.get("passed", 0) or 0
        entries.append({
            "model": run.get("model"),
            "provider": run.get("provider"),
            "overall": summary.get("overall"),
            "passed": passed,
            "failed": summary.get("failed", 0),
            "errors": summary.get("errors", 0),
            "count": total,
            "pass_rate": (passed / total) if total else None,
            "tool_selection": summary.get("tool_selection"),
            "argument_accuracy": summary.get("argument_accuracy"),
            "structure_validity": summary.get("structure_validity"),
            "hallucination_rate": summary.get("hallucination_rate"),
            "avg_latency_ms": summary.get("avg_latency_ms"),
            "error": run.get("error"),
            "summary": summary,
        })
    ranked = rank_entries(entries)
    best = ranked[0] if ranked else None
    return {
        "count": len(runs),
        "models": [entry["model"] for entry in ranked],
        "best_model": best["model"] if best else None,
        "best_overall": best["overall"] if best else None,
        "leaderboard": ranked,
        "runs": runs,
    }


def _fmt(value, spec: str = ".2f") -> str:
    return "n/a" if value is None else format(value, spec)


def format_batch_text(report: dict) -> str:
    """Terminal leaderboard: rank, model, overall bar, pass counts."""
    board = report.get("leaderboard") or []
    lines = [
        f"  Atlas batch — {report.get('count', 0)} model(s)",
        "  " + "─" * 66,
        "",
        f"  {'#':<3} {'model':<28} {'overall':<10} {'passed':<10} {'tool':<7} {'args':<7} {'halluc':<7}",
    ]
    for i, entry in enumerate(board, 1):
        if entry.get("error"):
            lines.append(f"  {i:<3} {str(entry['model']):<28} ERROR: {entry['error']}")
            continue
        overall = entry.get("overall")
        bar = ""
        if overall is not None:
            filled = round(max(0.0, min(1.0, overall)) * 10)
            bar = "█" * filled + "░" * (10 - filled)
        lines.append(
            f"  {i:<3} {str(entry['model']):<28} {_fmt(overall)} {bar:<10} "
            f"{entry.get('passed', 0)}/{entry.get('count', 0)}    "
            f"{_fmt(entry.get('tool_selection')):<7} {_fmt(entry.get('argument_accuracy')):<7} "
            f"{_fmt(entry.get('hallucination_rate')):<7}"
        )
    if report.get("best_model"):
        lines += [
            "",
            f"  Best: {report['best_model']} (overall {_fmt(report.get('best_overall'))})",
        ]
    return "\n".join(lines)


def format_batch_markdown(report: dict) -> str:
    """GitHub-flavoured Markdown table for pasting a leaderboard into a README/PR."""
    board = report.get("leaderboard") or []
    lines = [
        "| Rank | Model | Provider | Overall | Passed | Tool sel. | Arg acc. | Halluc. |",
        "|-----:|-------|----------|--------:|-------:|----------:|---------:|--------:|",
    ]
    for i, entry in enumerate(board, 1):
        if entry.get("error"):
            lines.append(f"| {i} | `{entry['model']}` | {entry.get('provider') or ''} | ERROR | — | — | — | — |")
            continue
        lines.append(
            f"| {i} | `{entry['model']}` | {entry.get('provider') or ''} | "
            f"{_fmt(entry.get('overall'))} | {entry.get('passed', 0)}/{entry.get('count', 0)} | "
            f"{_fmt(entry.get('tool_selection'))} | {_fmt(entry.get('argument_accuracy'))} | "
            f"{_fmt(entry.get('hallucination_rate'))} |"
        )
    return "\n".join(lines)


def format_batch_json(report: dict) -> str:
    """Machine-readable batch report (leaderboard + per-model summaries, no full case dumps)."""
    slim = {
        "count": report.get("count"),
        "best_model": report.get("best_model"),
        "best_overall": report.get("best_overall"),
        "leaderboard": [
            {k: v for k, v in entry.items() if k != "summary"}
            for entry in report.get("leaderboard") or []
        ],
        "runs": [
            {"model": run.get("model"), "provider": run.get("provider"),
             "summary": run.get("summary"), "error": run.get("error")}
            for run in report.get("runs") or []
        ],
    }
    return json.dumps(slim, indent=2, ensure_ascii=False)


def format_batch_html(report: dict) -> str:
    """Self-contained HTML leaderboard (inline CSS, no external assets)."""
    rows = ""
    for i, entry in enumerate(report.get("leaderboard") or [], 1):
        if entry.get("error"):
            rows += (
                f"<tr class=\"fail\"><td>{i}</td><td>{html_escape(str(entry['model']))}</td>"
                f"<td colspan=\"5\">ERROR: {html_escape(str(entry['error']))}</td></tr>\n"
            )
            continue
        rows += (
            f"<tr><td>{i}</td><td>{html_escape(str(entry['model']))}</td>"
            f"<td>{_fmt(entry.get('overall'))}</td>"
            f"<td>{entry.get('passed', 0)}/{entry.get('count', 0)}</td>"
            f"<td>{_fmt(entry.get('tool_selection'))}</td>"
            f"<td>{_fmt(entry.get('argument_accuracy'))}</td>"
            f"<td>{_fmt(entry.get('hallucination_rate'))}</td></tr>\n"
        )
    return f"""<!DOCTYPE html>
<html>
<head>
<title>Atlas Batch Report</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; color: #1a1a1a; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: .9em; }}
th {{ background: #f6f8fa; }}
tr.fail td {{ background: #fff5f5; color: #cf222e; }}
</style>
</head>
<body>
<h1>Atlas Batch Report</h1>
<p>{report.get('count', 0)} model(s) — best: <strong>{html_escape(str(report.get('best_model')))}</strong> (overall {_fmt(report.get('best_overall'))})</p>
<table>
<tr><th>Rank</th><th>Model</th><th>Overall</th><th>Passed</th><th>Tool sel.</th><th>Arg acc.</th><th>Halluc.</th></tr>
{rows}
</table>
</body>
</html>"""
