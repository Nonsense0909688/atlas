"""Human-readable, JSON, and HTML report formatting."""

from __future__ import annotations

import json
import os
import shutil
import sys
from html import escape as html_escape

from atlas.models import EvalResult


def _color(text: str, code: str) -> str:
    """Wrap text in an ANSI escape. Returns plain text when not a TTY."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _ok(text: str) -> str: return _color(f"✓ {text}", "32")


def _fail(text: str) -> str: return _color(f"✗ {text}", "31")


def _dim(text: str) -> str: return _color(text, "2")


def _bold(text: str) -> str: return _color(text, "1")


def _cyan(text: str) -> str: return _color(text, "36")


def _yellow(text: str) -> str: return _color(text, "33")


def _f(value, spec: str = ".2f") -> str:
    """Format a number, or ``n/a`` for ``None`` (axis had no applicable cases)."""
    return "n/a" if value is None else format(value, spec)


def _quality_code(q: float) -> str:
    """ANSI code for a 0..1 quality value: green ≥ .8, yellow ≥ .5, else red."""
    return "32" if q >= 0.8 else ("33" if q >= 0.5 else "31")


def _bar(v: float | None, width: int = 10, lower_is_better: bool = False) -> str:
    """Fixed-width block bar for a 0..1 value, colored by quality."""
    if v is None:
        return _dim("·" * width)
    v = max(0.0, min(1.0, v))
    filled = round(v * width)
    q = 1 - v if lower_is_better else v
    return _color("█" * filled, _quality_code(q)) + _dim("░" * (width - filled))


def _num(v: float | None, lower_is_better: bool = False) -> str:
    """Colored 2-decimal score, or dim ``n/a``."""
    if v is None:
        return _dim("n/a ")
    q = 1 - v if lower_is_better else v
    return _color(f"{v:.2f}", _quality_code(q))


def _fmt_call(name, args) -> str:
    """``get_weather(city="Paris", units="celsius")`` for expected/got lines."""
    if isinstance(args, dict):
        inner = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False, default=str)}"
                          for k, v in args.items())
    else:
        inner = repr(args)
    return f"{name}({inner})"


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: max(1, n - 1)] + "…"


def format_report_text(summary: dict, results: list[EvalResult], model_id: str,
                       elapsed: float | None = None) -> str:
    """Terminal report: headline, score bars, breakdowns, and *why* cases failed.

    Failed cases show the reason plus expected vs. actual calls side by
    side, so you can see the mismatch without re-running with ``-v``. Ends
    with a pytest-style one-liner. Plain (no ANSI) when not a TTY.
    """
    cols = max(60, min(shutil.get_terminal_size((100, 24)).columns, 110))
    rule = _dim("─" * (cols - 4))
    L: list[str] = []

    total = summary.get("count", 0)
    passed = summary.get("passed", 0)   # exact counts from aggregate_scores
    failed = summary.get("failed", 0)
    errors = summary.get("errors", 0)
    overall = summary.get("overall")
    took = f"{elapsed:.1f}s" if elapsed is not None else None

    # Headline
    L.append(f"  {_bold('Atlas report')}  ·  {_cyan(model_id)}")
    L.append(f"  {rule}")
    chips = [_color(f"✓ {passed} passed", "32"),
             _color(f"✗ {failed} failed", "31") if failed else _dim("0 failed")]
    if errors:
        chips.append(_yellow(f"{errors} errors"))
    meta = f"{total} cases" + (f" · {took}" if took else "")
    L.append(f"  {'   '.join(chips)}   {_dim(meta)}")
    L.append("")
    L.append(f"  {_bold('Overall'.ljust(20))} {_bar(overall, 20)} {_num(overall)}")

    # Score axes
    L.append("")
    L.append(f"  {_bold('Scores')}")
    for label, key, lower in [
        ("Tool selection", "tool_selection", False),
        ("Argument accuracy", "argument_accuracy", False),
        ("Argument precision", "argument_precision", False),
        ("Argument recall", "argument_recall", False),
        ("Structure validity", "structure_validity", False),
        ("Multi-tool plan", "multi_tool_planning", False),
        ("Hallucination rate", "hallucination_rate", True),
        ("Abstention acc.", "abstention_accuracy", False),
        ("Consistency", "consistency", False),
    ]:
        v = summary.get(key)
        if key == "consistency" and v is None:
            continue
        note = _dim("  lower is better") if lower else ""
        L.append(f"    {label:<20} {_bar(v)} {_num(v, lower)}{note}")

    # Performance
    lat = summary.get("avg_latency_ms")
    if lat:
        L.append("")
        L.append(f"  {_bold('Performance')}")
        lat_txt = f"{lat / 1000:.1f}s" if lat >= 1000 else f"{lat:.0f}ms"
        L.append(f"    {'Avg latency':<20} {lat_txt}")
        if summary.get("total_tokens"):
            pt, ct = summary.get("total_prompt_tokens"), summary.get("total_completion_tokens")
            L.append(f"    {'Tokens':<20} {summary['total_tokens']:,}  "
                     f"{_dim(f'prompt {pt:,} · completion {ct:,}') if pt is not None and ct is not None else ''}")

    # Breakdowns
    for label, section in [("By category", "by_category"), ("By difficulty", "by_difficulty")]:
        data = summary.get(section)
        if not data:
            continue
        names = sorted(data)
        w = max(len(n) for n in names)
        L.append("")
        L.append(f"  {_bold(label)}")
        for n in names:
            s = data[n]
            pr, ov, cnt = s["pass_rate"], s["overall"], s["count"]
            pct = _color(f"{pr:>4.0%}", _quality_code(pr))
            L.append(f"    {n:<{w}}  {pct}  {_bar(pr)}  {_dim(f'overall {ov:.2f} · n={cnt}')}")

    # Failures: reason + expected vs got
    failed_results = [r for r in results if not r.passed]
    if failed_results:
        multi = len({r.case_id for r in results}) != len(results)
        L.append("")
        L.append(f"  {_bold('Failed cases')} {_dim('(' + str(len(failed_results)) + ')')}")
        pad = " " * 8
        room = cols - len(pad) - 11
        for r in failed_results:
            err = r.notes == "ERROR"
            run = f" · run {r.run_index + 1}" if multi else ""
            score = "ERR" if err else f"{r.scores.overall:.2f}"
            L.append("")
            L.append(f"    {_color('✗', '31')} {_bold(r.case_id)}  "
                     f"{_dim(f'{r.category}/{r.difficulty}{run}')}  {_color(score, '31')}")
            reason = (r.error_analysis or "").strip()
            if reason:
                L.append(f"{pad}{_dim('why'.ljust(9))}{_clip(reason, room)}")
            if err:
                continue
            exp = [_fmt_call(e.name, e.arguments) for e in r.expected_calls] \
                or [_dim("(none — should abstain)")]
            act = [_fmt_call(c.name, c.arguments) for c in r.actual_calls] or [_dim("(no call)")]
            for tag, items in (("expected", exp), ("got", act)):
                for i, item in enumerate(items):
                    label = tag if i == 0 else ""
                    L.append(f"{pad}{_dim(label.ljust(9))}{_clip(item, room)}")

    # pytest-style footer
    L.append("")
    L.append(f"  {rule}")
    parts = []
    if failed:
        parts.append(_color(f"{failed} failed", "1;31"))
    parts.append(_color(f"{passed} passed", "32"))
    if errors:
        parts.append(_yellow(f"{errors} errors"))
    L.append(f"  {', '.join(parts)}" + (f" in {took}" if took else ""))
    return "\n".join(L)


def format_report_json(summary: dict, results: list[EvalResult],
                       model_id: str, provider_name: str) -> str:
    """Machine-readable JSON report (``--format json``).

    Mutates ``summary`` to stamp model/provider (kept for backward compat
    with ``atlas diff``/``atlas history``, which read them from summary).
    """
    summary["model"] = model_id
    summary["provider"] = provider_name
    report = {
        "summary": summary,
        "results": [
            {
                "case_id": r.case_id,
                "category": r.category,
                "difficulty": r.difficulty,
                "passed": r.passed,
                "scores": r.scores.to_dict(),
                "details": r.details,
                "error_analysis": r.error_analysis,
                "actual_calls": [{"name": c.name, "arguments": c.arguments} for c in r.actual_calls],
                "expected_calls": [{"name": e.name, "arguments": e.arguments} for e in r.expected_calls],
                "latency_ms": r.latency_ms,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "run_index": r.run_index,
                "final_answer": r.final_answer,
                "final_state": r.final_state,
                "had_tool_error": r.had_tool_error,
                "recovered": r.recovered,
            }
            for r in results
        ],
        "model": model_id,
        "provider": provider_name,
    }
    return json.dumps(report, indent=2, ensure_ascii=False)


def format_report_html(summary: dict, results: list[EvalResult], model_id: str) -> str:
    """Self-contained HTML report (inline CSS, no external assets)."""
    total = summary.get("count", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    overall = summary.get("overall")

    rows = ""
    for r in results:
        status = "pass" if r.passed else "fail"
        status_icon = "✓" if r.passed else "✗"
        rows += f"""<tr class="{status}">
<td>{html_escape(str(r.case_id))}</td>
<td>{html_escape(str(r.category))}</td>
<td>{html_escape(str(r.difficulty))}</td>
<td>{r.scores.overall:.2f}</td>
<td>{r.scores.tool_selection if r.scores.tool_selection is not None else '—'}</td>
<td>{r.scores.argument_accuracy if r.scores.argument_accuracy is not None else '—'}</td>
<td>{_f(r.scores.structure_validity)}</td>
<td>{status_icon}</td>
</tr>
"""

    cat_rows = ""
    for cat in sorted(summary.get("by_category", {}).keys()):
        s = summary["by_category"][cat]
        cat_rows += f"""<tr>
<td>{html_escape(str(cat))}</td>
<td>{s['count']}</td>
<td>{s['overall']:.2f}</td>
<td>{s['pass_rate']:.0%}</td>
</tr>
"""

    return f"""<!DOCTYPE html>
<html>
<head>
<title>Atlas Report — {html_escape(str(model_id))}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; color: #1a1a1a; }}
h1 {{ font-size: 1.5em; }}
.summary {{ display: flex; gap: 1em; margin: 1em 0; flex-wrap: wrap; }}
.card {{ background: #f6f8fa; border-radius: 8px; padding: 1em; min-width: 120px; text-align: center; }}
.card .value {{ font-size: 2em; font-weight: bold; }}
.card .label {{ color: #666; font-size: .85em; }}
.pass {{ color: #2ea043; }}
.fail {{ color: #cf222e; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: .9em; }}
th {{ background: #f6f8fa; }}
tr.fail td {{ background: #fff5f5; }}
section {{ margin: 2em 0; }}
</style>
</head>
<body>
<h1>Atlas Benchmark Report</h1>
<p>Model: <strong>{html_escape(str(model_id))}</strong> — {total} cases, {passed} passed, {failed} failed</p>

<div class="summary">
<div class="card"><div class="value">{_f(overall)}</div><div class="label">Overall</div></div>
<div class="card"><div class="value">{_f(summary.get('tool_selection'))}</div><div class="label">Tool selection</div></div>
<div class="card"><div class="value">{_f(summary.get('argument_accuracy'))}</div><div class="label">Arg accuracy</div></div>
<div class="card"><div class="value">{_f(summary.get('structure_validity'))}</div><div class="label">Structure</div></div>
<div class="card"><div class="value">{_f(summary.get('hallucination_rate'))}</div><div class="label">Hallucination</div></div>
<div class="card"><div class="value">{_f(summary.get('avg_latency_ms'), '.0f')}ms</div><div class="label">Avg latency</div></div>
</div>

<section>
<h2>Results</h2>
<table>
<tr><th>Case</th><th>Category</th><th>Difficulty</th><th>Overall</th><th>Tool sel</th><th>Args</th><th>Structure</th><th>Status</th></tr>
{rows}
</table>
</section>

<section>
<h2>By Category</h2>
<table>
<tr><th>Category</th><th>Cases</th><th>Overall</th><th>Pass rate</th></tr>
{cat_rows}
</table>
</section>

</body>
</html>"""
