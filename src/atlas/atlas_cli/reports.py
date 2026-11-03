"""Report validation, comparison, and history commands."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from atlas.atlas_cli.constants import EXIT_FAIL, EXIT_OK
from atlas.atlas_cli.terminal import (
    _err, _note, _score, _truncate, bold, cyan, dim, green, red, yellow,
)
from atlas.exceptions import ConfigError

# (summary key, number format, whether an increase is desirable)
_DIFF_METRICS = [
    ("overall", ".2f", True),
    ("tool_selection", ".2f", True),
    ("argument_accuracy", ".2f", True),
    ("argument_precision", ".2f", True),
    ("argument_recall", ".2f", True),
    ("structure_validity", ".2f", True),
    ("multi_tool_planning", ".2f", True),
    ("hallucination_rate", ".2f", False),
    ("abstention_accuracy", ".2f", True),
    ("consistency", ".2f", True),
    ("pass_rate", ".2f", True),
    ("avg_latency_ms", ".0f", False),
]


def _load_report(path: str) -> dict:
    """Read an Atlas JSON report or raise ``ConfigError`` with a useful message."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"report not found: {path}") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ConfigError(f"not valid JSON: {path}") from None
    except OSError as e:
        raise ConfigError(f"can't read {path}: {e.strerror or e}") from None
    if not (isinstance(data, dict) and isinstance(data.get("summary"), dict)
            and isinstance(data.get("results"), list)):
        raise ConfigError(f"{path} doesn't look like an Atlas report "
                          f"(create one with `atlas run --format json`)")
    return data


def _per_case(report: dict) -> dict[str, tuple[float, bool]]:
    """case_id → (mean overall across retries, passed in every run)."""
    groups: dict[str, list[dict]] = {}
    for r in report.get("results", []):
        groups.setdefault(r["case_id"], []).append(r)
    return {
        cid: (sum(r["scores"]["overall"] for r in rs) / len(rs),
              all(r.get("passed", False) for r in rs))
        for cid, rs in groups.items()
    }


def _render_diff(r1: dict, r2: dict, label1: str, label2: str, out,
                 name1: str = "", name2: str = "") -> int:
    """Print a metric + per-case diff to ``out``; return the pass→fail regression count.

    Columns are labeled A/B with a legend because the common case is the
    *same* model compared across two runs.
    """
    s1, s2 = r1.get("summary", {}), r2.get("summary", {})

    def who(s: dict, fallback: str, name: str) -> str:
        model = s.get("model") or fallback
        return f"{model}  {dim('(' + name + ')', out)}" if name else model

    print(bold("Atlas diff", out), file=out)
    print(f"  A  {who(s1, label1, name1)}", file=out)
    print(f"  B  {who(s2, label2, name2)}\n", file=out)
    print(dim(f"  {'Metric':<22}{'A':>10}{'B':>10}  {'Δ (B − A)':>11}", out), file=out)

    for key, fmt, higher_better in _DIFF_METRICS:
        v1, v2 = s1.get(key), s2.get(key)
        if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
            continue
        d = v2 - v1
        if abs(d) < (0.5 if fmt == ".0f" else 0.005):
            arrow, paint = "●", dim
        else:
            arrow = "▲" if d > 0 else "▼"
            paint = green if (d > 0) == higher_better else red
        label = key.replace("_", " ").capitalize()
        cell = f"{arrow} {d:+{fmt}}".rjust(11)
        print(f"  {label:<22}{format(v1, fmt):>10}{format(v2, fmt):>10}  {paint(cell, out)}", file=out)

    c1, c2 = _per_case(r1), _per_case(r2)
    changed, added, removed = [], [], []
    unchanged = 0
    for cid in sorted(set(c1) | set(c2)):
        if cid in c1 and cid in c2:
            (o1, p1), (o2, p2) = c1[cid], c2[cid]
            if abs(o2 - o1) > 0.001 or p1 != p2:
                changed.append((o2 - o1, cid, o1, o2, p1, p2))
            else:
                unchanged += 1
        elif cid in c1:
            removed.append(cid)
        else:
            added.append(cid)

    changed.sort()  # biggest drops first
    regressions = sum(1 for *_, p1, p2 in changed if p1 and not p2)
    improved = sum(1 for d, *_ in changed if d > 0)
    worse = len(changed) - improved

    if changed:
        print(f"\n  {bold('Case changes', out)} {dim('(worst first)', out)}", file=out)
        width = min(max(len(cid) for _, cid, *_ in changed), 40)
        for d, cid, o1, o2, p1, p2 in changed:
            arrow = green("▲", out) if d > 0 else red("▼", out)
            status = ""
            if p1 and not p2:
                status = red("PASS → FAIL", out)
            elif p2 and not p1:
                status = green("FAIL → PASS", out)
            print(f"    {arrow} {cid:<{width}}  {o1:.2f} → {o2:.2f}  {status}", file=out)
    for cid in removed:
        print(f"    {yellow('−', out)} {cid}  {dim('only in A', out)}", file=out)
    for cid in added:
        print(f"    {cyan('+', out)} {cid}  {dim('only in B', out)}", file=out)

    summary_bits = [green(f"{improved} improved", out) if improved else dim("0 improved", out),
                    red(f"{worse} regressed", out) if worse else dim("0 regressed", out),
                    dim(f"{unchanged} unchanged", out)]
    if removed:
        summary_bits.append(yellow(f"{len(removed)} removed", out))
    if added:
        summary_bits.append(cyan(f"{len(added)} added", out))
    print("\n  " + dim(" · ", out).join(summary_bits), file=out)
    return regressions


def _diff(args: argparse.Namespace) -> int:
    """``atlas diff``: compare two JSON reports."""
    r1, r2 = _load_report(args.report1), _load_report(args.report2)
    regressions = _render_diff(r1, r2, Path(args.report1).stem, Path(args.report2).stem,
                               sys.stdout, name1=args.report1, name2=args.report2)
    if args.fail_on_regression and regressions:
        _err(f"✗ {regressions} case(s) regressed from pass to fail")
        return EXIT_FAIL
    return EXIT_OK


def _history(args: argparse.Namespace) -> int:
    """``atlas history``: saved reports as a table, newest first."""
    directory = Path(args.dir)
    if not directory.is_dir():
        raise ConfigError(f"not a directory: {args.dir}")

    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    rows, skipped = [], 0
    for f in sorted(directory.glob("*.json"), key=mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            skipped += 1
            continue
        if not (isinstance(data, dict) and isinstance(data.get("summary"), dict)
                and "results" in data):
            skipped += 1  # some other JSON file (e.g. a test suite)
            continue
        s = data["summary"]
        model = str(s.get("model", "?"))
        if args.model and args.model.lower() not in model.lower():
            continue
        rows.append((f.name, model, s.get("overall"), s.get("passed"), s.get("count", 0),
                     datetime.fromtimestamp(mtime(f)).strftime("%Y-%m-%d %H:%M")))

    if not rows:
        _note(f"No Atlas reports in {args.dir}/"
              + (f" matching model '{args.model}'" if args.model else ""))
        _note("Create one with: atlas run -m <model> -o report.json")
        return EXIT_OK

    total = len(rows)
    rows = rows[: args.limit]
    nw = min(max(len(r[0]) for r in rows), 44)
    mw = min(max(len(r[1]) for r in rows), 28)
    print(bold(f"Atlas reports in {args.dir}/") + "\n")
    print(dim(f"  {'File':<{nw}}  {'Model':<{mw}}  {'Overall':>7}  {'Passed':>9}  When", ))
    for name, model, overall, passed, count, when in rows:
        passed_txt = f"{passed}/{count}" if isinstance(passed, int) else f"?/{count}"
        print(f"  {_truncate(name, nw):<{nw}}  {_truncate(model, mw):<{mw}}  "
              f"{_score(overall, 7)}  {passed_txt:>9}  {dim(when)}")
    footer = []
    if total > len(rows):
        footer.append(f"showing {len(rows)} of {total}")
    if skipped:
        footer.append(f"{skipped} non-report JSON file(s) skipped")
    if footer:
        print("\n" + dim("  " + " · ".join(footer)))
    if len(rows) >= 2:
        print(dim(f"\nCompare the latest two: atlas diff {rows[1][0]} {rows[0][0]}"))
    return EXIT_OK
