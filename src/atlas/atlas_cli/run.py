"""The ``atlas run`` command: preflight, execution, output, and CI gates."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

from atlas.atlas_cli.constants import EXIT_FAIL, EXIT_OK
from atlas.atlas_cli.providers import _preflight
from atlas.atlas_cli.reports import _load_report, _render_diff
from atlas.atlas_cli.terminal import (
    LiveReporter, _err, _note, _score, bold, cyan, dim, green, red, yellow,
)
from atlas.exceptions import ConfigError
from atlas.atlas_runner.execution import execute_run
from atlas.atlas_runner.reporting import (
    format_report_html,
    format_report_json,
    format_report_text,
)


def _infer_format(output: str | None) -> str:
    if output:
        ext = Path(output).suffix.lower()
        if ext == ".json":
            return "json"
        if ext in (".html", ".htm"):
            return "html"
    return "text"


def _run(args: argparse.Namespace) -> int:
    if not args.model:
        raise ConfigError("--model is required (or set ATLAS_MODEL)")
    if args.fail_on_regression and not args.baseline:
        raise ConfigError("--fail-on-regression needs --baseline REPORT.json")

    fmt = args.format or _infer_format(args.output)
    baseline = _load_report(args.baseline) if args.baseline else None  # fail before the long part
    if not args.dry_run:
        _preflight(args.provider, args.model)

    # Keep stdout clean for machine-readable reports sent to a pipe: the
    # runner prints live progress with print(), so shunt it to stderr then.
    report_to_stdout = not args.output
    shunt = report_to_stdout and fmt != "text"
    reporter = None if args.dry_run else LiveReporter(verbose=args.verbose, quiet=args.quiet)
    root = logging.getLogger()
    prev_level = root.level
    if not args.verbose:
        # Provider errors are shown inline by the reporter; stray log lines
        # would tear the live display.
        root.setLevel(logging.ERROR)
    t0 = time.monotonic()
    try:
        with redirect_stdout(sys.stderr) if shunt else nullcontext():
            summary, results = execute_run(args, reporter=reporter)
    finally:
        if reporter:
            reporter.close()
        root.setLevel(prev_level)
    elapsed = time.monotonic() - t0
    if args.dry_run:
        return EXIT_OK

    # format_report_json stamps model/provider onto summary, so call it first.
    report_json = format_report_json(summary, results, args.model, args.provider)
    if fmt == "json":
        output = report_json
    elif fmt == "html":
        output = format_report_html(summary, results, args.model)
    else:
        output = format_report_text(summary, results, args.model, elapsed=elapsed)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    else:
        print(output)

    # Status footer (stderr). The text report already ends with its own
    # pass/fail line, so only repeat it for file/json/html output.
    already_shown = fmt == "text" and not args.output
    count = summary.get("count", 0)
    passed, failed = summary.get("passed", 0), summary.get("failed", 0)
    errors = summary.get("errors", 0)
    overall = summary.get("overall")
    parts = [f"{count} cases in {elapsed:.1f}s",
             green(f"{passed} passed", sys.stderr),
             (red if failed else dim)(f"{failed} failed", sys.stderr)]
    if errors:
        parts.append(yellow(f"{errors} errors", sys.stderr))
    parts.append("overall " + ("n/a" if overall is None else _score(overall, stream=sys.stderr)))
    if not already_shown:
        print("\n" + dim(" · ", sys.stderr).join(parts), file=sys.stderr)
    if args.output:
        print(f"{dim('report →', sys.stderr)} {args.output}", file=sys.stderr)
    if errors:
        _note(f"{errors} case(s) hit provider errors — check `atlas doctor` and the provider logs.")

    # Gates.
    code = EXIT_OK
    if args.fail_under is not None and (overall is None or overall < args.fail_under):
        shown = "n/a" if overall is None else f"{overall:.2f}"
        _err(f"✗ gate: overall {shown} is below --fail-under {args.fail_under:.2f}")
        code = EXIT_FAIL
    if baseline is not None:
        new_report = json.loads(report_json)
        print(file=sys.stderr)
        regressions = _render_diff(baseline, new_report, "baseline", "this run", sys.stderr,
                                   name1=args.baseline, name2=args.output or "this run")
        if args.fail_on_regression and regressions:
            _err(f"✗ gate: {regressions} case(s) regressed from pass to fail")
            code = EXIT_FAIL
    return code
