"""The ``atlas batch`` command: one suite, many models, one leaderboard."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

from atlas.atlas_cli.constants import EXIT_FAIL, EXIT_OK
from atlas.atlas_cli.providers import _preflight
from atlas.atlas_cli.run import _infer_format
from atlas.atlas_cli.terminal import LiveReporter, _err, _note, dim, green, red
from atlas.atlas_runner.batch import (
    build_leaderboard,
    format_batch_html,
    format_batch_json,
    format_batch_markdown,
    format_batch_text,
    parse_models,
)
from atlas.atlas_runner.execution import execute_run
from atlas.exceptions import ConfigError, ProviderError

_FORMATTERS = {
    "text": format_batch_text,
    "json": format_batch_json,
    "html": format_batch_html,
    "markdown": format_batch_markdown,
    "md": format_batch_markdown,
}


def _infer_batch_format(output: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if output:
        ext = Path(output).suffix.lower()
        if ext == ".json":
            return "json"
        if ext in (".html", ".htm"):
            return "html"
        if ext in (".md", ".markdown"):
            return "markdown"
    return "text"


def _batch(args: argparse.Namespace) -> int:
    models = parse_models(getattr(args, "models", None), getattr(args, "models_file", None))
    if not models:
        raise ConfigError("--models a,b,c and/or --models-file models.txt is required")
    if args.dry_run:
        print(f"Atlas batch dry run — {len(models)} model(s): {', '.join(models)}")
        print(f"  provider={args.provider} suite={getattr(args, 'suite', None) or 'builtin'}")
        print("  (no model calls made)")
        return EXIT_OK
    _preflight(args.provider, models[0])

    runs: list[dict] = []
    t0 = time.monotonic()
    for model_id in models:
        print(f"\n{dim('──', sys.stderr)} {model_id} {dim('──', sys.stderr)}", file=sys.stderr)
        model_args = copy.copy(args)
        model_args.model = model_id
        # execute_run expects the full `run` namespace; fill batch-missing defaults.
        for attr, default in {
            "scale_tools": None, "shuffle_tools": False, "truncate_desc": None,
            "paraphrase": False, "rename_params": None, "verbose": False,
            "quiet": True,  # batch prints its own headers; keep per-case noise down
        }.items():
            if not hasattr(model_args, attr):
                setattr(model_args, attr, default)
        reporter = LiveReporter(quiet=True)
        try:
            summary, _results = execute_run(model_args, reporter=reporter)
            runs.append({"model": model_id, "provider": args.provider,
                         "summary": summary, "error": None})
            reporter.close()
        except Exception as e:  # one bad model must not sink the whole batch
            reporter.close()
            _err(f"  {model_id}: {e}")
            runs.append({"model": model_id, "provider": args.provider,
                         "summary": {}, "error": str(e)})
    elapsed = time.monotonic() - t0

    report = build_leaderboard(runs)
    fmt = _infer_batch_format(args.output, args.format)
    output = _FORMATTERS.get(fmt, format_batch_text)(report)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
        print(f"{dim('report →', sys.stderr)} {args.output} {dim(f'({elapsed:.1f}s)')}", file=sys.stderr)
    else:
        print(output)

    if args.fail_under is not None:
        best = report.get("best_overall")
        if best is None or best < args.fail_under:
            shown = "n/a" if best is None else f"{best:.2f}"
            _err(f"✗ gate: best overall {shown} is below --fail-under {args.fail_under:.2f}")
            return EXIT_FAIL
    if any(run.get("error") for run in runs):
        _note("some models errored — see leaderboard above")
    return EXIT_OK
