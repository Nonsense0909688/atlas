"""Standalone ``atlas run`` command used by the compatibility entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from atlas.atlas_runner.execution import execute_run
from atlas.atlas_runner.reporting import (
    format_report_html, format_report_json, format_report_text,
)
from atlas.exceptions import ConfigError

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``atlas run`` flags. Kept separate from ``main`` for testability."""
    p = argparse.ArgumentParser(prog="atlas run", description="Atlas — tool-use and agent evaluation framework")
    p.add_argument("--model", required=True, help="Model ID (e.g. qwen3.5:2b)")
    p.add_argument("--provider", default="ollama", choices=["ollama"], help="Provider name")
    p.add_argument("--suite", help="Custom test suite JSON file")
    p.add_argument("--case", help="Run only one case by ID")
    p.add_argument("--category", help="Run only one category")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard", "expert"], help="Run only this difficulty")
    p.add_argument("--output", help="Write report to file")
    p.add_argument("--format", default="text", choices=["text", "json", "html"], help="Report format")
    p.add_argument("--temp", type=float, default=0.2, help="Temperature")
    p.add_argument("--max-tokens", type=int, default=2048, help="Max tokens")
    p.add_argument("--retries", type=int, default=1, help="Run each case N times (consistency)")
    p.add_argument("--verbose", "-v", action="store_true", help="Show per-call details")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    p.add_argument("--dry-run", action="store_true", help="Don't call the model, show cases only")
    return p.parse_args(argv)


def main():
    """Entry point for ``atlas run``. Handles config errors with exit code 2."""
    args = parse_args()
    try:
        summary, results = execute_run(args)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C). Partial results discarded.", file=sys.stderr)
        sys.exit(130)
    if args.dry_run:
        return

    if args.format == "json":
        output = format_report_json(summary, results, args.model, args.provider)
    elif args.format == "html":
        output = format_report_html(summary, results, args.model)
    else:
        output = format_report_text(summary, results, args.model)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"\nReport written to {args.output}")
        logger.info("report written to %s", args.output)
    else:
        print()
        print(output)
