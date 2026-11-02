"""Argument validation and command-specific parser construction."""

from __future__ import annotations

import argparse
import os

from atlas.atlas_cli.constants import DIFFICULTIES, PROVIDER_CHOICES
from atlas.atlas_cli.terminal import _version

_RUN_EXAMPLES = """\
examples:
  atlas run -m qwen3.5:2b                         quick run, text report
  atlas run -m qwen3.5:2b --difficulty hard -v    only hard cases, per-call detail
  atlas run -m qwen3.5:2b --retries 3 -o out.html  consistency run, HTML report
  atlas run -m qwen3.5:2b -o new.json --baseline old.json --fail-on-regression
  atlas run -m qwen3.5:2b --format json | jq .summary.overall

Environment:
  ATLAS_MODEL   default for --model
  NO_COLOR      disable colored output
"""


def _positive_int(value: str) -> int:
    """Parse a positive integer for counts, limits, and sizes."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an integer"
        ) from None

    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _temperature(value: str) -> float:
    """Parse a model temperature in the supported 0..2 range."""
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a number"
        ) from None

    if not 0.0 <= parsed <= 2.0:
        raise argparse.ArgumentTypeError("must be between 0 and 2")
    return parsed


def _unit_float(value: str) -> float:
    """Parse a normalized score threshold in the 0..1 range."""
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a number"
        ) from None

    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _add_provider_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=PROVIDER_CHOICES,
        help="Provider backend",
    )


def _add_run_parser(subcommands) -> None:
    parser = subcommands.add_parser(
        "run",
        help="Run the benchmark against a model",
        epilog=_RUN_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        default=os.environ.get("ATLAS_MODEL"),
        help="Model ID, e.g. qwen3.5:2b (default: $ATLAS_MODEL)",
    )
    _add_provider_argument(parser)

    selection = parser.add_argument_group("case selection")
    selection.add_argument("--suite", help="Custom test suite JSON file (or 'full' for 230 cases)")
    selection.add_argument("--case", help="Run only one case by ID")
    selection.add_argument("--category", help="Run only one category")
    selection.add_argument("--difficulty", choices=DIFFICULTIES, help="Filter by difficulty")
    selection.add_argument("--split", choices=["train", "held_out", "all"], help="Suite split: train, held_out, or all")
    selection.add_argument("--language", help="Filter by language")

    generation = parser.add_argument_group("generation")
    generation.add_argument(
        "--temp",
        type=_temperature,
        default=0.2,
        help="Temperature (default: 0.2)",
    )
    generation.add_argument("--max-tokens", type=_positive_int, default=2048, help="Max completion tokens")
    generation.add_argument(
        "--retries",
        type=_positive_int,
        default=1,
        help="Run each case N times to measure consistency",
    )
    generation.add_argument(
        "--calling-mode",
        choices=["native", "prompt"],
        default="native",
        help="Tool calling mode",
    )
    generation.add_argument(
        "--closed-loop",
        action="store_true",
        help="Force closed-loop multi-turn",
    )
    generation.add_argument("--api-base", help="Provider base URL")
    generation.add_argument("--api-key", help="Provider API key")

    perturbations = parser.add_argument_group("robustness")
    perturbations.add_argument(
        "--scale-tools",
        type=_positive_int,
        metavar="N",
        help="Pad each case to N tools",
    )
    perturbations.add_argument(
        "--shuffle-tools",
        action="store_true",
        help="Shuffle tool order",
    )
    perturbations.add_argument(
        "--truncate-desc",
        type=_positive_int,
        metavar="CHARS",
        help="Truncate descriptions",
    )
    perturbations.add_argument(
        "--paraphrase",
        action="store_true",
        help="Paraphrase prompts",
    )
    perturbations.add_argument(
        "--rename-params",
        metavar="JSON",
        help="Rename params JSON",
    )

    output = parser.add_argument_group("output")
    output.add_argument("-o", "--output", help="Write report to this file")
    output.add_argument(
        "--format",
        choices=["text", "json", "html"],
        help="Infer from output extension, otherwise use text",
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected cases without calling a model",
    )
    verbosity = output.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", action="store_true", help="Show per-call details"
    )
    verbosity.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress output"
    )

    gates = parser.add_argument_group("CI gates")
    gates.add_argument(
        "--fail-under",
        type=_unit_float,
        metavar="SCORE",
        help="Exit 1 if overall score is below SCORE (0-1)",
    )
    gates.add_argument(
        "--baseline",
        metavar="REPORT.json",
        help="Compare this run against a previous JSON report",
    )
    gates.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="With --baseline: fail if a case regresses from pass to fail",
    )


def _add_batch_parser(subcommands) -> None:
    parser = subcommands.add_parser(
        "batch",
        help="Run one suite against many models (leaderboard)",
        epilog="""\
examples:
  atlas batch --models qwen3.5:2b,llama3.1:8b -o batch.json
  atlas batch --models-file models.txt --suite full --format markdown -o LEADERBOARD.md
  atlas batch --models a,b --dry-run

Output formats: text (default), json, html, markdown
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", help="Comma-separated model IDs")
    parser.add_argument("--models-file", help="File with one model ID per line (# comments ok)")
    _add_provider_argument(parser)
    parser.add_argument("--suite", help="Custom test suite JSON file (or 'full' for 230 cases)")
    parser.add_argument("--case", help="Run only one case by ID")
    parser.add_argument("--category", help="Run only one category")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, help="Filter by difficulty")
    parser.add_argument("--split", choices=["train", "held_out", "all"], help="Suite split: train, held_out, or all")
    parser.add_argument("--language", help="Filter by language")
    parser.add_argument("--temp", type=_temperature, default=0.2, help="Temperature (default: 0.2)")
    parser.add_argument("--max-tokens", type=_positive_int, default=2048, help="Max completion tokens")
    parser.add_argument("--retries", type=_positive_int, default=1, help="Run each case N times to measure consistency")
    parser.add_argument("--calling-mode", choices=["native", "prompt"], default="native", help="Tool calling mode")
    parser.add_argument("--closed-loop", action="store_true", help="Force closed-loop multi-turn")
    parser.add_argument("--api-base", help="Provider base URL")
    parser.add_argument("--api-key", help="Provider API key")
    parser.add_argument("-o", "--output", help="Write leaderboard to this file (.json/.html/.md inferred)")
    parser.add_argument("--format", choices=["text", "json", "html", "markdown", "md"])
    parser.add_argument("--dry-run", action="store_true", help="Show models + selection without calling models")
    parser.add_argument("--fail-under", type=_unit_float, metavar="SCORE",
                        help="Exit 1 if the best model's overall is below SCORE")


def _add_list_parser(subcommands) -> None:
    parser = subcommands.add_parser("list", help="Browse test cases")
    parser.add_argument("--suite", help="List a custom suite instead of built-ins")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, help="Filter by difficulty")
    parser.add_argument("-s", "--search", help="Search case IDs and prompts")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")


def _add_show_parser(subcommands) -> None:
    parser = subcommands.add_parser("show", help="Show one case in detail")
    parser.add_argument("case_id", help="A full case ID or unique prefix")
    parser.add_argument("--suite", help="Look in a custom suite instead of built-ins")


def _add_models_parser(subcommands) -> None:
    parser = subcommands.add_parser("models", help="List models available from a provider")
    _add_provider_argument(parser)


def _add_diff_parser(subcommands) -> None:
    parser = subcommands.add_parser("diff", help="Compare two JSON reports")
    parser.add_argument("report1", help="Baseline report (JSON)")
    parser.add_argument("report2", help="New report (JSON)")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 1 if a case regresses from pass to fail",
    )


def _add_history_parser(subcommands) -> None:
    parser = subcommands.add_parser("history", help="List saved reports")
    parser.add_argument("--dir", default=".", help="Reports directory")
    parser.add_argument("--model", help="Filter reports by model ID")
    parser.add_argument(
        "-n",
        "--limit",
        type=_positive_int,
        default=20,
        help="Maximum reports to show",
    )


def _add_import_parser(subcommands) -> None:
    parser = subcommands.add_parser("import", help="Import a dataset or tool specification")
    parser.add_argument(
        "kind",
        choices=["bfcl", "tau-bench", "toolbench", "openapi", "mcp"],
        help="Source format",
    )
    parser.add_argument("source", help="File path or server URL")
    parser.add_argument(
        "-o",
        "--output",
        default="imported_suite.json",
        help="Output suite JSON file",
    )


def _add_generate_parser(subcommands) -> None:
    parser = subcommands.add_parser(
        "generate",
        help="Generate benchmark cases for human review",
    )
    parser.add_argument("--provider", default="ollama", choices=PROVIDER_CHOICES, help="Provider backend")
    parser.add_argument("-m", "--model", required=True, help="Model ID")
    parser.add_argument("-n", "--count", type=_positive_int, default=5, help="Number of cases to generate")
    parser.add_argument("--domain", default="calendar", help="Case domain, e.g. calendar, finance, sql")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default="medium", help="Difficulty of generated cases")
    parser.add_argument("--api-base", help="Provider base URL")
    parser.add_argument("--api-key", help="Provider API key")
    parser.add_argument("-o", "--output", default="review_queue.json", help="Review queue output file")


def _add_doctor_parser(subcommands) -> None:
    parser = subcommands.add_parser("doctor", help="Check Python and provider setup")
    _add_provider_argument(parser)
    parser.add_argument("--api-base", help="Provider base URL")
    parser.add_argument("--api-key", help="Provider API key")


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI parser from small command-specific builders."""
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Atlas — tool-use and agent evaluation framework",
    )
    parser.add_argument("--version", action="version", version=f"atlas {_version()}")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    subcommands = parser.add_subparsers(dest="command", metavar="<command>")

    for add_command in (
        _add_run_parser,
        _add_batch_parser,
        _add_list_parser,
        _add_show_parser,
        _add_models_parser,
        _add_diff_parser,
        _add_history_parser,
        _add_import_parser,
        _add_generate_parser,
        _add_doctor_parser,
    ):
        add_command(subcommands)

    return parser
