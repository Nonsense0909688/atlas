"""CLI dispatcher and top-level exception/exit-code policy."""

from __future__ import annotations

import logging
import os
import sys
import traceback

from atlas.atlas_cli.constants import EXIT_CONFIG, EXIT_FAIL, EXIT_INTERRUPT, EXIT_OK
from atlas.atlas_cli.cases import _list_cases, _show_case
from atlas.atlas_cli.imports import _import_cmd
from atlas.atlas_cli.parser import build_parser
from atlas.atlas_cli.providers import _doctor, _list_models, _generate_cmd
from atlas.atlas_cli.reports import _diff, _history
from atlas.atlas_cli.batch import _batch
from atlas.atlas_cli.help import format_command_help, format_top_help, help_target
from atlas.atlas_cli.run import _run
from atlas.atlas_cli.terminal import _err, _note
from atlas.atlas_cli.welcome import _welcome
from atlas.exceptions import ConfigError, ProviderError
from atlas.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``atlas`` script + ``python cli.py``). Returns the exit code."""
    parser = build_parser()
    # Heaven-style help: `atlas`, `atlas --help`, `atlas help <cmd>`,
    # `atlas <cmd> --help` all render grouped help instead of argparse's
    # default flat layout. Unknown `help <x>` falls through to argparse
    # so the error message stays standard.
    target = help_target(list(argv) if argv is not None else sys.argv[1:], parser)
    if target is not None:
        print(format_top_help(parser) if target == "" else format_command_help(parser, target))
        return EXIT_OK
    args = parser.parse_args(argv)
    if os.name == "nt":
        os.system("")  # enables ANSI escapes in classic Windows consoles
    for stream in (sys.stdout, sys.stderr):
        try:  # never crash on ✓/▲ when the console codepage can't encode them
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    if args.no_color:
        os.environ["NO_COLOR"] = "1"
    # Logging first so every handler below honors --verbose. Library INFO
    # lines (e.g. "list_models: found 13 models") are noise for normal use,
    # so default to WARNING+ and only show INFO/DEBUG with -v.
    verbose = getattr(args, "verbose", False)
    configure_logging(verbose=verbose, quiet=not verbose)

    if not args.command:
        _welcome(parser)
        return EXIT_OK

    handlers = {
        "run": _run,
        "batch": _batch,
        "list": _list_cases,
        "show": _show_case,
        "models": _list_models,
        "diff": _diff,
        "history": _history,
        "doctor": _doctor,
        "import": _import_cmd,
        "generate": _generate_cmd,
    }
    try:
        return handlers[args.command](args) or EXIT_OK
    except ConfigError as e:
        _err(f"error: {e}")
        return EXIT_CONFIG
    except ProviderError as e:
        _err(f"error: {e}")
        return EXIT_FAIL
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C)", file=sys.stderr)
        return EXIT_INTERRUPT
    except BrokenPipeError:
        # `atlas list | head` — silence the interpreter's flush-on-exit error.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return EXIT_FAIL
    except Exception as e:  # unexpected: short message by default, traceback on -v
        logger.debug("unhandled error", exc_info=True)
        if getattr(args, "verbose", False):
            traceback.print_exc()
        else:
            _err(f"error: {type(e).__name__}: {e}")
            _note("  re-run with --verbose for a traceback")
        return EXIT_FAIL
