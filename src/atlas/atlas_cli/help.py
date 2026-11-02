"""Heaven-style help: grouped commands, inline examples, one screen.

``atlas`` (no args), ``atlas --help`` and ``atlas help <command>`` all
render through here instead of argparse's default flat layout, so the
CLI reads like heaven: section -> commands -> example invocations.
Option lists for a command are derived from the live argparse parser,
so help never drifts from the real flags.
"""

from __future__ import annotations

import argparse

from atlas.atlas_cli.terminal import _version, bold, cyan, dim

# name -> (tagline, [example invocations])
COMMAND_INFO: dict[str, tuple[str, list[str]]] = {
    "run": (
        "Benchmark one model against the suite (32 builtin / 230 full)",
        [
            "atlas run -m qwen3.5:2b",
            "atlas run -m qwen3.5:2b --difficulty hard -v",
            "atlas run -m qwen3.5:2b --retries 3 -o report.html",
            "atlas run -m qwen3.5:2b -o new.json --baseline old.json --fail-on-regression",
            "atlas run -m qwen3.5:2b --suite full --split held_out",
        ],
    ),
    "batch": (
        "Rank many models on one suite (leaderboard)",
        [
            "atlas batch --models qwen3.5:2b,llama3.1:8b -o batch.json",
            "atlas batch --models-file models.txt --suite full --format markdown -o LEADERBOARD.md",
            "atlas batch --models a,b --dry-run",
        ],
    ),
    "list": (
        "Browse test cases",
        [
            "atlas list",
            "atlas list --difficulty hard",
            "atlas list -s weather --json",
        ],
    ),
    "show": (
        "Inspect one case in detail",
        ["atlas show weather-single"],
    ),
    "models": (
        "List models installed on the provider",
        ["atlas models", "atlas models --provider openai"],
    ),
    "diff": (
        "Compare two JSON reports (regressions)",
        [
            "atlas diff old.json new.json",
            "atlas diff old.json new.json --fail-on-regression",
        ],
    ),
    "history": (
        "List saved reports",
        ["atlas history", "atlas history --model qwen3.5:2b -n 5"],
    ),
    "import": (
        "Import BFCL / tau-bench / ToolBench / OpenAPI / MCP data as a suite",
        [
            "atlas import bfcl data.jsonl -o suite.json",
            "atlas import openapi spec.yaml -o suite.json",
            "atlas import mcp http://localhost:3000/mcp -o suite.json",
        ],
    ),
    "generate": (
        "Draft new benchmark cases for human review",
        ["atlas generate -m qwen3.5:2b -n 10 --domain calendar -o review_queue.json"],
    ),
    "doctor": (
        "Check Python, cases, and provider setup",
        ["atlas doctor", "atlas doctor --provider openai"],
    ),
}

# Top-level groups: (heading, [command names], [example lines under the group])
_GROUPS: list[tuple[str, list[str], list[str]]] = [
    (
        "EVALUATE",
        ["run", "batch"],
        [
            "atlas run -m qwen3.5:2b --difficulty easy",
            "atlas batch --models qwen3.5:2b,llama3.1:8b -o batch.json",
        ],
    ),
    (
        "BROWSE & COMPARE",
        ["list", "show", "models", "diff", "history"],
        [
            "atlas list --difficulty hard",
            "atlas show weather-single",
            "atlas diff old.json new.json",
        ],
    ),
    (
        "EXTEND",
        ["import", "generate"],
        [
            "atlas import openapi spec.yaml -o suite.json",
            "atlas generate -m qwen3.5:2b -n 10 -o review_queue.json",
        ],
    ),
    (
        "SETUP",
        ["doctor"],
        ["atlas doctor"],
    ),
]


def _subparsers(parser: argparse.ArgumentParser) -> dict:
    """Return the {name: subparser} map from a built parser."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _option_line(action: argparse.Action) -> tuple[str, str]:
    """(left, help) for one argparse action, heaven-style: ``--flag META  help``.

    Choices go in the help text (not the left column) so the column stays
    narrow: ``--provider PROVIDER  Provider backend (ollama|openai|...)``.
    """
    if action.option_strings:
        left = ", ".join(action.option_strings)
        metavar = action.metavar
        if action.nargs == 0:
            pass  # store_true / store_false / help / version: no value shown
        elif metavar:
            left += f" {metavar}"
        else:
            left += f" {action.dest.upper()}"
    else:
        left = action.dest.upper()
    help_text = action.help or ""
    if action.choices:
        joined = "|".join(str(c) for c in action.choices)
        if joined not in help_text:
            help_text = f"{help_text} ({joined})" if help_text else joined
    default = action.default
    if (
        help_text
        and default not in (None, False, argparse.SUPPRESS)
        and "default" not in help_text.lower()
        and action.option_strings
        and not isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
    ):
        help_text = f"{help_text} (default: {default})"
    return left, help_text


def _render_options(rendered: list[tuple[str, str]]) -> list[str]:
    """Render option rows; long help wraps onto an indented second line."""
    lines: list[str] = []
    for left, help_text in rendered:
        if help_text and (len(left) >= 32 or len(left) + len(help_text) > 62):
            lines.append(f"  {left}")
            lines.append(f"      {help_text}")
        else:
            lines.append(f"  {left:<34}{help_text}".rstrip())
    return lines


def format_top_help(parser: argparse.ArgumentParser) -> str:
    """Full bare-``atlas`` / ``atlas --help`` screen."""
    lines = [
        f"{bold('Atlas')} {dim(_version())} — tool-use and agent evaluation framework",
        "",
    ]
    for heading, names, examples in _GROUPS:
        lines.append(bold(heading))
        for name in names:
            tagline, _ = COMMAND_INFO[name]
            lines.append(f"  {cyan(f'{name:<10}')}{tagline}")
        lines.append("")
        for example in examples:
            lines.append(f"  {dim('$')} {example}")
        lines.append("")
    lines += [
        bold("GLOBAL OPTIONS"),
        "  --no-color    Disable colored output",
        "  --version     Show version",
        "  -h, --help    Show this help",
        "",
        f'Run {cyan("atlas help <command>")} for detailed help on one command, e.g. {cyan("atlas help batch")}',
        dim("Docs: docs/ (getting-started, cli-reference, batch, providers, scoring, ci)"),
    ]
    return "\n".join(lines)


def format_command_help(parser: argparse.ArgumentParser, name: str) -> str:
    """Detailed heaven-style help for one command, options taken from argparse."""
    subparsers = _subparsers(parser)
    if name not in subparsers:
        known = ", ".join(subparsers) or "(none)"
        raise KeyError(f"unknown command: {name} (known: {known})")
    sub = subparsers[name]
    tagline, examples = COMMAND_INFO.get(name, (sub.description or "", []))

    # Positionals (show/diff/import) go in the usage line, heaven-style:
    # ``atlas show <case_id> [options]`` — not argparse's full flag dump.
    positionals_all = [
        a for group in sub._action_groups for a in group._group_actions
        if not a.option_strings and a.dest != "help"
    ]
    usage_parts = ["atlas", name]
    for action in positionals_all:
        if action.choices:
            usage_parts.append("{" + "|".join(str(c) for c in action.choices) + "}")
        else:
            usage_parts.append(f"<{action.dest}>")
    if any(a.option_strings for group in sub._action_groups for a in group._group_actions):
        usage_parts.append("[options]")
    lines = [f"{bold('atlas')} {bold(name)} — {tagline}", "", bold("USAGE"), "  " + " ".join(usage_parts)]
    for example in examples[:2]:
        lines.append(f"  {example}")

    # Grouped options straight from argparse's own groups (single source of truth).
    for group in sub._action_groups:
        rendered: list[tuple[str, str]] = []
        for action in group._group_actions:
            if action.dest in ("help",):
                continue
            rendered.append(_option_line(action))
        if not rendered:
            continue
        title = (group.title or "options").upper()
        # argparse calls the first group "options" and puts positionals there too.
        positionals = [r for r in rendered if not r[0].startswith("-")]
        options = [r for r in rendered if r[0].startswith("-")]
        if positionals:
            lines += ["", bold("ARGUMENTS")]
            lines += _render_options(positionals)
        if options:
            lines += ["", bold(title)]
            lines += _render_options(options)

    if examples:
        lines += ["", bold("EXAMPLES")]
        for example in examples:
            lines.append(f"  {dim('$')} {example}")

    if name in ("run", "batch"):
        lines += [
            "",
            bold("EXIT CODES"),
            "  0  Success (and CI gates passed)",
            "  1  Benchmark failed, or a --fail-under / --fail-on-regression gate failed",
            "  2  Configuration error (bad flags, missing model, bad suite)",
            "  130  Interrupted (Ctrl+C)",
        ]
    return "\n".join(lines)


def help_target(argv: list[str], parser: argparse.ArgumentParser) -> str | None:
    """Return the command name to show help for, ``""`` for top help, else None.

    Handles: (nothing) handled by caller, ``--help``/``-h``/``help``,
    ``help <cmd>``, ``<cmd> --help``/``-h``, ``--help <cmd>``.
    Returns None when the argv is a normal invocation (parse it for real).
    """
    if not argv:
        return None
    known = set(_subparsers(parser))
    first = argv[0]
    if first in ("-h", "--help"):
        if len(argv) > 1 and argv[1] in known:
            return argv[1]
        return ""
    if first == "help":
        if len(argv) == 1:
            return ""
        return argv[1] if argv[1] in known else None
    if first in known and len(argv) > 1 and argv[1] in ("-h", "--help"):
        return first
    return None
