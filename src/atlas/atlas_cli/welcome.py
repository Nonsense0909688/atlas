"""First-run help shown when Atlas is invoked without a subcommand."""

from __future__ import annotations

import argparse

from atlas.atlas_cli.help import format_top_help


def _welcome(parser: argparse.ArgumentParser) -> None:
    """Shown for bare ``atlas``: heaven-style grouped overview (see help.py)."""
    print(format_top_help(parser))
