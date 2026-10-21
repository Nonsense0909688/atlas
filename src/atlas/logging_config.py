"""Central logging setup for Atlas.

Production-level logging rules used here:
- One ``get_logger(__name__)`` per module (standard Python convention).
- Library code logs at DEBUG/INFO; the CLI decides what the user sees.
- ``--verbose`` lowers the level to DEBUG; ``--quiet`` raises it to WARNING.
- Format is ``HH:MM:SS LEVEL module - message`` so log lines are scannable.
- Logs go to stderr, never stdout — so ``atlas run --format json`` output
  on stdout stays clean/pipelineable.

Usage::

    # in any module
    import logging
    logger = logging.getLogger(__name__)

    # called once from cli.main()
    from atlas.logging_config import configure_logging
    configure_logging(verbose=args.verbose, quiet=args.quiet)
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(*, verbose: bool = False, quiet: bool = False) -> None:
    """Configure root logging once. Safe to call multiple times (no-op after first).

    Args:
        verbose: show DEBUG logs (``--verbose``).
        quiet:   show only WARNING+ (``--quiet``). Wins over verbose.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    _CONFIGURED = True
