"""Backward-compatible Atlas CLI entry point.

The implementation lives in :mod:`atlas_cli`, grouped by responsibility.
"""

from atlas.atlas_cli.app import main
from atlas.atlas_cli.cases import _find_case, _list_cases, _show_case
from atlas.atlas_cli.constants import DIFFICULTIES, EXIT_CONFIG, EXIT_FAIL, EXIT_INTERRUPT, EXIT_OK
from atlas.atlas_cli.imports import _import_cmd
from atlas.atlas_cli.parser import _positive_int, _temperature, _unit_float, build_parser
from atlas.atlas_cli.providers import _doctor, _generate_cmd, _list_models, _model_name, _preflight
from atlas.atlas_cli.reports import _diff, _history, _load_report, _per_case, _render_diff
from atlas.atlas_cli.run import _infer_format, _run
from atlas.atlas_cli.terminal import (
    LiveReporter, _color_on, _difficulty, _err, _human_size, _note, _paint,
    _score, _styler, _truncate, _version, bold, cyan, dim, green, red, yellow,
)
from atlas.atlas_cli.help import format_command_help, format_top_help
from atlas.atlas_cli.welcome import _welcome

if __name__ == "__main__":
    raise SystemExit(main())
