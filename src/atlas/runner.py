"""Backward-compatible Atlas runner imports.

Implementation is split by responsibility under :mod:`atlas_runner`: case
loading, providers, response parsing, execution, and report formatting.
"""

from atlas.atlas_runner.case_loading import _case_from_json, filter_cases, load_cases
from atlas.atlas_runner.cli import main, parse_args
from atlas.atlas_runner.execution import (
    _error_result, _print_case_detail, _prompt_mode_system, execute_run,
    run_case, run_case_closed_loop, run_case_single,
)
from atlas.atlas_runner.parsing import (
    _tool_calls_from_json, response_to_tool_calls, salvage_tool_calls_from_text,
)
from atlas.atlas_runner.providers import DummyProvider, make_provider
from atlas.atlas_runner.reporting import (
    _bar, _bold, _clip, _color, _cyan, _dim, _f, _fail, _fmt_call, _num,
    _ok, _quality_code, _yellow, format_report_html, format_report_json,
    format_report_text,
)

if __name__ == "__main__":
    main()
