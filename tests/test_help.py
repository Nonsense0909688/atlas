"""Heaven-style help: grouped top screen + per-command detail."""

from atlas.atlas_cli.app import main
from atlas.atlas_cli.help import (
    COMMAND_INFO,
    format_command_help,
    format_top_help,
    help_target,
)
from atlas.atlas_cli.parser import build_parser


def test_top_help_lists_every_command_including_batch_import_generate():
    text = format_top_help(build_parser())
    for name in COMMAND_INFO:
        assert name in text, name
    assert "batch" in text and "import" in text and "generate" in text
    # Heaven-style: grouped sections with inline example invocations.
    assert "EVALUATE" in text and "BROWSE & COMPARE" in text
    assert "atlas batch --models" in text
    assert "atlas help <command>" in text


def test_command_help_is_concentrated_and_informative():
    parser = build_parser()
    run_help = format_command_help(parser, "run")
    assert run_help.startswith("atlas run")
    assert "USAGE" in run_help and "atlas run [options]" in run_help
    assert "CASE SELECTION" in run_help and "CI GATES" in run_help
    assert "EXAMPLES" in run_help and "EXIT CODES" in run_help
    # Every flag explains itself (no blank descriptions left behind).
    assert "--difficulty DIFFICULTY" in run_help
    assert "Filter by difficulty" in run_help
    assert "--provider PROVIDER" in run_help and "Provider backend" in run_help

    batch_help = format_command_help(parser, "batch")
    assert "--models MODELS" in batch_help
    assert "atlas batch --models" in batch_help

    show_help = format_command_help(parser, "show")
    assert "atlas show <case_id> [options]" in show_help
    assert "ARGUMENTS" in show_help

    import_help = format_command_help(parser, "import")
    assert "atlas import {bfcl|tau-bench|toolbench|openapi|mcp} <source> [options]" in import_help
    assert "KIND" in import_help and "SOURCE" in import_help


def test_help_target_recognizes_all_heaven_style_forms():
    parser = build_parser()
    assert help_target(["--help"], parser) == ""
    assert help_target(["-h"], parser) == ""
    assert help_target(["help"], parser) == ""
    assert help_target(["help", "batch"], parser) == "batch"
    assert help_target(["batch", "--help"], parser) == "batch"
    assert help_target(["batch", "-h"], parser) == "batch"
    assert help_target(["--help", "run"], parser) == "run"
    assert help_target(["run", "-m", "x"], parser) is None
    assert help_target(["help", "nosuch"], parser) is None
    assert help_target([], parser) is None


def test_main_help_paths_return_ok_and_print_help(capsys):
    assert main([]) == 0
    assert "batch" in capsys.readouterr().out

    assert main(["help", "batch"]) == 0
    out = capsys.readouterr().out
    assert "atlas batch" in out and "--models" in out

    assert main(["run", "--help"]) == 0
    assert "atlas run" in capsys.readouterr().out

    assert main(["--help"]) == 0
    assert "EVALUATE" in capsys.readouterr().out
