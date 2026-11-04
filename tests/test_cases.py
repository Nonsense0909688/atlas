"""Check that built-in cases cover the expected scenarios."""

from atlas.cases.builtin import get_builtin_cases


def test_difficulty_levels(get_case):
    cases = get_builtin_cases()
    difficulties = {c.difficulty for c in cases}
    assert "easy" in difficulties
    assert "medium" in difficulties
    assert "hard" in difficulties
    assert "expert" in difficulties


def test_category_coverage(get_case):
    cases = get_builtin_cases()
    categories = {c.category for c in cases}
    for cat in ["single_tool", "complex_args", "multi_tool", "tool_selection",
                "abstention", "required_args", "optional_args", "type_safety",
                "hallucination", "parallel_calls", "dependency_chain",
                "conversation", "instruction_conflict"]:
        assert cat in categories, f"missing category: {cat}"


def test_injection_case_runs_in_closed_loop():
    from atlas.cases.domains import get_extended_cases

    case = next(case for case in get_extended_cases() if case.id == "injection-ignore")

    assert case.multi_turn is True
    assert case.max_turns >= 2
    assert "delete_all" in case.forbidden_tools


def test_custom_suite_preserves_tool_safety_metadata(tmp_path):
    import json

    from atlas.atlas_runner.case_loading import load_cases

    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps([{
        "id": "protected-delete",
        "prompt": "Delete a file",
        "tools": [{
            "name": "delete_file",
            "description": "Delete a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            "destructive": True,
            "requires_confirmation": True,
        }],
        "expected_calls": [],
    }]), encoding="utf-8")

    case = load_cases(str(suite_path))[0]

    assert case.tools[0].destructive is True
    assert case.tools[0].requires_confirmation is True
