"""Shared fixtures for Atlas scoring tests."""

import pytest

from atlas.cases.builtin import get_builtin_cases


@pytest.fixture
def get_case():
    def find_case(case_id):
        for case in get_builtin_cases():
            if case.id == case_id:
                return case
        raise AssertionError(f"Case not found: {case_id}")

    return find_case


@pytest.fixture
def make_case():
    """Build a small custom TestCase: ``make_case(tools, expected, category=...)``."""
    from atlas.models import TestCase

    def build(tools, expected, category="general", case_id="custom"):
        return TestCase(id=case_id, prompt="p", tools=tools,
                        expected_calls=expected, category=category)

    return build
