"""Atlas data models - shared vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Difficulty = str


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    destructive: bool = False
    requires_confirmation: bool = False


@dataclass
class ModelConfig:
    id: str
    provider: str
    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 1.0
    timeout: int = 60
    api_base: str | None = None
    api_key: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    calling_mode: str = "native"


@dataclass
class ExpectedToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResult:
    name: str | None = None
    arguments: Any = field(default_factory=dict)
    raw: Any = None
    parse_error: str | None = None


@dataclass
class TurnLog:
    turn_index: int = 0
    model_text: str = ""
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    tool_errors: list[Any] = field(default_factory=list)


@dataclass
class TestCase:
    __test__ = False  # Prevent pytest from treating this data model as a test class.

    id: str
    prompt: str
    tools: list[ToolDefinition] = field(default_factory=list)
    expected_calls: list[ExpectedToolCall] = field(default_factory=list)
    mock_outputs: dict[str, Any] = field(default_factory=dict)
    category: str = "general"
    difficulty: Difficulty = "easy"
    rubric: list[str] = field(default_factory=list)
    matchers: dict[str, Any] = field(default_factory=dict)
    alternative_sequences: list = field(default_factory=list)
    multi_turn: bool = False
    max_turns: int = 5
    initial_state: dict[str, Any] = field(default_factory=dict)
    expected_state: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    final_answer_matcher: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] | None = None
    forbidden_tools: list[str] = field(default_factory=list)
    mock_errors: dict[str, Any] = field(default_factory=dict)
    injection: dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    split: str = "train"
    version: str = "2.0.0"
    requires_confirmation: bool = False


@dataclass
class ScoreBreakdown:
    tool_selection: float | None = None
    argument_accuracy: float | None = None
    structure_validity: float | None = None
    multi_tool_planning: float | None = None
    argument_precision: float | None = None
    argument_recall: float | None = None
    hallucination_rate: float | None = None
    abstention_accuracy: float | None = None
    consistency: float | None = None
    final_answer_accuracy: float | None = None
    recovery_score: float | None = None
    state_accuracy: float | None = None
    safety_score: float | None = None
    overcall_score: float | None = None
    injection_resistance: float | None = None
    overall: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tool_selection": self.tool_selection,
            "argument_accuracy": self.argument_accuracy,
            "structure_validity": self.structure_validity,
            "multi_tool_planning": self.multi_tool_planning,
            "argument_precision": self.argument_precision,
            "argument_recall": self.argument_recall,
            "hallucination_rate": self.hallucination_rate,
            "abstention_accuracy": self.abstention_accuracy,
            "consistency": self.consistency,
            "final_answer_accuracy": self.final_answer_accuracy,
            "recovery_score": self.recovery_score,
            "state_accuracy": self.state_accuracy,
            "safety_score": self.safety_score,
            "overcall_score": self.overcall_score,
            "injection_resistance": self.injection_resistance,
            "overall": self.overall,
        }


@dataclass
class EvalResult:
    case_id: str
    category: str
    model_id: str
    scores: ScoreBreakdown
    details: list[str] = field(default_factory=list)
    actual_calls: list[ToolCallResult] = field(default_factory=list)
    expected_calls: list[ExpectedToolCall] = field(default_factory=list)
    raw_response: Any = None
    latency_ms: float | None = None
    passed: bool = False
    notes: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error_analysis: str = ""
    difficulty: str = "easy"
    run_index: int = 0
    final_answer: str | None = None
    final_state: dict[str, Any] = field(default_factory=dict)
    turns: list[TurnLog] = field(default_factory=list)
    had_tool_error: bool = False
    recovered: bool | None = None
