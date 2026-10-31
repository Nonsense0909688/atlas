"""Single-case and suite execution, retry handling, and progress hooks."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time

from atlas.atlas_runner.case_loading import filter_cases, load_cases
from atlas.atlas_runner.parsing import response_to_tool_calls
from atlas.atlas_runner.providers import make_provider
from atlas.atlas_runner.reporting import _cyan, _fail, _ok
from atlas.evaluator.closed_loop import StatefulToolbox, score_recovery
from atlas.evaluator.perturbations import (
    paraphrase_prompt, rename_params, shuffle_tools, truncate_descriptions,
)
from atlas.evaluator.reliability import reliability_summary
from atlas.evaluator.scaling import scale_tools
from atlas.evaluator.scoring import aggregate_scores, compute_consistency, score_case
from atlas.evaluator.validation import validate_structure
from atlas.exceptions import ConfigError, ProviderError
from atlas.models import (
    EvalResult, ModelConfig, ScoreBreakdown, TestCase, ToolCallResult, TurnLog,
)
from atlas.providers.base import BaseProvider

logger = logging.getLogger(__name__)


def _calling_mode(model: ModelConfig) -> str:
    """Return the configured tool-calling mode for a model."""
    extra_params = getattr(model, "extra_params", None) or {}
    return getattr(model, "calling_mode", None) or extra_params.get(
        "calling_mode",
        "native",
    )


def _prompt_mode_system(case: TestCase) -> str:
    """Build instructions and a tool catalogue for text-based tool calling."""
    tool_descriptions = [
        (
            f"- {tool.name}: {tool.description}\n"
            f"  Parameters: {json.dumps(tool.parameters, ensure_ascii=False)}"
        )
        for tool in case.tools
    ]
    tools_text = "\n".join(tool_descriptions) or "(no tools available)"
    return (
        "You are completing a tool-use benchmark. Do not claim to execute tools. "
        "Select tools only when needed. Respond with one JSON object per call, "
        'using this shape: {"name": "tool_name", "arguments": {}}. '
        "For multiple calls, return a JSON array. If no tool is appropriate, "
        "answer normally without inventing a call.\n\n"
        f"AVAILABLE TOOLS:\n{tools_text}"
    )


def run_case_closed_loop(
    provider: BaseProvider,
    model: ModelConfig,
    case: TestCase,
    *,
    verbose: bool = False,
) -> EvalResult:
    """Execute a bounded mock-tool conversation and grade its final state."""
    toolbox = StatefulToolbox(
        case,
        initial_state=copy.deepcopy(case.initial_state),
    )
    prompt_mode = _calling_mode(model) == "prompt"
    messages = [{"role": "user", "content": case.prompt}]
    if prompt_mode:
        messages.insert(
            0,
            {"role": "system", "content": _prompt_mode_system(case)},
        )

    turns: list[TurnLog] = []
    all_calls: list[ToolCallResult] = []
    all_tool_outputs: list = []
    final_answer = ""
    had_tool_error = False
    total_latency_ms = 0.0
    prompt_tokens = completion_tokens = total_tokens = 0
    max_turns = max(1, int(getattr(case, "max_turns", 5) or 1))

    tools_by_name = {tool.name: tool for tool in case.tools}

    for turn_index in range(max_turns):
        started_at = time.monotonic()
        response = provider.chat_messages(
            model,
            messages,
            tools=[] if prompt_mode else case.tools,
        )
        measured_latency = (time.monotonic() - started_at) * 1000
        response_latency = (
            response.latency_ms
            if response.latency_ms is not None
            else measured_latency
        )
        total_latency_ms += response_latency
        prompt_tokens += response.prompt_tokens or 0
        completion_tokens += response.completion_tokens or 0
        total_tokens += response.total_tokens or 0

        calls = response_to_tool_calls(response)
        turn = TurnLog(
            turn_index=turn_index,
            model_text=response.text or "",
            tool_calls=calls,
        )
        turns.append(turn)
        all_calls.extend(calls)

        if not calls:
            final_answer = response.text or ""
            break

        assistant_content = response.text or ""
        if not assistant_content:
            assistant_content = "Requested tool calls: " + ", ".join(
                call.name or "<unknown>" for call in calls
            )
        messages.append({"role": "assistant", "content": assistant_content})

        for call in calls:
            if call.parse_error:
                result = {"tool": call.name, "error": call.parse_error, "output": None}
            else:
                tool = tools_by_name.get(call.name or "")
                is_valid, validation_error = validate_structure(call, tool)
                if not is_valid:
                    result = {
                        "tool": call.name,
                        "error": validation_error,
                        "output": None,
                    }
                else:
                    result = toolbox.execute(call.name, call.arguments)

            turn.tool_results.append(result)
            if result.get("error"):
                had_tool_error = True
                turn.tool_errors.append(result["error"])
            if result.get("output") is not None:
                all_tool_outputs.append(result["output"])
            messages.append({
                "role": "user",
                "content": (
                    f"Tool result for {call.name}: "
                    f"{json.dumps(result, ensure_ascii=False, default=str)}"
                ),
            })

    result = score_case(
        case,
        all_calls,
        model_id=model.id,
        actual_text=final_answer,
        final_state=toolbox.snapshot(),
        tool_outputs=all_tool_outputs,
    )
    result.turns = turns
    result.final_answer = final_answer
    result.final_state = toolbox.snapshot()
    result.had_tool_error = had_tool_error
    result.latency_ms = total_latency_ms
    result.prompt_tokens = prompt_tokens or None
    result.completion_tokens = completion_tokens or None
    result.total_tokens = total_tokens or None

    recovery = score_recovery(turns, result.scores.final_answer_accuracy, had_tool_error)
    result.scores.recovery_score = recovery
    result.recovered = None if recovery is None else recovery >= 1.0

    if verbose:
        _print_case_detail(case, result)
    return result


def run_case(
    provider: BaseProvider,
    model: ModelConfig,
    case: TestCase,
    *,
    verbose: bool = False,
) -> EvalResult:
    """Run a case once, selecting single-turn or closed-loop execution."""
    extra_params = getattr(model, "extra_params", None) or {}
    if (
        getattr(case, "multi_turn", False)
        or extra_params.get("closed_loop", False)
    ):
        return run_case_closed_loop(provider, model, case, verbose=verbose)
    return run_case_single(provider, model, case, verbose=verbose)


def run_case_single(
    provider: BaseProvider,
    model: ModelConfig,
    case: TestCase,
    *,
    verbose: bool = False,
) -> EvalResult:
    """Run one case once: call the model, convert to calls, score them.

    Provider exceptions propagate to the caller, which records them as
    ERROR results and continues with the next case.
    """
    t0 = time.monotonic()
    # Prompt-based calling mode: inject tool descriptions into prompt, parse JSON
    if _calling_mode(model) == "prompt":
        sys_prompt = _prompt_mode_system(case)
        user_prompt = f"{sys_prompt}\n\nUSER TASK:\n{case.prompt}"
        response = provider.chat(model, user_prompt, tools=[])
    else:
        response = provider.chat_case(model, case)
    latency_ms = (time.monotonic() - t0) * 1000

    actual_calls = response_to_tool_calls(response)
    result = score_case(
        case,
        actual_calls,
        model_id=model.id,
        actual_text=response.text,
    )
    # Enrich with performance + provenance fields the scorer doesn't know.
    result.latency_ms = latency_ms
    result.raw_response = response.raw
    result.prompt_tokens = response.prompt_tokens
    result.completion_tokens = response.completion_tokens
    result.total_tokens = response.total_tokens

    if verbose:
        _print_case_detail(case, result)
    return result


def _error_result(case: TestCase, model: ModelConfig, exc: Exception) -> EvalResult:
    """Build a failed ``EvalResult`` for a case whose provider call crashed."""
    return EvalResult(
        case_id=case.id,
        category=case.category,
        model_id=model.id,
        scores=ScoreBreakdown(overall=0.0),
        details=[f"ERROR: {exc}"],
        notes="ERROR",
        error_analysis=str(exc),
        difficulty=case.difficulty,
    )


def _print_case_detail(case: TestCase, result: EvalResult) -> None:
    """Verbose per-case dump (``--verbose``): expected/actual/scores/details."""
    print(f"  prompt: {case.prompt}")
    print(f"  expected: {[(e.name, e.arguments) for e in case.expected_calls]}")
    print(f"  actual:   {[(c.name, c.arguments) for c in result.actual_calls]}")
    s = result.scores
    print(
        f"  scores: overall={s.overall:.2f} tool={s.tool_selection} "
        f"args={s.argument_accuracy} struct={s.structure_validity} "
        f"multi={s.multi_tool_planning} precision={s.argument_precision} "
        f"recall={s.argument_recall} halluc={s.hallucination_rate}"
    )
    for d in result.details:
        print(f"    · {d}")
    if result.error_analysis != "No major errors":
        print(f"  error: {result.error_analysis}")


def execute_run(args: argparse.Namespace, reporter=None) -> tuple[dict, list[EvalResult]]:
    """Run the benchmark for parsed args. Returns (summary, results).

    ``reporter`` is an optional progress sink (the CLI passes a live one).
    It needs ``start(model, provider, cases, retries)``,
    ``case_start(case, attempt, retries)`` and
    ``case_done(case, result, attempt, retries)``. Without one, plain
    ``print`` lines are used, so library/test callers behave as before.

    Split out from ``main`` so tests (and future library use) can drive a
    full run without going through argparse/stdout.
    """
    cases = load_cases(args.suite)
    cases = filter_cases(
        cases,
        case_id=args.case,
        category=args.category,
        difficulty=args.difficulty,
    )
    model = ModelConfig(
        id=args.model,
        provider=getattr(args, "provider", "ollama"),
        temperature=getattr(args, "temp", 0.2),
        max_tokens=getattr(args, "max_tokens", 2048),
        calling_mode=getattr(args, "calling_mode", "native") or "native",
        extra_params={
            "closed_loop": bool(getattr(args, "closed_loop", False)),
            "calling_mode": getattr(args, "calling_mode", "native") or "native",
        },
    )
    # Suite split + language filter
    _split = getattr(args, "split", None)
    _lang = getattr(args, "language", None)
    if _split and _split != "all":
        cases = [c for c in cases if getattr(c, "split", "train") == _split]
    if _lang:
        cases = [c for c in cases if getattr(c, "language", "en") == _lang]
    # Perturbations + scaling are applied per-case below via loop flags
    _scale_to = getattr(args, "scale_tools", None)
    _shuffle = bool(getattr(args, "shuffle_tools", False))
    _truncate = getattr(args, "truncate_desc", None)
    _paraphrase = bool(getattr(args, "paraphrase", False))
    rename_mapping = None
    rename_json = getattr(args, "rename_params", None)
    if rename_json:
        try:
            rename_mapping = json.loads(rename_json)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"--rename-params must be valid JSON: {exc.msg}") from exc
        if not isinstance(rename_mapping, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in rename_mapping.items()
        ):
            raise ConfigError("--rename-params must be a JSON object of string keys and values")

    if args.dry_run:
        print(f"Atlas dry run — {len(cases)} cases")
        for c in cases:
            print(f"  {c.id} [{c.category}/{c.difficulty}]: {c.prompt[:80]}")
        return aggregate_scores([]), []

    provider = make_provider(
        getattr(args, "provider", "ollama"),
        api_base=getattr(args, "api_base", None),
        api_key=getattr(args, "api_key", None),
    )
    if reporter:
        reporter.start(
            model,
            getattr(args, "provider", "ollama"),
            cases,
            getattr(args, "retries", 1),
        )
    else:
        print(f"{'=' * 70}")
        print(f"Atlas — model={_cyan(model.id)} provider={args.provider} "
              f"cases={len(cases)} retries={args.retries}")
        print(f"{'=' * 70}")

    # Run each case × retries. Provider errors become failed results so one
    # bad call never aborts the whole benchmark.
    all_results: list[EvalResult] = []
    per_case_runs: dict[str, list[EvalResult]] = {}
    for _base_case in cases:
        for attempt in range(max(1, getattr(args, "retries", 1))):
            # Apply each transform to a fresh case so retries stay independent.
            case = copy.deepcopy(_base_case)
            if _scale_to:
                case = scale_tools(case, int(_scale_to))
            if _shuffle:
                case = shuffle_tools(case, seed=attempt)
            if _truncate:
                case = truncate_descriptions(case, max_chars=int(_truncate))
            if _paraphrase:
                case.prompt = paraphrase_prompt(case.prompt, seed=attempt)
            if rename_mapping:
                case = rename_params(case, rename_mapping)
            if reporter:
                reporter.case_start(case, attempt, args.retries)
            try:
                result = run_case(provider, model, case, verbose=args.verbose)
            except ProviderError as e:
                # With a reporter the error is shown inline; don't also log it
                # (a log line would tear the live progress display).
                (logger.debug if reporter else logger.warning)(
                    "case %s provider error: %s", case.id, e)
                result = _error_result(case, model, e)
            result.run_index = attempt
            all_results.append(result)
            per_case_runs.setdefault(case.id, []).append(result)
            if reporter:
                reporter.case_done(case, result, attempt, args.retries)
            else:
                status = _ok(result.case_id) if result.passed else _fail(result.case_id)
                rep = f" (run {attempt + 1}/{args.retries})" if args.retries > 1 else ""
                print(f"  {status} overall={result.scores.overall:.2f}{rep}")

    summary = aggregate_scores(all_results)
    # Reliability: pass@k / pass^k and confidence intervals.
    results_by_case = {
        case_id: [result.passed for result in runs]
        for case_id, runs in per_case_runs.items()
    }
    overall_scores = [result.scores.overall for result in all_results]
    reliability = reliability_summary(results_by_case, overall_scores)
    summary["reliability"] = reliability
    for key, value in reliability.items():
        if key.startswith(("pass_at", "pass_pow")):
            summary[key] = value
    if args.retries > 1 and per_case_runs:
        summary["consistency"] = (
            sum(compute_consistency(runs) for runs in per_case_runs.values())
            / len(per_case_runs)
        )
    return summary, all_results
