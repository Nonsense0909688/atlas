"""Provider discovery, preflight checks, and model/setup commands."""

from __future__ import annotations

import argparse
import difflib
import sys

from atlas.atlas_cli.constants import EXIT_FAIL, EXIT_OK
from atlas.atlas_cli.terminal import (
    _human_size, _note, _truncate, bold, dim, green, red, yellow,
)
from atlas.exceptions import ConfigError, ProviderError
from atlas.atlas_runner.providers import make_provider


def _model_name(m) -> str:
    if isinstance(m, dict):
        return str(m.get("name") or m.get("model") or "")
    return str(m)


def _preflight(provider: str, model_id: str) -> None:
    """Fail fast (seconds, not after a long run) if the provider or model is unusable."""
    if provider != "ollama":
        return
    try:
        installed = {_model_name(m) for m in make_provider(provider).list_models()} - {""}
    except ProviderError as e:
        raise ProviderError(
            f"can't reach Ollama ({e}). Is it running? Try `ollama serve`, "
            "or run `atlas doctor`."
        ) from e
    if model_id in installed or f"{model_id}:latest" in installed:
        return
    hint = ""
    close = difflib.get_close_matches(model_id, sorted(installed), n=3, cutoff=0.4)
    if close:
        hint = f" Did you mean: {', '.join(close)}?"
    raise ConfigError(
        f"model '{model_id}' is not installed.{hint} "
        f"Install it with `ollama pull {model_id}` or see `atlas models`.")


def _list_models(args: argparse.Namespace) -> int:
    """``atlas models``: what the provider has, with sizes."""
    try:
        models = make_provider(args.provider).list_models()
    except ProviderError as e:
        if args.provider == "ollama":
            raise ProviderError(
                f"can't reach Ollama ({e}). Is it running? Try `ollama serve`."
            ) from e
        raise ProviderError(f"can't reach {args.provider} ({e}).") from e
    except NotImplementedError:
        _note(f"provider '{args.provider}' does not support listing models.")
        return EXIT_OK
    rows = sorted(
        (
            _model_name(model),
            _human_size(model.get("size")) if isinstance(model, dict) else "",
        )
        for model in models
    )
    if not rows:
        _note("No models installed. Try: ollama pull qwen3.5:2b")
        return EXIT_OK
    width = max(len(n) for n, _ in rows)
    print(f"{bold(str(len(rows)))} model{'s' if len(rows) != 1 else ''} on {args.provider}\n")
    for name, size in rows:
        print(f"  {name:<{width}}  {dim(size)}")
    print("\n" + dim("Run one: atlas run -m <name>"))
    return EXIT_OK


def _doctor(args: argparse.Namespace) -> int:
    """``atlas doctor``: one-screen health check."""
    provider = getattr(args, "provider", "ollama") or "ollama"
    ok_all = True

    def line(ok: bool, label: str, detail: str = "", hint: str = "") -> None:
        nonlocal ok_all
        ok_all = ok_all and ok
        mark = green("✓") if ok else red("✗")
        print(f"  {mark} {label:<18} {dim(detail)}")
        if hint and not ok:
            print(f"      {yellow(hint)}")

    print(bold("Atlas doctor") + "\n")
    v = sys.version_info
    line(v >= (3, 10), "Python", f"{v.major}.{v.minor}.{v.micro}", "Python 3.10+ required")

    try:
        from atlas.cases.builtin import get_builtin_cases
        line(True, "Built-in cases", f"{len(get_builtin_cases())} loaded")
    except Exception as e:
        line(False, "Built-in cases", str(e))

    try:
        prov = make_provider(
            provider,
            api_base=getattr(args, "api_base", None),
            api_key=getattr(args, "api_key", None),
        )
        line(True, provider.capitalize(), "reachable")
        try:
            models = prov.list_models()
        except NotImplementedError:
            line(True, "Models", "n/a (not listable)")
        else:
            if models:
                line(True, "Models", f"{len(models)} installed")
            elif provider == "ollama":
                line(False, "Models", "0 installed", "Install one: ollama pull qwen3.5:2b")
            else:
                line(True, "Models", "n/a (not listable or none installed)")
    except Exception as e:
        hint = "Start it with: ollama serve" if provider == "ollama" else (
            "Check the provider is reachable / API key is set" if provider != "dummy" else "")
        line(False, provider.capitalize(), _truncate(str(e), 70), hint)

    print()
    print(green("All good.") if ok_all else yellow("Fix the items above, then re-run `atlas doctor`."))
    return EXIT_OK if ok_all else EXIT_FAIL


def _generate_cmd(args: argparse.Namespace) -> int:
    from atlas.models import ModelConfig
    from atlas.tools.generator import generate_cases_llm, save_review_queue
    provider = make_provider(args.provider, api_base=args.api_base, api_key=args.api_key)
    model = ModelConfig(id=args.model, provider=args.provider)
    cases = generate_cases_llm(
        provider,
        model,
        count=args.count,
        domain=args.domain,
        difficulty=args.difficulty,
    )
    save_review_queue(cases, args.output)
    print(f"generated {len(cases)} cases, review queue at {args.output}", file=sys.stderr)
    print("Human review: edit/approve then move to suite.", file=sys.stderr)
    return EXIT_OK
