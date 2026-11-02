"""Terminal styling, small display helpers, and live run progress."""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time

from atlas.atlas_cli.constants import DIFFICULTIES


def _color_on(stream) -> bool:
    """Color only for real terminals, and never when NO_COLOR/TERM=dumb."""
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _paint(text, code: str, stream=None) -> str:
    stream = stream or sys.stdout
    return f"\033[{code}m{text}\033[0m" if _color_on(stream) else str(text)


def _styler(code: str):
    return lambda text, stream=None: _paint(text, code, stream)


def _difficulty(d: str) -> str:
    return _paint(f"{d:<6}", _DIFF_COLOR.get(d, "0"))


def _score(v, width: int = 0, stream=None) -> str:
    """Score colored by quality band; pad *before* coloring so columns line up."""
    if v is None:
        return dim("n/a".rjust(width), stream)
    txt = f"{v:.2f}".rjust(width)
    return (green if v >= 0.8 else yellow if v >= 0.5 else red)(txt, stream)


def _err(msg: str) -> None:
    print(red(msg, sys.stderr), file=sys.stderr)


def _note(msg: str) -> None:
    print(dim(msg, sys.stderr), file=sys.stderr)


def _truncate(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: max(1, n - 1)] + "…"


def _human_size(n) -> str:
    if not isinstance(n, (int, float)):
        return ""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("atlas")
    except Exception:
        return "dev"


class LiveReporter:
    """Progress display for ``atlas run``, plugged into ``runner.execute_run``.

    On a TTY the running case gets a live status line (spinner, elapsed
    time, progress bar) that is erased and replaced by a result line when
    the case finishes. Off a TTY (pipes, CI) you just get the result lines.
    ``-v`` disables the live line because the runner prints its own detail
    blocks; ``-q`` prints failures only.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, *, verbose: bool = False, quiet: bool = False) -> None:
        self.verbose, self.quiet = verbose, quiet
        self.out = sys.stdout
        self.live = False
        self.total = self.done = self.idw = 0
        self._current = None  # (case, started_at, attempt, retries)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- runner hooks --
    def start(self, model, provider: str, cases: list, retries: int) -> None:
        self.out = sys.stdout  # resolved late: may be redirected to stderr
        self.live = (self.out.isatty() and os.environ.get("TERM") != "dumb"
                     and not self.verbose and not self.quiet)
        self.total = len(cases) * max(1, retries)
        self.idw = min(max((len(c.id) for c in cases), default=10), 36)
        if not self.quiet:
            n = len(cases)
            head = (
                f"  {bold('Atlas')}  ·  {cyan(model.id)}  ·  {provider}  ·  "
                f"{n} case{'s' if n != 1 else ''}"
            )
            if retries > 1:
                head += f"  ·  {retries} runs each"
            self._print(head)
            self._print(dim("  " + "─" * min(70, self._cols() - 4)))
        if self.live:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

    def case_start(self, case, attempt: int, retries: int) -> None:
        with self._lock:
            self._current = (case, time.monotonic(), attempt, retries)
            self._draw(0)

    def case_done(self, case, result, attempt: int, retries: int) -> None:
        with self._lock:
            self._clear()
            self._current = None
            self.done += 1
            if self.quiet and result.passed:
                return
            self._write(self._result_line(case, result, attempt, retries) + "\n")

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        with self._lock:
            self._clear()
            self._current = None
        if not self.quiet and self.total:
            self._write("\n")

    # -- rendering --
    @staticmethod
    def _cols() -> int:
        return shutil.get_terminal_size((100, 24)).columns

    def _write(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()

    def _print(self, text: str) -> None:
        with self._lock:
            self._clear()
            self._write(text + "\n")

    def _clear(self) -> None:
        if self.live:
            self._write("\r\033[K")

    @staticmethod
    def _reason(result) -> str:
        msg = (result.error_analysis or "").strip()
        if result.notes == "ERROR":
            m = re.search(r"timed out after \d+s", msg)
            return m.group(0) if m else msg
        return "" if msg in ("", "No major errors") else msg

    def _result_line(self, case, result, attempt: int, retries: int) -> str:
        err = result.notes == "ERROR"
        ctr = f"{self.done:>{len(str(self.total))}}/{self.total}"
        mark = green("✓") if result.passed else red("✗")
        score_plain = "ERR" if err else f"{result.scores.overall:.2f}"
        score = red(f"{score_plain:>4}") if err else _score(result.scores.overall, 4)
        lat = f"{result.latency_ms / 1000:.1f}s" if getattr(result, "latency_ms", 0) else ""
        extra = f" run {attempt + 1}/{retries}" if retries > 1 else ""
        line = (f"  {dim(ctr)}  {mark} {_truncate(case.id, self.idw):<{self.idw}}  "
                f"{_difficulty(case.difficulty)} {score}  {dim(f'{lat:>6}')}{dim(extra)}")
        if not result.passed:
            reason = self._reason(result)
            if reason:
                prefix = len(f"  {ctr}  X {'':<{self.idw}}  {'':6} {'':4}  {'':6}{extra}")
                room = max(10, self._cols() - prefix - 3)
                line += "  " + (red if err else dim)(_truncate(reason, room))
        return line

    def _draw(self, frame: int) -> None:
        if not (self.live and self._current):
            return
        case, t0, attempt, retries = self._current
        elapsed = time.monotonic() - t0
        width = 12
        filled = int(width * self.done / self.total) if self.total else 0
        bar = "█" * filled + "░" * (width - filled)
        tail = f"  {elapsed:>4.0f}s  {bar} {self.done}/{self.total}"
        avail = max(8, self._cols() - 1 - 4 - len(tail))
        name = _truncate(case.id, avail)
        self._write(f"\r\033[K  {cyan(self.FRAMES[frame % len(self.FRAMES)])} {name}{dim(tail)}")

    def _spin(self) -> None:
        i = 0
        while not self._stop.wait(0.1):
            i += 1
            with self._lock:
                self._draw(i)

bold, dim, red, green, yellow, cyan = (_styler(code) for code in ("1", "2", "31", "32", "33", "36"))

_DIFF_COLOR = {"easy": "32", "medium": "33", "hard": "31", "expert": "35"}
