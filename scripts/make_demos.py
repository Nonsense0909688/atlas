
"""
Render animated terminal demo GIFs for the Atlas README.

Usage:
    python scripts/make_demos.py
    python scripts/make_demos.py --out docs/assets
    python scripts/make_demos.py --only demo-run.gif

Requirements:
    pip install pillow

All terminal output is normalized to ASCII-safe characters to prevent
missing Unicode glyphs and emoji boxes across Windows, Linux and macOS.
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Configuration
# ============================================================

WIDTH = 960
HEIGHT = 540

BG = (13, 17, 23)
BAR = (22, 27, 34)
FG = (230, 237, 243)
DIM = (139, 148, 158)

GREEN = (63, 185, 80)
RED = (248, 81, 73)
CYAN = (121, 192, 255)
YELLOW = (210, 153, 34)
PROMPT = (126, 231, 135)

FONT_SIZE = 15
TITLE_FONT_SIZE = 13

LINE_HEIGHT = 22
LEFT_PADDING = 20
TOP_PADDING = 56

MAX_LINES = 21
MAX_TEXT_WIDTH = WIDTH - LEFT_PADDING * 2

# Prefer monospace fonts for consistent terminal alignment.
FONT_PATHS = [
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\lucon.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]

TITLE_FONT_PATHS = [
    r"C:\Windows\Fonts\consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


# ============================================================
# Font loading
# ============================================================

def load_font(
    paths: list[str],
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:

    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, ValueError):
            continue

    print(
        "[WARN] No compatible monospace font found. "
        "Using Pillow's default font."
    )

    return ImageFont.load_default()


FONT = load_font(FONT_PATHS, FONT_SIZE)
FONT_BOLD = FONT
TITLE_FONT = load_font(TITLE_FONT_PATHS, TITLE_FONT_SIZE)


# ============================================================
# Unicode normalization
# ============================================================

GLYPH_MAP = str.maketrans({
    # Status indicators
    "✓": "[OK]",
    "✔": "[OK]",
    "✗": "[FAIL]",
    "✘": "[FAIL]",

    # Progress bars
    "█": "#",
    "▓": "#",
    "▒": "=",
    "░": ".",

    # Separators
    "─": "-",
    "━": "-",
    "═": "=",
    "│": "|",
    "┃": "|",
    "·": "|",

    # Arrows
    "→": "->",
    "←": "<-",
    "⇒": "=>",
    "➜": "->",

    # Typography
    "—": "-",
    "–": "-",
    "…": "...",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",

    # Miscellaneous
    "⚡": "[FAST]",
    "🚀": "[RUN]",
    "💡": "[INFO]",
    "⚠": "[WARN]",
})


def safe_text(value: str) -> str:
    """
    Convert terminal output into ASCII-safe text.

    This avoids missing-glyph boxes on systems where Pillow
    cannot locate a suitable Unicode or emoji font.
    """

    value = str(value).translate(GLYPH_MAP)

    # Remove remaining non-ASCII characters.
    value = value.encode(
        "ascii",
        errors="replace",
    ).decode("ascii")

    return value


# ============================================================
# Terminal text helpers
# ============================================================

def get_text_width(text: str, font=FONT) -> float:
    return font.getlength(text)


def fit_text(
    text: str,
    max_width: int = MAX_TEXT_WIDTH,
    font=FONT,
) -> str:
    """
    Truncate text if it exceeds the available terminal width.
    """

    text = safe_text(text)

    if get_text_width(text, font) <= max_width:
        return text

    suffix = "..."

    while text and get_text_width(text + suffix, font) > max_width:
        text = text[:-1]

    return text + suffix


def prepare_lines(lines: list[str]) -> list[str]:
    """
    Normalize and wrap output lines to fit the terminal.
    """

    result = []

    for line in lines:
        line = safe_text(line)

        if not line:
            result.append("")
            continue

        wrapped = textwrap.wrap(
            line,
            width=85,
            subsequent_indent="    ",
            replace_whitespace=False,
            drop_whitespace=False,
        )

        result.extend(wrapped or [""])

    return result


def color_for(line: str) -> tuple[int, int, int]:
    """
    Determine terminal text color from output content.
    """

    stripped = line.strip()

    if stripped.startswith("$") or stripped.startswith("atlas "):
        return PROMPT

    if "[FAIL]" in line or "failed" in line.lower():
        return RED

    if "[OK]" in line or "passed" in line.lower():
        return GREEN

    if (
        line.startswith("  Atlas")
        or "report" in line.lower()
        or "Best:" in line
    ):
        return CYAN

    if "WARN" in line:
        return YELLOW

    return FG


# ============================================================
# Terminal rendering
# ============================================================

def draw_title_bar(draw: ImageDraw.ImageDraw) -> None:
    """
    Draw the terminal window title bar.
    """

    draw.rectangle(
        [0, 0, WIDTH, 36],
        fill=BAR,
    )

    buttons = [
        (255, 95, 86),
        (255, 189, 46),
        (39, 201, 63),
    ]

    for index, color in enumerate(buttons):
        x = 14 + index * 20

        draw.ellipse(
            [x, 12, x + 12, 24],
            fill=color,
        )

    draw.text(
        (84, 10),
        "atlas - demo",
        font=TITLE_FONT,
        fill=DIM,
    )


def draw_terminal_body(
    draw: ImageDraw.ImageDraw,
    command: str,
    shown_lines: list[str],
    cursor: bool,
) -> None:
    """
    Draw the command and currently visible output.
    """

    x = LEFT_PADDING
    y = TOP_PADDING

    display_command = fit_text("$ " + command)

    draw.text(
        (x, y),
        display_command,
        font=FONT,
        fill=PROMPT,
    )

    if cursor:
        cursor_x = x + get_text_width(display_command)

        draw.rectangle(
            [
                cursor_x + 2,
                y + 2,
                cursor_x + 10,
                y + 17,
            ],
            fill=FG,
        )

    y += 30

    for line in shown_lines[:MAX_LINES]:
        if y > HEIGHT - 24:
            break

        display_line = fit_text(line)

        draw.text(
            (x, y),
            display_line,
            font=FONT,
            fill=color_for(line),
        )

        y += LINE_HEIGHT


def render_frame(
    typed: str,
    shown_lines: list[str],
    cursor: bool = False,
) -> Image.Image:
    """
    Render one complete terminal frame.
    """

    image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        BG,
    )

    draw = ImageDraw.Draw(image)

    draw_title_bar(draw)

    draw_terminal_body(
        draw,
        typed,
        shown_lines,
        cursor,
    )

    return image


# ============================================================
# GIF generation
# ============================================================

def make_gif(
    command: str,
    output_lines: list[str],
    output_path: Path,
    hold_ms: int = 1800,
) -> Path:
    """
    Create an animated terminal GIF.

    Animation:
        1. Command is typed character by character.
        2. Command executes.
        3. Output appears line by line.
        4. Final output remains visible.
    """

    frames: list[Image.Image] = []
    durations: list[int] = []

    output_lines = prepare_lines(output_lines)
    command = safe_text(command)

    # --------------------------------------------------------
    # Phase 1: Typing animation
    # --------------------------------------------------------

    typing_step = 2

    for index in range(
        0,
        len(command) + 1,
        typing_step,
    ):
        frames.append(
            render_frame(
                command[:index],
                [],
                cursor=True,
            )
        )

        durations.append(28)

    # --------------------------------------------------------
    # Phase 2: Command execution
    # --------------------------------------------------------

    frames.append(
        render_frame(
            command,
            [],
            cursor=False,
        )
    )

    durations.append(260)

    # --------------------------------------------------------
    # Phase 3: Output animation
    # --------------------------------------------------------

    shown_lines: list[str] = []

    index = 0

    while index < len(output_lines):
        remaining = len(output_lines) - index

        step = 1 if remaining > 14 else 2

        shown_lines.extend(
            output_lines[index:index + step]
        )

        index += step

        frames.append(
            render_frame(
                command,
                shown_lines,
                cursor=False,
            )
        )

        durations.append(110)

    # --------------------------------------------------------
    # Phase 4: Final frame
    # --------------------------------------------------------

    frames.append(
        render_frame(
            command,
            output_lines,
            cursor=True,
        )
    )

    durations.append(hold_ms)

    # --------------------------------------------------------
    # Save GIF
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Quantize frames to reduce GIF size.
    palette_frames = [
        frame.quantize(
            colors=128,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        for frame in frames
    ]

    palette_frames[0].save(
        output_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )

    size_kb = output_path.stat().st_size // 1024

    print(
        f"[OK] Wrote {output_path} "
        f"({size_kb} KB, {len(frames)} frames)"
    )

    return output_path


# ============================================================
# Demo definitions
# ============================================================

DEMOS: dict[str, tuple[str, list[str]]] = {

    "demo-run.gif": (
        "atlas run --model qwen3.5:2b",
        [
            "  Atlas  |  qwen3.5:2b  |  ollama  |  32 cases",
            "  " + "-" * 62,

            "   1/32   [OK] weather-single              easy   1.00",
            "   2/32   [OK] math-single                 easy   1.00",
            "   3/32   [OK] search-single             medium 0.98",

            "  ...",

            "  30/32   [OK] parallel-weather-three      hard   0.92",
            "  31/32   [OK] depchain-weather-search     hard   0.88",
            "  32/32   [FAIL] multi-three               hard   0.45",

            "",

            "  Atlas report  |  qwen3.5:2b",
            "  [OK] 31 passed   [FAIL] 1 failed   32 cases in 12.4s",

            "  Overall              ##################.. 0.91",
            "  Tool selection       ########## 1.00",
            "  Argument accuracy    #########. 0.89",
            "  Hallucination rate   .......... 0.00",

            "",

            "  1 failed, 31 passed in 12.4s",
        ],
    ),

    "demo-batch.gif": (
        "atlas batch --models qwen3.5:2b,llama3.1:8b -o batch.json",
        [
            "  -- qwen3.5:2b --",
            "  -- llama3.1:8b --",

            "",

            "  Atlas batch - 2 model(s)",
            "  " + "-" * 66,

            "",

            "  #   model                    overall    passed    tool    args    halluc",

            "  1   qwen3.5:2b               0.91 ####### 31/32     1.00    0.89    0.00",
            "  2   llama3.1:8b              0.84 ######. 27/32     0.94    0.81    0.03",

            "",

            "  Best: qwen3.5:2b (overall 0.91)",
            "  report -> batch.json",
        ],
    ),

    "demo-list.gif": (
        "atlas list --difficulty hard",
        [
            "Atlas - 7 cases",

            "",

            "complex_args (1)",
            "  event-attendees               hard    Create a meeting titled 'Sprint Planning' ...",

            "",

            "dependency_chain (2)",
            "  depchain-weather-search       hard    Step 1: get weather for Chicago ...",
            "  depchain-calc-weather         expert  Calculate the average of 10 and 20 ...",

            "",

            "multi_tool (1)",
            "  multi-three                   hard    Calculate 100 + 200, then check weather ...",

            "",

            "parallel_calls (2)",
            "  parallel-weather-three        hard    Get the weather in Tokyo, Sydney, Paris ...",
            "  expert-search-calc-weather    expert  Search for 'best hiking trails Colorado' ...",
        ],
    ),
}


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate animated Atlas terminal demos."
    )

    parser.add_argument(
        "--out",
        default="docs/assets",
        help="Output directory for GIFs.",
    )

    parser.add_argument(
        "--only",
        help="Generate only one demo GIF.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_directory = Path(args.out)

    if args.only and args.only not in DEMOS:
        raise SystemExit(
            f"Unknown demo: {args.only}\n"
            f"Available demos: {', '.join(DEMOS)}"
        )

    selected_demos = (
        {args.only: DEMOS[args.only]}
        if args.only
        else DEMOS
    )

    print("Atlas terminal demo generator")
    print("-" * 40)

    for filename, (command, output_lines) in selected_demos.items():
        make_gif(
            command=command,
            output_lines=output_lines,
            output_path=output_directory / filename,
        )

    print("-" * 40)
    print(f"Generated {len(selected_demos)} demo(s).")


if __name__ == "__main__":
    main()
