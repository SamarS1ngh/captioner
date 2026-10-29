"""Build an ASS subtitle document with per-word karaoke highlight.

In ASS karaoke (\\k) a word starts in SecondaryColour and flips to
PrimaryColour once it is sung. So the style's PrimaryColour is the *highlight*
and SecondaryColour is the resting text colour.
"""
from __future__ import annotations

from dataclasses import dataclass

from captioner.models import Transcript
from captioner.timing import distribute_timing


@dataclass(frozen=True)
class AssStyle:
    font_name: str = "Arial"
    font_size: int = 72
    primary_color: str = "&H00FFFFFF"    # resting text: white (&HAABBGGRR)
    highlight_color: str = "&H0000E5FF"  # sung word: amber/yellow (BGR)
    outline_color: str = "&H00000000"
    back_color: str = "&H00000000"       # box/shadow colour (&HAABBGGRR)
    outline: int = 5
    shadow: int = 0
    bold: bool = True
    border_style: int = 1  # 1 = outline+shadow, 3 = opaque box behind text
    margin_v: int = 260
    alignment: int = 2  # bottom-center
    play_res_x: int = 1080
    play_res_y: int = 1920


def format_ass_time(seconds: float) -> str:
    total_cs = max(0, round(seconds * 100))
    h, rem = divmod(total_cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape_ass(text: str) -> str:
    """Make text literal: braces/backslashes cannot start override tags."""
    text = " ".join(text.split())  # collapses newlines/tabs/runs of spaces
    # U+2060 word joiner after a backslash stops "\N" / "\h" being read as codes.
    return (
        text.replace("\\", "\\\u2060")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _karaoke_text(seg) -> str:
    tokens = seg.text.strip().split()
    if not tokens:
        return ""
    words = distribute_timing(
        seg.start, seg.end, tokens, [(w.start, w.end) for w in seg.words]
    )
    # Boundaries in cs relative to seg.start; each word runs until the next
    # word starts (the last until seg.end), so the \k values sum to the
    # segment duration exactly.
    def cs(t: float) -> int:
        return round((t - seg.start) * 100)

    total = cs(seg.end)
    starts = [min(max(cs(w.start), 0), total) for w in words]
    for i in range(1, len(starts)):
        starts[i] = max(starts[i], starts[i - 1])
    bounds = starts + [total]

    out = []
    if starts[0] > 0:
        out.append(f"{{\\k{starts[0]}}}")
    for i, w in enumerate(words):
        sep = " " if i < len(words) - 1 else ""
        out.append(f"{{\\k{bounds[i + 1] - bounds[i]}}}{escape_ass(w.text)}{sep}")
    return "".join(out)


def build_ass(t: Transcript, style: AssStyle = AssStyle()) -> str:
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {style.play_res_x}\n"
        f"PlayResY: {style.play_res_y}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{style.font_name},{style.font_size},"
        f"{style.highlight_color},{style.primary_color},{style.outline_color},"
        f"{style.back_color},{-1 if style.bold else 0},0,0,0,100,100,0,0,"
        f"{style.border_style},{style.outline},{style.shadow},"
        f"{style.alignment},60,60,{style.margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    lines = [
        f"Dialogue: 0,{format_ass_time(seg.start)},{format_ass_time(seg.end)},"
        f"Default,,0,0,0,,{_karaoke_text(seg)}"
        for seg in t.segments
    ]
    return header + "\n".join(lines) + "\n"


# --- Named style presets -----------------------------------------------------
# Colours are &HAABBGGRR (AA=00 is opaque). primary_color = resting word,
# highlight_color = the word currently being spoken (karaoke flip).
PRESETS: dict[str, AssStyle] = {
    # Clean white text, amber highlight — the original default.
    "classic": AssStyle(),
    # Big bold MrBeast-style pop: heavy font, pure-yellow highlight, fat outline.
    "bold_pop": AssStyle(
        font_name="Arial Black", font_size=88,
        primary_color="&H00FFFFFF", highlight_color="&H0000FFFF",
        outline=8, shadow=2,
    ),
    # Hot-pink highlight on bold white.
    "pink_pop": AssStyle(
        font_name="Arial Black", font_size=84,
        primary_color="&H00FFFFFF", highlight_color="&H00AA00FF",
        outline=7, shadow=2,
    ),
    # Neon: cyan highlight, strong black outline for glow-on-dark legibility.
    "neon": AssStyle(
        font_name="Arial", font_size=78,
        primary_color="&H00FFFFFF", highlight_color="&H00FFFF00",
        outline=5, shadow=3,
    ),
    # Minimal: soft grey resting text, clean white highlight, thin outline.
    "minimal": AssStyle(
        font_name="Helvetica", font_size=60, bold=False,
        primary_color="&H00C8C8C8", highlight_color="&H00FFFFFF",
        outline=3,
    ),
    # Boxed: opaque black box behind the text (BorderStyle 3), yellow highlight.
    "boxed": AssStyle(
        font_name="Arial", font_size=68,
        primary_color="&H00FFFFFF", highlight_color="&H0000FFFF",
        border_style=3, back_color="&H80000000", outline=4, shadow=0,
    ),
}

DEFAULT_PRESET = "classic"


def preset_names() -> list[str]:
    """Preset names in display order (default first)."""
    return list(PRESETS.keys())


def get_style(name: str) -> AssStyle:
    """Return the preset's AssStyle; unknown names fall back to the default."""
    return PRESETS.get(name, PRESETS[DEFAULT_PRESET])
