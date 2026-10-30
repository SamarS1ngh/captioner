"""Build an ASS subtitle document in one of several caption *styles*.

Each style is a distinct layout/animation (not just colours):

  karaoke      full line, active word fills colour as spoken (\\k)
  highlighter  full line, a translucent marker box sweeps word to word
  wordbox      full line, active word sits in a solid colour box (Submagic look)
  boxed        full line inside a solid box bar, active word highlighted
  mrbeast      full line, every word a different bright colour, bold
  bounce       words accumulate, each bounces in
  hormozi      1-2 words at a time, huge ALL-CAPS, centred, pops in
  pop          one word at a time, centred, scales/bounces in
  phrase       a whole phrase fades in, replaced by the next
  typewriter   words appear one-by-one and accumulate

Any word can carry a per-word font/size override (Segment.word_styles,
keyed by token index) applied on top of the chosen style.
"""
from __future__ import annotations

from dataclasses import dataclass

from captioner.models import Transcript, Word
from captioner.timing import distribute_timing


@dataclass(frozen=True)
class AssStyle:
    kind: str = "karaoke"
    font_name: str = "Arial"
    font_size: int = 72
    primary_color: str = "&H00FFFFFF"    # resting text: white (&HAABBGGRR)
    highlight_color: str = "&H0000E5FF"  # active word / marker (BGR)
    outline_color: str = "&H00000000"
    back_color: str = "&H00000000"       # box/shadow colour (&HAABBGGRR)
    outline: int = 5
    shadow: int = 0
    bold: bool = True
    border_style: int = 1  # 1 = outline+shadow, 3 = opaque box behind text
    margin_v: int = 260
    alignment: int = 2  # 2 = bottom-centre, 5 = screen centre
    uppercase: bool = False
    palette: tuple = ()  # mrbeast per-word colour cycle
    pos_x: float = -1.0  # 0..1 of PlayResX; <0 = use alignment+margin instead
    pos_y: float = -1.0  # 0..1 of PlayResY
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
    text = " ".join(text.split())
    return (
        text.replace("\\", "\\⁠")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


# --- shared helpers ----------------------------------------------------------

def _timed(seg) -> list[Word]:
    tokens = seg.text.strip().split()
    if not tokens:
        return []
    return distribute_timing(
        seg.start, seg.end, tokens, [(w.start, w.end) for w in seg.words]
    )


def _cap(text: str, style: AssStyle) -> str:
    return text.upper() if style.uppercase else text


def _fmt_word(text_escaped: str, ov: dict | None, style: AssStyle) -> str:
    """Wrap one word with its per-word font/size override, restoring the
    style's own font/size afterwards (so only this word changes)."""
    if not ov:
        return text_escaped
    tags = ""
    if ov.get("font"):
        tags += "\\fn" + str(ov["font"])
    if ov.get("size"):
        tags += "\\fs" + str(int(ov["size"]))
    if not tags:
        return text_escaped
    restore = f"\\fn{style.font_name}\\fs{style.font_size}"
    return f"{{{tags}}}{text_escaped}{{{restore}}}"


def _wtext(w: Word, idx: int, ws: dict, style: AssStyle) -> str:
    return _fmt_word(escape_ass(_cap(w.text, style)), ws.get(idx), style)


def _event(t0: float, t1: float, text: str) -> str:
    if t1 <= t0:
        t1 = t0 + 0.2
    return (f"Dialogue: 0,{format_ass_time(t0)},{format_ass_time(t1)},"
            f"Default,,0,0,0,,{text}")


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield i, seq[i:i + n]


def _phrase_groups(words: list[Word], max_words: int = 4):
    groups, cur, i0 = [], [], 0
    for i, w in enumerate(words):
        cur.append((i, w))
        if len(cur) >= max_words or w.text.rstrip()[-1:] in ".,!?।":
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _karaoke_run(words, t0, t1, style, ws, base=0):
    """\\k-tagged text for one event over [t0,t1] (per-word overrides applied)."""
    def cs(t):
        return round((t - t0) * 100)

    total = max(cs(t1), 0)
    starts = [min(max(cs(w.start), 0), total) for w in words]
    for i in range(1, len(starts)):
        starts[i] = max(starts[i], starts[i - 1])
    bounds = starts + [total]
    out = []
    if starts and starts[0] > 0:
        out.append(f"{{\\k{starts[0]}}}")
    for i, w in enumerate(words):
        sep = " " if i < len(words) - 1 else ""
        out.append(f"{{\\k{bounds[i + 1] - bounds[i]}}}{_wtext(w, base + i, ws, style)}{sep}")
    return "".join(out)


def _box_bord(style: AssStyle) -> int:
    return max(style.outline, round(style.font_size * 0.3))


# --- per-style renderers (each returns a list of Dialogue lines) -------------

def _render_karaoke(words, seg, style):
    return [_event(seg.start, seg.end,
                   _karaoke_run(words, seg.start, seg.end, style, seg.word_styles))]


def _render_mrbeast(words, seg, style):
    palette = style.palette or ("&H0000FFFF", "&H00FFFFFF", "&H0000FF00")
    parts = []
    for i, w in enumerate(words):
        sep = " " if i < len(words) - 1 else ""
        parts.append(f"{{\\c{palette[i % len(palette)]}}}{_wtext(w, i, seg.word_styles, style)}{sep}")
    return [_event(seg.start, seg.end, "{\\fad(60,40)}" + "".join(parts))]


def _render_hormozi(words, seg, style):
    lines = []
    for base, chunk in _chunks(list(words), 2):
        t0, t1 = chunk[0].start, chunk[-1].end
        run = _karaoke_run(chunk, t0, t1, style, seg.word_styles, base)
        lines.append(_event(t0, t1,
            "{\\an5\\fad(40,0)\\t(0,110,\\fscx112\\fscy112)\\t(110,190,\\fscx100\\fscy100)}" + run))
    return lines


def _render_pop(words, seg, style):
    lines = []
    for i, w in enumerate(words):
        t1 = max(w.end, w.start + 0.18)
        lines.append(_event(w.start, t1,
            "{\\an5\\fad(40,0)\\t(0,120,\\fscx122\\fscy122)\\t(120,210,\\fscx100\\fscy100)}"
            + _wtext(w, i, seg.word_styles, style)))
    return lines


def _render_phrase(words, seg, style):
    lines = []
    for grp in _phrase_groups(list(words)):
        t0, t1 = grp[0][1].start, grp[-1][1].end
        txt = " ".join(_wtext(w, i, seg.word_styles, style) for i, w in grp)
        lines.append(_event(t0, t1, "{\\fad(150,120)}" + txt))
    return lines


def _render_typewriter(words, seg, style):
    lines = []
    for i, w in enumerate(words):
        t0 = w.start
        t1 = words[i + 1].start if i + 1 < len(words) else seg.end
        shown = " ".join(_wtext(x, j, seg.word_styles, style)
                         for j, x in enumerate(words[: i + 1]))
        lines.append(_event(t0, t1, shown))
    return lines


def _render_bounce(words, seg, style):
    lines = []
    for i, w in enumerate(words):
        t1 = words[i + 1].start if i + 1 < len(words) else seg.end
        parts = []
        for j in range(i + 1):
            sep = " " if j < i else ""
            txt = _wtext(words[j], j, seg.word_styles, style)
            if j == i:
                txt = "{\\t(0,90,\\fscx118\\fscy118)\\t(90,170,\\fscx100\\fscy100)}" + txt
            parts.append(txt + sep)
        lines.append(_event(w.start, t1, "".join(parts)))
    return lines


def _render_highlighter(words, seg, style):
    """Full line; the active word gets a translucent thick coloured border that
    reads as a marker highlight, moving word to word."""
    lines = []
    bord = _box_bord(style)
    for i, w in enumerate(words):
        t1 = words[i + 1].start if i + 1 < len(words) else seg.end
        parts = []
        for j, x in enumerate(words):
            sep = " " if j < len(words) - 1 else ""
            txt = _wtext(x, j, seg.word_styles, style)
            if j == i:
                txt = f"{{\\3c{style.highlight_color}\\3a&H30&\\bord{bord}}}{txt}{{\\r}}"
            parts.append(txt + sep)
        lines.append(_event(w.start, t1, "".join(parts)))
    return lines


def _render_wordbox(words, seg, style):
    """Full line; the active word sits in a solid colour box with dark text."""
    lines = []
    bord = _box_bord(style)
    for i, w in enumerate(words):
        t1 = words[i + 1].start if i + 1 < len(words) else seg.end
        parts = []
        for j, x in enumerate(words):
            sep = " " if j < len(words) - 1 else ""
            txt = _wtext(x, j, seg.word_styles, style)
            if j == i:
                txt = (f"{{\\1c&H000000&\\3c{style.highlight_color}\\3a&H00&"
                       f"\\bord{bord}}}{txt}{{\\r}}")
            parts.append(txt + sep)
        lines.append(_event(w.start, t1, "".join(parts)))
    return lines


_RENDERERS = {
    "karaoke": _render_karaoke,
    "boxed": _render_karaoke,       # box from the Style line (border_style 3)
    "mrbeast": _render_mrbeast,
    "hormozi": _render_hormozi,
    "pop": _render_pop,
    "phrase": _render_phrase,
    "typewriter": _render_typewriter,
    "bounce": _render_bounce,
    "highlighter": _render_highlighter,
    "wordbox": _render_wordbox,
}


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
    render = _RENDERERS.get(style.kind, _render_karaoke)
    lines: list[str] = []
    for seg in t.segments:
        words = _timed(seg)
        if not words:
            continue
        lines.extend(render(words, seg, style))
    # custom drag position: pin every event at (pos_x,pos_y), centred on it
    if style.pos_x >= 0 and style.pos_y >= 0:
        x = round(style.pos_x * style.play_res_x)
        y = round(style.pos_y * style.play_res_y)
        prefix = f"{{\\an5\\pos({x},{y})}}"
        lines = [l.replace(",Default,,0,0,0,,", ",Default,,0,0,0,," + prefix, 1)
                 for l in lines]
    return header + "\n".join(lines) + "\n"


# --- Named style presets (10 structurally distinct looks) --------------------
PRESETS: dict[str, AssStyle] = {
    "karaoke": AssStyle(
        kind="karaoke", font_name="Arial", font_size=72,
        primary_color="&H00FFFFFF", highlight_color="&H0000E5FF", outline=5,
    ),
    "highlighter": AssStyle(
        kind="highlighter", font_name="Arial", font_size=74,
        primary_color="&H00FFFFFF", highlight_color="&H0000E5FF", outline=5,
    ),
    "wordbox": AssStyle(
        kind="wordbox", font_name="Arial Black", font_size=72, uppercase=True,
        primary_color="&H00FFFFFF", highlight_color="&H0000FF00", outline=4,
    ),
    "hormozi": AssStyle(
        kind="hormozi", font_name="Arial Black", font_size=120,
        primary_color="&H00FFFFFF", highlight_color="&H0000FFFF",
        outline=9, shadow=2, alignment=5, uppercase=True,
    ),
    "pop": AssStyle(
        kind="pop", font_name="Arial Black", font_size=110,
        primary_color="&H00FFFFFF", highlight_color="&H00FFFFFF",
        outline=8, shadow=2, alignment=5,
    ),
    "bounce": AssStyle(
        kind="bounce", font_name="Arial Black", font_size=84,
        primary_color="&H00FFFFFF", highlight_color="&H00FFFFFF",
        outline=7, shadow=2,
    ),
    "mrbeast": AssStyle(
        kind="mrbeast", font_name="Arial Black", font_size=88,
        primary_color="&H00FFFFFF", highlight_color="&H0000FFFF",
        outline=8, shadow=2, uppercase=True,
        palette=("&H0000FFFF", "&H00FFFFFF", "&H0000FF00", "&H00FF8A00"),
    ),
    "boxed": AssStyle(
        kind="boxed", font_name="Arial", font_size=66,
        primary_color="&H00FFFFFF", highlight_color="&H0000FFFF",
        border_style=3, back_color="&HA0000000", outline=6, shadow=0,
    ),
    "phrase": AssStyle(
        kind="phrase", font_name="Arial", font_size=74,
        primary_color="&H00FFFFFF", highlight_color="&H00FFFFFF",
        outline=5, shadow=1,
    ),
    "typewriter": AssStyle(
        kind="typewriter", font_name="Consolas", font_size=64,
        primary_color="&H00FFFFFF", highlight_color="&H00FFFFFF",
        outline=4, shadow=1,
    ),
}

DEFAULT_PRESET = "karaoke"


def preset_names() -> list[str]:
    return list(PRESETS.keys())


def get_style(name: str) -> AssStyle:
    return PRESETS.get(name, PRESETS[DEFAULT_PRESET])
