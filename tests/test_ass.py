import re

from captioner.ass import AssStyle, build_ass, format_ass_time
from captioner.models import Segment, Transcript, Word


def _t(text="hello world", start=1.0, end=2.5, words=None):
    if words is None:
        words = [Word("hello", start, (start + end) / 2), Word("world", (start + end) / 2, end)]
    return Transcript([Segment(text, start, end, words)], "hi")


def _dialogues(doc):
    return [l for l in doc.splitlines() if l.startswith("Dialogue:")]


def test_ass_has_script_info_and_dialogue():
    t = Transcript([_t().segments[0], _t("bye", 3.0, 4.0, [Word("bye", 3.0, 4.0)]).segments[0]], "hi")
    doc = build_ass(t, AssStyle())
    assert "[Script Info]" in doc and "[V4+ Styles]" in doc and "[Events]" in doc
    assert len(_dialogues(doc)) == 2


def test_dialogue_timecodes_match_segment():
    line = _dialogues(build_ass(_t()))[0]
    assert line.startswith("Dialogue: 0,0:00:01.00,0:00:02.50,")
    assert format_ass_time(3725.5) == "1:02:05.50"


def test_karaoke_durations_sum_to_segment():
    # leading gap, inter-word gap and a leading-space text (Whisper style)
    words = [Word("ek", 1.13, 1.50), Word("do", 1.90, 2.31), Word("teen", 2.31, 2.97)]
    doc = build_ass(_t(" ek do teen", 1.0, 3.0, words))
    line = _dialogues(doc)[0]
    assert "}ek" in line and "} ek" not in line
    total = sum(int(n) for n in re.findall(r"\\k(\d+)", line))
    assert abs(total - 200) <= 1


def test_escapes_brace_characters():
    doc = build_ass(_t("a {b} c\\N d", 0.0, 1.0, []))
    line = _dialogues(doc)[0]
    text = line.split(",", 9)[9]
    # only our own \k overrides may remain as raw tags
    assert not re.search(r"(?<!\\)\{(?!\\k\d+\})", text)
    assert "\\{b\\}" in text


# --- style presets -----------------------------------------------------------
from captioner.ass import PRESETS, preset_names, get_style, AssStyle  # noqa: E402
from captioner.models import Transcript, Segment, Word  # noqa: E402


def _one_seg_transcript():
    return Transcript(
        segments=[Segment(text="main office", start=0.0, end=1.0,
                          words=[Word("main", 0.0, 0.5), Word("office", 0.5, 1.0)])],
        language="hi",
    )


def test_presets_registry_has_classic_first_and_all_assstyle():
    names = preset_names()
    assert names and names[0] == "classic"
    assert all(isinstance(s, AssStyle) for s in PRESETS.values())
    assert len(PRESETS) >= 5


def test_get_style_unknown_falls_back_to_classic():
    assert get_style("does-not-exist") is PRESETS["classic"]


def test_build_ass_uses_preset_font_and_size():
    out = build_ass(_one_seg_transcript(), get_style("bold_pop"))
    assert "Arial Black" in out
    # font size is the 3rd field of the Style line
    style_line = next(l for l in out.splitlines() if l.startswith("Style:"))
    assert style_line.split(",")[2] == "88"


def test_boxed_preset_sets_border_style_3():
    out = build_ass(_one_seg_transcript(), get_style("boxed"))
    style_line = next(l for l in out.splitlines() if l.startswith("Style:"))
    fields = style_line.split(",")
    # fields: 0"Style: Default" 1font 2size 3pri 4sec 5outlineCol 6back 7bold
    # 8italic 9underline 10strike 11scaleX 12scaleY 13spacing 14angle
    # 15BorderStyle 16Outline 17Shadow 18alignment
    assert fields[15] == "3" and fields[16] == "4"  # boxed: border 3, outline 4
