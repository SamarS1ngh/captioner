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


def test_presets_registry_has_distinct_kinds():
    names = preset_names()
    assert names and names[0] == "karaoke"
    assert len(PRESETS) == 10
    kinds = {s.kind for s in PRESETS.values()}
    assert kinds == {"karaoke", "highlighter", "wordbox", "hormozi", "pop",
                     "bounce", "mrbeast", "boxed", "phrase", "typewriter"}
    assert all(isinstance(s, AssStyle) for s in PRESETS.values())


def test_highlighter_marker_moves_word_to_word():
    t = Transcript([Segment("ek do teen", 0.0, 1.5, [
        Word("ek", 0.0, 0.5), Word("do", 0.5, 1.0), Word("teen", 1.0, 1.5)])], "hi")
    dl = [l for l in build_ass(t, get_style("highlighter")).splitlines()
          if l.startswith("Dialogue:")]
    assert len(dl) == 3                 # one event per word (marker position)
    assert all("\\3a&H30&" in d for d in dl)  # translucent marker border


def test_per_word_font_and_size_override():
    seg = Segment("main office", 0.0, 1.0,
                  [Word("main", 0.0, 0.5), Word("office", 0.5, 1.0)],
                  word_styles={1: {"font": "Impact", "size": 120}})
    out = build_ass(Transcript([seg], "hi"), get_style("karaoke"))
    assert "\\fnImpact" in out and "\\fs120" in out
    # the override restores to the style's own font/size afterwards
    assert "\\fnArial\\fs72" in out


def test_get_style_unknown_falls_back_to_default():
    assert get_style("does-not-exist") is PRESETS["karaoke"]


def test_build_ass_uses_preset_font_and_size():
    out = build_ass(_one_seg_transcript(), get_style("mrbeast"))
    assert "Arial Black" in out
    style_line = next(l for l in out.splitlines() if l.startswith("Style:"))
    assert style_line.split(",")[2] == "88"  # mrbeast font size


def test_boxed_preset_sets_border_style_3():
    out = build_ass(_one_seg_transcript(), get_style("boxed"))
    style_line = next(l for l in out.splitlines() if l.startswith("Style:"))
    fields = style_line.split(",")
    assert fields[15] == "3"  # BorderStyle = opaque box


def test_hormozi_chunks_into_separate_events_centered_uppercase():
    # 4 words -> 2 chunks of 2 -> 2 dialogue events, centered (\an5), caps
    t = Transcript([Segment("main office ja raha", 0.0, 2.0, [
        Word("main", 0.0, 0.5), Word("office", 0.5, 1.0),
        Word("ja", 1.0, 1.5), Word("raha", 1.5, 2.0)])], "hi")
    dl = [l for l in build_ass(t, get_style("hormozi")).splitlines()
          if l.startswith("Dialogue:")]
    assert len(dl) == 2
    assert all("\\an5" in d for d in dl)
    assert "MAIN" in dl[0] and "OFFICE" in dl[0]


def test_pop_one_event_per_word():
    t = Transcript([Segment("ek do teen", 0.0, 1.5, [
        Word("ek", 0.0, 0.5), Word("do", 0.5, 1.0), Word("teen", 1.0, 1.5)])], "hi")
    dl = [l for l in build_ass(t, get_style("pop")).splitlines()
          if l.startswith("Dialogue:")]
    assert len(dl) == 3  # one word per event
    assert all("\\fscx" in d for d in dl)  # scale pop


def test_typewriter_accumulates_words():
    t = Transcript([Segment("ek do teen", 0.0, 1.5, [
        Word("ek", 0.0, 0.5), Word("do", 0.5, 1.0), Word("teen", 1.0, 1.5)])], "hi")
    dl = [l for l in build_ass(t, get_style("typewriter")).splitlines()
          if l.startswith("Dialogue:")]
    assert len(dl) == 3
    # last event shows the whole line
    assert "ek do teen" in dl[-1]
    assert dl[0].rstrip().endswith("ek")  # first shows only first word


def test_mrbeast_colors_each_word():
    out = build_ass(_one_seg_transcript(), get_style("mrbeast"))
    dl = [l for l in out.splitlines() if l.startswith("Dialogue:")][0]
    assert dl.count("\\c&H") >= 2  # a colour override per word


def test_custom_drag_position_pins_events():
    from dataclasses import replace
    st = replace(get_style("karaoke"), pos_x=0.5, pos_y=0.25)
    out = build_ass(_one_seg_transcript(), st)
    assert "\\pos(540,480)" in out  # 1080*0.5, 1920*0.25
    assert out.count("\\pos(") == len([l for l in out.splitlines()
                                       if l.startswith("Dialogue:")])
