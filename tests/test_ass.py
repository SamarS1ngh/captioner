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
