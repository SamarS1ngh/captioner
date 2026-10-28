from captioner.models import Word
from captioner.timing import distribute_timing


def test_one_to_one_uses_word_spans():
    tokens = ["aap", "kaise", "ho"]
    spans = [(0.0, 0.5), (0.5, 1.2), (1.2, 2.0)]

    result = distribute_timing(0.0, 2.0, tokens, spans)

    assert result == [
        Word("aap", 0.0, 0.5),
        Word("kaise", 0.5, 1.2),
        Word("ho", 1.2, 2.0),
    ]


def test_more_tokens_than_spans_distributes_proportionally():
    tokens = ["aap", "kaise", "ho", "bhai"]
    spans = [(10.0, 11.0), (11.0, 12.0)]

    result = distribute_timing(10.0, 12.0, tokens, spans)

    assert len(result) == 4
    assert [w.text for w in result] == tokens
    assert result[0].start == 10.0
    assert result[-1].end == 12.0
    for prev, cur in zip(result, result[1:]):
        assert prev.end <= cur.start
    for w in result:
        assert w.start < w.end


def test_fewer_tokens_than_spans():
    result = distribute_timing(5.0, 8.0, ["namaste"], [(5.0, 6.0), (6.0, 7.0), (7.0, 8.0)])

    assert result == [Word("namaste", 5.0, 8.0)]


def test_empty_tokens_returns_empty():
    assert distribute_timing(0.0, 1.0, [], []) == []
    assert distribute_timing(0.0, 1.0, [], [(0.0, 1.0)]) == []
