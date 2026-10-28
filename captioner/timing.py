"""Map an owner-edited caption line back onto per-word timestamps.

The owner edits the romanized line, so its token count can differ from the
original word spans. When counts match, spans map 1:1. Otherwise the line's
[start, end] is split across tokens in proportion to their character length.
"""
from __future__ import annotations

from captioner.models import Word

# Minimum weight per token so empty or one-character tokens still get a slice.
_MIN_TOKEN_WEIGHT = 1


def distribute_timing(
    start: float,
    end: float,
    tokens: list[str],
    word_spans: list[tuple[float, float]],
) -> list[Word]:
    if not tokens:
        return []

    if end < start:
        end = start

    if len(tokens) == len(word_spans):
        return [Word(text, s, e) for text, (s, e) in zip(tokens, word_spans)]

    weights = [max(len(token), _MIN_TOKEN_WEIGHT) for token in tokens]
    total = sum(weights)
    duration = end - start

    words: list[Word] = []
    cumulative = 0
    for index, (token, weight) in enumerate(zip(tokens, weights)):
        slice_start = start + duration * cumulative / total
        cumulative += weight
        slice_end = start + duration * cumulative / total
        if index == 0:
            slice_start = start
        if index == len(tokens) - 1:
            slice_end = end
        words.append(Word(token, slice_start, slice_end))
    return words
