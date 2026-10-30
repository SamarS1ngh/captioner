"""Data model shared by the pipeline and the web layer.

Segment.text is the romanized line (the owner's edit unit); Segment.words
carries the original per-word time spans.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return {"text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict) -> Word:
        return cls(text=data["text"], start=data["start"], end=data["end"])


@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    end: float
    words: list[Word]
    # per-word overrides keyed by token index: {i: {"font": str, "size": int}}
    word_styles: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [w.to_dict() for w in self.words],
            "word_styles": self.word_styles,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Segment:
        raw = data.get("word_styles") or {}
        # JSON object keys arrive as strings; token indices are ints
        word_styles = {int(k): v for k, v in raw.items()}
        return cls(
            text=data["text"],
            start=data["start"],
            end=data["end"],
            words=[Word.from_dict(w) for w in data["words"]],
            word_styles=word_styles,
        )


@dataclass(frozen=True)
class Transcript:
    segments: list[Segment]
    language: str

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Transcript:
        return cls(
            segments=[Segment.from_dict(s) for s in data["segments"]],
            language=data["language"],
        )
