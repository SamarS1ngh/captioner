"""Data model shared by the pipeline and the web layer.

Segment.text is the romanized line (the owner's edit unit); Segment.words
carries the original per-word time spans.
"""
from __future__ import annotations

from dataclasses import dataclass


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

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Segment:
        return cls(
            text=data["text"],
            start=data["start"],
            end=data["end"],
            words=[Word.from_dict(w) for w in data["words"]],
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
