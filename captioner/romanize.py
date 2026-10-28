"""Romanization adapter: Devanagari/English text -> casual Latin (IndicXlit).

The engine is built lazily so importing this module stays cheap and offline-safe.
The first construction downloads the indic2en model (~800MB, one time); after
that no network is used.
"""
from __future__ import annotations

import dataclasses

from captioner.models import Transcript

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        import torch

        # torch >= 2.6 defaults weights_only=True, which breaks fairseq checkpoints.
        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
        from ai4bharat.transliteration import XlitEngine

        _engine = XlitEngine("hi", beam_width=10, rescore=False, src_script_type="indic")
    return _engine


def romanize_transcript(t: Transcript) -> Transcript:
    engine = _get_engine()
    segments = [
        dataclasses.replace(s, text=engine.translit_sentence(s.text, lang_code="hi"))
        for s in t.segments
    ]
    return Transcript(segments=segments, language=t.language)
