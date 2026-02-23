from __future__ import annotations

import re
from dataclasses import dataclass

from langdetect import detect, LangDetectException

FILLER_WORDS = {
    "um", "uh", "like", "you know", "i mean", "sort of", "kind of"
}


@dataclass(slots=True)
class NlpPipeline:
    _filler_pattern: re.Pattern[str] = re.compile(
        r"\b(" + "|".join(re.escape(word) for word in sorted(FILLER_WORDS, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )

    _sentence_pattern: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+|(?:\n+)")

    def clean_text(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", text.strip())
        no_fillers = self._filler_pattern.sub("", compact)
        return re.sub(r"\s+", " ", no_fillers).strip()

    def normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def detect_language(self, text: str, default: str = "en") -> str:
        if len(text.strip()) < 4:
            return default
        try:
            return detect(text)
        except LangDetectException:
            return default

    def split_sentences(self, text: str) -> list[str]:
        sentences = [chunk.strip() for chunk in self._sentence_pattern.split(text) if chunk.strip()]
        return sentences

    def extract_finalizable(self, text: str) -> list[str]:
        sentences = self.split_sentences(text)
        if text.endswith((".", "!", "?")):
            return sentences
        return sentences[:-1]

    def finalize_sentence(self, text: str, punctuate_missing: bool) -> str:
        sentence = self.normalize_text(text)
        if punctuate_missing and sentence and not sentence.endswith((".", "!", "?")):
            sentence = f"{sentence}."
        return sentence
