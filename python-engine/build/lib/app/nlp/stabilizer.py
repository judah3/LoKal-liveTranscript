from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher


_PUNCTUATION = ".?!"
_FILLER_WORDS = ("um", "uh", "like", "you know", "i mean")
_BRAND_CASE_MAP = {
    "acme cloud": "Acme Cloud",
}


def stabilize_prefix(previous_text: str, new_text: str) -> str:
    """Stabilize noisy streaming text by preserving confirmed prefix regions.

    The function protects against regressions where the decoder temporarily
    drops the beginning of the transcript and only emits a trailing suffix.
    """
    prev = (previous_text or "").strip()
    new = (new_text or "").strip()

    if not prev:
        return new
    if not new:
        return prev

    # Whisper can occasionally backtrack and emit only a trailing slice.
    # Keep the previous text instead of regressing the UI.
    if len(new) < len(prev) and prev.lower().endswith(new.lower()):
        return prev

    lcp = _longest_common_prefix(prev, new)
    if lcp == 0:
        match = SequenceMatcher(None, prev.lower(), new.lower()).find_longest_match(0, len(prev), 0, len(new))
        if match.size >= max(6, int(len(prev) * 0.5)):
            return prev
        return new

    lock_index = _backtrack_to_word_boundary(prev, lcp)
    return prev[:lock_index] + new[lock_index:]


def remove_partial_trailing_word(text: str) -> str:
    """Remove unstable trailing fragments such as `sub-`, `char-`, `Do-`."""
    if not text:
        return ""

    working = text.rstrip()
    if not working:
        return ""

    # Already sentence-complete.
    if working[-1] in _PUNCTUATION:
        return working

    token_match = re.search(r"(\S+)$", working)
    if token_match is None:
        return working

    token = token_match.group(1)
    token_alpha = re.sub(r"[^A-Za-z]", "", token)
    has_vowel = bool(re.search(r"[aeiouAEIOU]", token_alpha))

    should_trim = (
        token.endswith("-")
        or (len(token_alpha) < 3 and not has_vowel)
        or _looks_abrupt(token)
    )

    if not should_trim:
        return working

    return re.sub(r"\s*\S+$", "", working).rstrip()


@dataclass(slots=True)
class SentenceCommitManager:
    """Manages stable committed sentences and a live unstable buffer."""

    committed_sentences: list[str] = field(default_factory=list)
    live_buffer: str = ""

    def update(self, new_text: str, silence: bool = False, vad_end: bool = False) -> dict[str, object]:
        stabilized = stabilize_prefix(self.live_buffer, new_text)
        finalized: list[str] = []

        sentences = _split_sentences_with_remainder(stabilized)
        finalized.extend(sentences["final"])
        remainder = sentences["remainder"]

        if (silence or vad_end) and remainder:
            finalized.append(remainder.strip())
            remainder = ""

        finalized = [item for item in finalized if item.strip()]
        self.committed_sentences.extend(finalized)
        self.live_buffer = remove_partial_trailing_word(remainder)

        return {
            "partial": self.live_buffer,
            "finalized": finalized,
        }


def clean_final_text(text: str) -> str:
    """Apply production cleanup for finalized transcript sentences."""
    normalized = _normalize_spacing(text)
    normalized = _normalize_email_spelling(normalized)
    normalized = _remove_repeated_fillers(normalized)
    normalized = _normalize_brand_casing(normalized)
    normalized = _normalize_sentence_case(normalized)
    normalized = _remove_duplicate_consecutive_sentences(normalized)
    if normalized and normalized[-1] not in _PUNCTUATION:
        normalized = f"{normalized}."
    return normalized.strip()


@dataclass(slots=True)
class StreamingTranscriptionStabilizer:
    """Production-ready stabilizer for streaming Whisper output.

    Example:
        stabilizer = StreamingTranscriptionStabilizer()
        partial = stabilizer.process_partial("Hi, I was charged twi-")
        # {"type": "partial", "text": partial}

        out = stabilizer.process_final("Hi, I was charged twice this month for Acme Cloud.", silence=True)
        # out["finalized_events"] -> [{"type": "final", "text": "..."}]
    """

    commit_manager: SentenceCommitManager = field(default_factory=SentenceCommitManager)
    _last_raw_partial: str = ""
    _last_emitted_final: str = ""
    _pending_finalized: list[str] = field(default_factory=list)

    def process_partial(self, text: str) -> str:
        stabilized = stabilize_prefix(self._last_raw_partial, text)
        self._last_raw_partial = stabilized

        update = self.commit_manager.update(stabilized, silence=False, vad_end=False)
        newly_finalized = self._clean_and_filter(update["finalized"])  # type: ignore[arg-type]
        if newly_finalized:
            self._pending_finalized.extend(newly_finalized)

        return str(update["partial"])

    def consume_pending_finalized(self) -> list[str]:
        if not self._pending_finalized:
            return []
        out = list(self._pending_finalized)
        self._pending_finalized.clear()
        return out

    def process_final(self, text: str, silence: bool = True, vad_end: bool = False) -> dict[str, object]:
        stabilized = stabilize_prefix(self._last_raw_partial, text)
        self._last_raw_partial = stabilized

        update = self.commit_manager.update(stabilized, silence=silence, vad_end=vad_end)
        finalized = self._clean_and_filter(update["finalized"])  # type: ignore[arg-type]
        if self._pending_finalized:
            finalized = [*self._pending_finalized, *finalized]
            self._pending_finalized.clear()

        events = [{"type": "final", "text": sentence} for sentence in finalized]
        return {
            "partial": str(update["partial"]),
            "finalized": finalized,
            "finalized_events": events,
        }

    def _clean_and_filter(self, finalized: list[str]) -> list[str]:
        out: list[str] = []
        for sentence in finalized:
            cleaned = clean_final_text(sentence)
            if not cleaned:
                continue
            if self._is_duplicate(cleaned):
                continue
            self._last_emitted_final = cleaned
            out.append(cleaned)
        return out

    def _is_duplicate(self, candidate: str) -> bool:
        if not self._last_emitted_final:
            return False
        ratio = SequenceMatcher(None, self._last_emitted_final.lower(), candidate.lower()).ratio()
        return ratio >= 0.9


def _longest_common_prefix(a: str, b: str) -> int:
    max_len = min(len(a), len(b))
    idx = 0
    while idx < max_len and a[idx] == b[idx]:
        idx += 1
    return idx


def _backtrack_to_word_boundary(text: str, idx: int) -> int:
    if idx <= 0 or idx >= len(text):
        return idx
    if text[idx - 1].isspace() or text[idx - 1] in ",.;:!?":
        return idx
    boundary = text.rfind(" ", 0, idx)
    return boundary + 1 if boundary >= 0 else idx


def _looks_abrupt(token: str) -> bool:
    if not token:
        return False
    if token[-1] in _PUNCTUATION:
        return False
    if "-" in token:
        return True
    if len(token) <= 2:
        return True
    return token[-1].isalpha()


def _split_sentences_with_remainder(text: str) -> dict[str, list[str] | str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if not parts or parts == [""]:
        return {"final": [], "remainder": ""}

    final: list[str] = []
    for chunk in parts[:-1]:
        trimmed = chunk.strip()
        if trimmed:
            final.append(trimmed)

    last = parts[-1].strip()
    if last.endswith(tuple(_PUNCTUATION)):
        if last:
            final.append(last)
        return {"final": final, "remainder": ""}
    return {"final": final, "remainder": last}


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", text)
    return text


def _normalize_email_spelling(text: str) -> str:
    pattern = re.compile(
        r"\b([a-z0-9]+(?:\s+dot\s+[a-z0-9]+)*)\s+at\s+([a-z0-9]+(?:\s+dot\s+[a-z0-9]+)+)\b",
        flags=re.IGNORECASE,
    )

    def repl(match: re.Match[str]) -> str:
        local = re.sub(r"\s+dot\s+", ".", match.group(1), flags=re.IGNORECASE)
        domain = re.sub(r"\s+dot\s+", ".", match.group(2), flags=re.IGNORECASE)
        return f"{local.lower()}@{domain.lower()}"

    return pattern.sub(repl, text)


def _remove_repeated_fillers(text: str) -> str:
    out = text
    for filler in _FILLER_WORDS:
        escaped = re.escape(filler)
        out = re.sub(
            rf"\b({escaped})(?:\s+\1\b)+",
            r"\1",
            out,
            flags=re.IGNORECASE,
        )
    return out


def _normalize_brand_casing(text: str) -> str:
    out = text
    for lower, proper in _BRAND_CASE_MAP.items():
        out = re.sub(rf"\b{re.escape(lower)}\b", proper, out, flags=re.IGNORECASE)
    return out


def _normalize_sentence_case(text: str) -> str:
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    normalized: list[str] = []
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        normalized.append(s[0].upper() + s[1:] if len(s) > 1 else s.upper())
    return " ".join(normalized)


def _remove_duplicate_consecutive_sentences(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return ""
    out = [sentences[0]]
    for sentence in sentences[1:]:
        ratio = SequenceMatcher(None, out[-1].lower(), sentence.lower()).ratio()
        if ratio >= 0.9:
            continue
        out.append(sentence)
    return " ".join(out)
