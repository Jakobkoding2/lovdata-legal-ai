from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÆØÅÄÖÜ])")
PARAGRAPH_RE = re.compile(r"(?i)(?=§\s*\d+)")
LEDD_RE = re.compile(r"(?i)(?=\b\d+\.\s*ledd\b)")


def _split_with_pattern(text: str, pattern: re.Pattern) -> List[Tuple[str, int, int]]:
    if not text:
        return []
    matches = list(pattern.finditer(text))
    if not matches:
        stripped = text.strip()
        if not stripped:
            return []
        start = text.find(stripped)
        return [(stripped, start, start + len(stripped))]
    segments: List[Tuple[str, int, int]] = []
    last = 0
    for match in matches:
        idx = match.start()
        if idx > last:
            raw_segment = text[last:idx]
            segment = raw_segment.strip()
            if segment:
                local_offset = raw_segment.find(segment)
                segment_start = last + local_offset
                segment_end = segment_start + len(segment)
                segments.append((segment, segment_start, segment_end))
        last = idx
    tail_raw = text[last:]
    tail = tail_raw.strip()
    if tail:
        local_offset = tail_raw.find(tail)
        tail_start = last + local_offset
        segments.append((tail, tail_start, tail_start + len(tail)))
    return segments


def split_sections(text: str) -> List[Tuple[str, int, int]]:
    return _split_with_pattern(text, PARAGRAPH_RE)


def split_ledd(text: str) -> List[Tuple[str, int, int]]:
    return _split_with_pattern(text, LEDD_RE)


def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    if not text:
        return []
    sentences: List[Tuple[str, int, int]] = []
    cursor = 0
    for match in SENTENCE_END_RE.finditer(text):
        end = match.start()
        sentence = text[cursor:end].strip()
        if sentence:
            local_offset = text[cursor:end].find(sentence)
            start_idx = cursor + local_offset
            sentences.append((sentence, start_idx, start_idx + len(sentence)))
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        local_offset = text[cursor:].find(tail)
        start_idx = cursor + local_offset
        sentences.append((tail, start_idx, start_idx + len(tail)))
    return sentences


def sliding_windows(sequence: Sequence[str], size: int, overlap: int) -> Iterable[Tuple[int, int]]:
    if not sequence:
        return []
    step = max(1, size - overlap)
    start = 0
    length = len(sequence)
    while start < length:
        end = min(length, start + size)
        yield start, end
        if end == length:
            break
        start += step
