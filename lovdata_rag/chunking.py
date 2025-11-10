from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import pandas as pd
from transformers import AutoTokenizer

from .config import CHUNKS_PATH, CORPUS_PATH, MODELS_DIR, PROCESSED_DIR
from .logging_utils import get_logger
from .text_utils import split_ledd, split_sections, split_sentences

logger = get_logger("lovdata_rag.chunking")

CHUNK_MODEL = "intfloat/multilingual-e5-large"


@dataclass
class ChunkRecord:
    chunk_id: str
    law_id: str
    law_name: str
    kapittel: Optional[str]
    paragraf: Optional[str]
    ledd: Optional[str]
    date: Optional[str]
    source_url: str
    group: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    doc_id: str
    doc_title: str
    section_num: Optional[str]
    section_title: Optional[str]


def _load_corpus() -> pd.DataFrame:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus missing at {CORPUS_PATH}. Run scripts/data_pipeline.py first.")
    return pd.read_parquet(CORPUS_PATH)


def _build_tokenizer():
    logger.info("Loading tokenizer %s", CHUNK_MODEL)
    return AutoTokenizer.from_pretrained(CHUNK_MODEL)


def _chunk_sentences(sentences, tokenizer, chunk_size: int, overlap: int):
    tokens: List[int] = []
    buffered: List[Dict] = []
    for sentence, start, end in sentences:
        encoded = tokenizer(sentence, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        token_ids = encoded["input_ids"]
        buffered.append({"text": sentence, "start": start, "end": end, "tokens": token_ids})
        tokens.extend(token_ids)

    if not buffered:
        return []

    chunks = []
    idx = 0
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size
    while idx < len(tokens):
        window_tokens = tokens[idx : min(len(tokens), idx + chunk_size)]
        window_start_token = idx
        window_end_token = window_start_token + len(window_tokens)
        char_start = None
        char_end = None
        text_parts: List[str] = []
        consumed = 0
        for entry in buffered:
            entry_len = len(entry["tokens"])
            entry_start = consumed
            entry_end = consumed + entry_len
            consumed = entry_end
            intersects = entry_end > window_start_token and entry_start < window_end_token
            if intersects:
                if char_start is None or entry["start"] < char_start:
                    char_start = entry["start"]
                if char_end is None or entry["end"] > char_end:
                    char_end = entry["end"]
                text_parts.append(entry["text"])
        if text_parts and char_start is not None and char_end is not None:
            chunk_text = " ".join(text_parts).strip()
            chunks.append(
                {
                    "text": chunk_text,
                    "start": char_start,
                    "end": char_end,
                    "token_count": len(window_tokens),
                }
            )
        if window_end_token >= len(tokens):
            break
        idx += step
    return chunks


def build_chunks(chunk_size: int = 350, overlap: int = 60, force: bool = False) -> pd.DataFrame:
    if CHUNKS_PATH.exists() and not force:
        logger.info("Chunks already exist at %s", CHUNKS_PATH)
        return pd.read_parquet(CHUNKS_PATH)

    df = _load_corpus()
    tokenizer = _build_tokenizer()

    records: List[Dict] = []
    for row in df.itertuples(index=False):
        body = row.text or ""
        sections = split_sections(body)
        if not sections:
            stripped = body.strip()
            if not stripped:
                continue
            start_idx = body.find(stripped)
            sections = [(stripped, start_idx, start_idx + len(stripped))]
        for section_text, section_start, section_end in sections:
            section_slice = body[section_start:section_end]
            ledd_segments = split_ledd(section_slice)
            if not ledd_segments:
                stripped = section_slice.strip()
                local_start = section_slice.find(stripped) if stripped else 0
                ledd_segments = [(stripped, local_start, local_start + len(stripped))]
            for ledd_text, ledd_start_local, ledd_end_local in ledd_segments:
                global_ledd_start = section_start + ledd_start_local
                global_ledd_end = section_start + ledd_end_local
                ledd_slice = body[global_ledd_start:global_ledd_end]
                sentences = split_sentences(ledd_slice)
                if not sentences:
                    stripped = ledd_slice.strip()
                    if not stripped:
                        continue
                    local_start = ledd_slice.find(stripped)
                    sentences = [(stripped, local_start, local_start + len(stripped))]
                sentence_spans = [
                    (
                        sentence_text,
                        global_ledd_start + sentence_start,
                        global_ledd_start + sentence_end,
                    )
                    for sentence_text, sentence_start, sentence_end in sentences
                    if sentence_text.strip()
                ]
                sentence_chunks = _chunk_sentences(sentence_spans, tokenizer, chunk_size, overlap)
                if not sentence_chunks:
                    continue
                for idx, chunk in enumerate(sentence_chunks, start=1):
                    safe_paragraf = (row.paragraf or "p").replace(" ", "_")
                    safe_ledd = (row.ledd or "l").replace(" ", "_")
                    chunk_id = f"{row.law_id}_{safe_paragraf}_{safe_ledd}_{chunk['start']}_{idx}"
                    records.append(
                        asdict(
                            ChunkRecord(
                                chunk_id=chunk_id,
                                law_id=row.law_id,
                                law_name=row.law_name,
                                kapittel=row.kapittel,
                                paragraf=row.paragraf,
                                ledd=row.ledd,
                                date=row.date,
                                source_url=row.source_url,
                                group=row.group,
                                text=chunk["text"],
                                start_char=int(chunk["start"]),
                                end_char=int(chunk["end"]),
                                token_count=int(chunk["token_count"]),
                                doc_id=row.law_id,
                                doc_title=getattr(row, "doc_title", row.law_name),
                                section_num=row.paragraf,
                                section_title=row.ledd,
                            )
                        )
                    )

    if not records:
        raise RuntimeError("Chunking produced zero records. Check corpus content.")

    chunk_df = pd.DataFrame(records)
    chunk_df.to_parquet(CHUNKS_PATH, index=False)

    stats = {
        "total_chunks": len(chunk_df),
        "avg_tokens": float(chunk_df["token_count"].mean()),
        "max_tokens": int(chunk_df["token_count"].max()),
    }
    stats_path = PROCESSED_DIR / "lovdata_chunks_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Wrote %s chunks to %s", len(chunk_df), CHUNKS_PATH)
    return chunk_df
