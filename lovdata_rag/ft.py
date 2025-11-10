from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from openai import OpenAI

from .config import MODELS_DIR, PROCESSED_DIR
from .logging_utils import get_logger

logger = get_logger("lovdata_rag.ft")

FT_DATASET_PATH = PROCESSED_DIR / "legal_qa_training.jsonl"


def _load_chunks(limit: int = 5000) -> pd.DataFrame:
    chunks_path = PROCESSED_DIR / "lovdata_chunks.parquet"
    if not chunks_path.exists():
        raise FileNotFoundError("Missing chunks parquet. Run chunk pipeline first.")
    df = pd.read_parquet(chunks_path)
    if limit:
        df = df.head(limit)
    return df


def prepare_ft_dataset(limit: int = 5000) -> Path:
    df = _load_chunks(limit=limit)
    records: List[str] = []
    for row in df.itertuples(index=False):
        snippet = row.text.strip()
        if not snippet:
            continue
        context = snippet[:600]
        topic = " ".join(snippet.split()[:12])
        question = f"Hva sier {row.law_name} {row.paragraf or ''} om {topic}?"
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "Du er en norsk jurist. Svar konsist og siter paragraf.",
                },
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": f"{context}\n\n[KILDE: {row.law_name} {row.paragraf or ''}]",
                },
            ],
            "citations": [
                {
                    "law_id": row.law_id,
                    "paragraf": row.paragraf,
                    "ledd": row.ledd,
                    "snippet": context[:280],
                }
            ],
        }
        records.append(json.dumps(payload, ensure_ascii=False))

    FT_DATASET_PATH.write_text("\n".join(records), encoding="utf-8")
    stats = {"total_examples": len(records), "limit": limit}
    stats_path = MODELS_DIR / "chat_training_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Wrote %s fine-tune examples to %s", len(records), FT_DATASET_PATH)
    return FT_DATASET_PATH


def start_fine_tune(training_file: Path, model_name: str) -> str:
    client = OpenAI()
    with open(training_file, "rb") as fh:
        file_obj = client.files.create(file=fh, purpose="fine-tune")
    job = client.fine_tuning.jobs.create(training_file=file_obj.id, model=model_name)
    logger.info("Started fine-tune job %s using %s", job.id, model_name)
    return job.id


def resolve_active_model(default_model: str, fallback_model: str = "gpt-4o-mini") -> str:
    ft_model = os.getenv("FINE_TUNED_MODEL")
    if ft_model:
        logger.info("Using fine-tuned model %s", ft_model)
        return ft_model
    env_model = os.getenv("MODEL_NAME")
    if env_model:
        return env_model
    return default_model or fallback_model
