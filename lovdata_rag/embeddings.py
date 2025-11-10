from __future__ import annotations

import json
from typing import Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import CHUNKS_PATH, EMBEDDINGS_PATH, INDEX_PATH, MODELS_DIR
from .logging_utils import get_logger

logger = get_logger("lovdata_rag.embeddings")

EMBED_MODEL = "intfloat/multilingual-e5-large"


def _load_chunks() -> pd.DataFrame:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunk file missing at {CHUNKS_PATH}. Run scripts/chunk_pipeline.py.")
    return pd.read_parquet(CHUNKS_PATH)


def _load_model() -> SentenceTransformer:
    logger.info("Loading embedding model %s", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def build_embeddings(force: bool = False) -> np.ndarray:
    if EMBEDDINGS_PATH.exists() and not force:
        logger.info("Reusing embeddings from %s", EMBEDDINGS_PATH)
        return np.load(EMBEDDINGS_PATH)

    chunk_df = _load_chunks()
    if chunk_df.empty:
        raise RuntimeError("Chunk dataframe is empty. Cannot build embeddings.")

    model = _load_model()
    texts = [f"passage: {text}" for text in chunk_df["text"].tolist()]
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    np.save(EMBEDDINGS_PATH, embeddings)
    stats = {"total_embeddings": int(embeddings.shape[0]), "dimension": int(embeddings.shape[1])}
    stats_path = MODELS_DIR / "embedding_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Saved embeddings to %s", EMBEDDINGS_PATH)
    return embeddings


def build_faiss_index(force: bool = False) -> faiss.IndexHNSWFlat:
    if INDEX_PATH.exists() and not force:
        logger.info("Loading FAISS index from %s", INDEX_PATH)
        return faiss.read_index(str(INDEX_PATH))

    embeddings = build_embeddings(force=force)
    if embeddings.size == 0:
        raise RuntimeError("Embeddings array empty. Cannot build index.")

    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 120
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    logger.info("Saved HNSW index with %s vectors to %s", index.ntotal, INDEX_PATH)
    return index


def ensure_embeddings(force: bool = False) -> Tuple[pd.DataFrame, np.ndarray, faiss.IndexHNSWFlat]:
    chunk_df = pd.read_parquet(CHUNKS_PATH) if CHUNKS_PATH.exists() else None
    if chunk_df is None or chunk_df.empty or force:
        chunk_df = None
    if chunk_df is None:
        raise RuntimeError("Chunks must be built before embeddings.")
    embeddings = build_embeddings(force=force)
    index = build_faiss_index(force=force)
    return chunk_df, embeddings, index
