from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / ".cache"

CORPUS_PATH = PROCESSED_DIR / "lovdata_corpus.parquet"
CHUNKS_PATH = PROCESSED_DIR / "lovdata_chunks.parquet"
EMBEDDINGS_PATH = MODELS_DIR / "codex_rag_embeddings.npy"
INDEX_PATH = MODELS_DIR / "codex_rag_hnsw.index"
OVERLAP_MODEL_PATH = MODELS_DIR / "overlap_classifier.joblib"

DEFAULT_LAW_SOURCE_URL = "https://lovdata.no"

for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, CACHE_DIR):
    path.mkdir(parents=True, exist_ok=True)
