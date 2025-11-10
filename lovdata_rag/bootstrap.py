from __future__ import annotations

from pathlib import Path

from . import config
from .chunking import build_chunks
from .data_pipeline import build_corpus
from .embeddings import build_embeddings, build_faiss_index
from .logging_utils import get_logger

logger = get_logger("lovdata_rag.bootstrap")


def ensure_assets_ready(force: bool = False) -> None:
    if force:
        logger.info("Force rebuilding all assets")
    _ensure_path(config.CORPUS_PATH, build_corpus, force=force)
    _ensure_path(config.CHUNKS_PATH, build_chunks, force=force)
    _ensure_path(config.EMBEDDINGS_PATH, build_embeddings, force=force)
    _ensure_path(config.INDEX_PATH, build_faiss_index, force=force)


def _ensure_path(path: Path, builder, force: bool = False) -> None:
    if not path.exists() or force:
        logger.info("Building asset %s", path.name)
        builder(force=force)
    else:
        logger.info("Asset %s already present", path.name)
