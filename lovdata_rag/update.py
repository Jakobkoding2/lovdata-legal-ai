from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict

from .chunking import build_chunks
from .config import PROCESSED_DIR, RAW_DIR
from .data_pipeline import LovdataFetcher, build_corpus
from .embeddings import build_embeddings, build_faiss_index
from .logging_utils import get_logger

logger = get_logger("lovdata_rag.update")

MANIFEST_PATH = PROCESSED_DIR / "update_manifest.json"


def _file_hash(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> Dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _save_manifest(data: Dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_from_api(force: bool = False) -> bool:
    fetcher = LovdataFetcher(RAW_DIR)
    archives = fetcher.fetch_all(force=force)
    manifest = _load_manifest()
    changed = force or not manifest

    for label, path in archives.items():
        file_hash = _file_hash(path)
        if manifest.get(label, {}).get("hash") != file_hash:
            changed = True
        manifest[label] = {"path": str(path), "hash": file_hash, "size": path.stat().st_size}

    if changed:
        logger.info("Detected changes in Lovdata datasets. Rebuilding pipeline.")
        build_corpus(force=True)
        build_chunks(force=True)
        build_embeddings(force=True)
        build_faiss_index(force=True)
    else:
        logger.info("No changes detected in Lovdata datasets.")

    manifest["last_checked"] = datetime.now(timezone.utc).isoformat()
    manifest["updated"] = changed
    _save_manifest(manifest)
    return changed
