from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    from api.bm25 import BM25Okapi

try:
    from cross_encoder import CrossEncoder
except ImportError:  # pragma: no cover
    from sentence_transformers import CrossEncoder  # type: ignore

from lovdata_rag.config import BASE_DIR, CHUNKS_PATH, CORPUS_PATH, EMBEDDINGS_PATH, INDEX_PATH
from lovdata_rag.chunking import build_chunks
from lovdata_rag.embeddings import build_embeddings, build_faiss_index
from lovdata_rag.logging_utils import get_logger

logger = get_logger("lovdata_rag.pipeline")


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    doc_title: str
    section_num: Optional[str]
    section_title: Optional[str]
    group: str
    text: str
    score: float = 0.0
    date: Optional[str] = None
    source_url: Optional[str] = None
    kapittel: Optional[str] = None
    ledd: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class CodexRAGPipeline:
    def __init__(
        self,
        base_dir=BASE_DIR,
        dense_weight: float = 0.8,
        bm25_weight: float = 0.2,
        rerank_top_k: int = 50,
        rerank_keep: int = 5,
    ) -> None:
        self.base_dir = base_dir
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rerank_top_k = rerank_top_k
        self.rerank_keep = rerank_keep

        self.embedding_model = self._load_embedding_model()
        self.embedding_dimension = (
            self.embedding_model.get_sentence_embedding_dimension() if self.embedding_model else 0
        )
        self.cross_encoder = self._load_cross_encoder()

        self.corpus_df = self._load_corpus()
        self.chunk_df = self._load_chunks()
        self.embeddings = self._load_embeddings()
        self.faiss_index = self._load_faiss()
        self.bm25 = self._build_bm25()

        if self.faiss_index is not None and hasattr(self.faiss_index, "hnsw"):
            self.faiss_index.hnsw.efSearch = 120

    def _load_embedding_model(self) -> Optional[SentenceTransformer]:
        try:
            return SentenceTransformer("intfloat/multilingual-e5-large")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load embedding model: %s", exc)
            return None

    def _load_cross_encoder(self) -> Optional[CrossEncoder]:
        try:
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load cross-encoder: %s", exc)
            return None

    def _load_corpus(self) -> pd.DataFrame:
        if CORPUS_PATH.exists():
            return pd.read_parquet(CORPUS_PATH)
        logger.warning("Corpus file missing at %s", CORPUS_PATH)
        return pd.DataFrame()

    def _load_chunks(self) -> pd.DataFrame:
        if not CHUNKS_PATH.exists():
            logger.info("Chunk file missing. Building chunks.")
            build_chunks()
        return pd.read_parquet(CHUNKS_PATH)

    def _load_embeddings(self) -> np.ndarray:
        if not EMBEDDINGS_PATH.exists():
            logger.info("Embedding file missing. Building embeddings.")
            build_embeddings()
        return np.load(EMBEDDINGS_PATH)

    def _load_faiss(self):
        if not INDEX_PATH.exists():
            logger.info("FAISS index missing. Building index.")
            build_faiss_index()
        return faiss.read_index(str(INDEX_PATH))

    def _build_bm25(self) -> Optional[BM25Okapi]:
        if self.chunk_df.empty:
            return None
        tokenized_corpus = [self._tokenize(text) for text in self.chunk_df["text"].tolist()]
        return BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _encode_query(self, query: str) -> np.ndarray:
        if self.embedding_model is None:
            return np.zeros((1, self.embedding_dimension), dtype=np.float32)
        embedding = self.embedding_model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        return embedding

    def search(self, query: str, top_k: int = 10) -> List[ChunkRecord]:
        candidates = self._hybrid_candidates(query, max(top_k, self.rerank_top_k))
        return candidates[:top_k]

    def _hybrid_candidates(self, query: str, top_k: int) -> List[ChunkRecord]:
        if self.faiss_index is None or self.chunk_df.empty:
            return []

        dense_scores = self._dense_scores(query, top_k)
        bm25_scores = self._bm25_scores(query)

        combined: Dict[int, float] = {}
        for idx, score in dense_scores.items():
            combined[idx] = combined.get(idx, 0.0) + self.dense_weight * score
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            for idx, score in bm25_scores.items():
                norm = (score / max_bm25) if max_bm25 > 0 else 0.0
                combined[idx] = combined.get(idx, 0.0) + self.bm25_weight * norm

        sorted_indices = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        records: List[ChunkRecord] = []
        for idx, score in sorted_indices[:top_k]:
            row = self.chunk_df.iloc[idx]
            records.append(
                ChunkRecord(
                    chunk_id=row.get("chunk_id", str(idx)),
                    doc_id=row.get("doc_id", ""),
                    doc_title=row.get("doc_title", "") or row.get("law_name", ""),
                    section_num=row.get("section_num") or row.get("paragraf"),
                    section_title=row.get("section_title") or row.get("ledd"),
                    group=row.get("group", ""),
                    text=row.get("text", ""),
                    score=float(score),
                    date=row.get("date"),
                    source_url=row.get("source_url"),
                    kapittel=row.get("kapittel"),
                    ledd=row.get("ledd"),
                    start_char=int(row.get("start_char", 0)),
                    end_char=int(row.get("end_char", 0)),
                )
            )
        return records

    def _dense_scores(self, query: str, top_k: int) -> Dict[int, float]:
        if self.faiss_index is None:
            return {}
        vector = self._encode_query(query)
        requested = max(top_k, self.rerank_top_k)
        distances, indices = self.faiss_index.search(vector, requested)
        scores: Dict[int, float] = {}
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            scores[int(idx)] = float(score)
        return scores

    def _bm25_scores(self, query: str) -> Dict[int, float]:
        if self.bm25 is None:
            return {}
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][: self.rerank_top_k]
        return {int(idx): float(scores[idx]) for idx in top_indices if scores[idx] > 0}

    def rerank(self, query: str, candidates: List[ChunkRecord], top_n: Optional[int] = None) -> List[ChunkRecord]:
        if not candidates:
            return []
        if self.cross_encoder is None:
            return candidates[: top_n or self.rerank_keep]
        top_n = top_n or self.rerank_keep
        pairs = [[query, candidate.text] for candidate in candidates[: self.rerank_top_k]]
        scores = self.cross_encoder.predict(pairs)
        reranked: List[Tuple[ChunkRecord, float]] = []
        for candidate, score in zip(candidates[: len(scores)], scores):
            reranked.append((candidate, float(score)))
        reranked.sort(key=lambda item: item[1], reverse=True)
        results: List[ChunkRecord] = []
        for record, score in reranked[:top_n]:
            record.score = float(score)
            results.append(record)
        return results

    def top_contexts(self, query: str) -> List[ChunkRecord]:
        candidates = self._hybrid_candidates(query, self.rerank_top_k)
        return self.rerank(query, candidates, self.rerank_keep)


def format_sources(records: List[ChunkRecord]) -> List[Dict[str, Optional[str]]]:
    payload = []
    for record in records:
        payload.append(
            {
                "doc_id": record.doc_id,
                "doc_title": record.doc_title,
                "section_num": record.section_num,
                "section_title": record.section_title,
                "group": record.group,
                "score": record.score,
                "text": record.text,
                "date": record.date,
                "source_url": record.source_url,
            }
        )
    return payload


def compute_overlap(
    embedding_model: Optional[SentenceTransformer],
    text1: str,
    text2: str,
) -> float:
    if embedding_model is None:
        return 0.0
    vectors = embedding_model.encode(
        [f"passage: {text1}", f"passage: {text2}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return float(np.dot(vectors[0], vectors[1]))
