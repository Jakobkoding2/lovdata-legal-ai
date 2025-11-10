import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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

from transformers import AutoTokenizer


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


class SimpleTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> List[str]:
        return text.split()

    def decode(self, tokens: List[str], skip_special_tokens: bool = True) -> str:
        return " ".join(tokens)


class CodexRAGPipeline:
    def __init__(
        self,
        base_dir: Path,
        chunk_size: int = 350,
        chunk_overlap: int = 60,
        dense_weight: float = 0.8,
        bm25_weight: float = 0.2,
        rerank_top_k: int = 50,
        rerank_keep: int = 5,
    ) -> None:
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "processed"
        self.models_dir = base_dir / "models"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rerank_top_k = rerank_top_k
        self.rerank_keep = rerank_keep

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = self._load_tokenizer()
        self.embedding_model = self._load_embedding_model()
        self.embedding_dimension = (
            self.embedding_model.get_sentence_embedding_dimension() if self.embedding_model else 0
        )
        self.cross_encoder = self._load_cross_encoder()

        self.corpus_df = self._load_corpus()
        self.chunk_df = self._load_or_build_chunks()
        self.embeddings = self._load_or_build_embeddings()
        self.faiss_index = self._load_or_build_faiss()
        self.bm25 = self._build_bm25_index()

        if self.faiss_index is not None:
            self.faiss_index.hnsw.efSearch = 120

    def _load_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
        except Exception:  # pragma: no cover
            print("WARNING: Could not load multilingual-e5 tokenizer. Falling back to simple tokenizer.")
            return SimpleTokenizer()

    def _load_embedding_model(self) -> Optional[SentenceTransformer]:
        try:
            return SentenceTransformer("intfloat/multilingual-e5-large")
        except Exception:  # pragma: no cover
            print("WARNING: Could not load multilingual-e5 embedding model. Retrieval will be disabled.")
            return None

    def _load_cross_encoder(self) -> Optional[CrossEncoder]:
        try:
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:  # pragma: no cover
            print("WARNING: Could not load cross-encoder reranker. Reranking will be disabled.")
            return None

    def _load_corpus(self) -> pd.DataFrame:
        corpus_path = self.data_dir / "lovdata_corpus.parquet"
        if corpus_path.exists():
            return pd.read_parquet(corpus_path)
        return pd.DataFrame(
            columns=[
                "unit_id",
                "doc_id",
                "doc_title",
                "group",
                "section_num",
                "section_title",
                "text_clean",
            ]
        )

    def _load_or_build_chunks(self) -> pd.DataFrame:
        chunk_path = self.data_dir / "lovdata_chunks.parquet"
        if chunk_path.exists():
            return pd.read_parquet(chunk_path)

        if self.corpus_df.empty:
            return pd.DataFrame(
                columns=[
                    "chunk_id",
                    "doc_id",
                    "doc_title",
                    "section_num",
                    "section_title",
                    "group",
                    "text",
                ]
            )

        records: List[Dict[str, Optional[str]]] = []
        for _, row in self.corpus_df.iterrows():
            text = row.get("text_clean", "") or ""
            doc_id = str(row.get("doc_id", ""))
            doc_title = row.get("doc_title", "") or ""
            section_num = row.get("section_num")
            section_title = row.get("section_title")
            group = row.get("group", "") or ""

            chunks = self._chunk_text(text)
            if not chunks:
                continue

            for chunk_idx, chunk_text in enumerate(chunks, start=1):
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                records.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "section_num": section_num,
                        "section_title": section_title,
                        "group": group,
                        "text": chunk_text,
                    }
                )

        chunk_df = pd.DataFrame(records)
        if not chunk_df.empty:
            chunk_df.to_parquet(chunk_path, index=False)
        return chunk_df

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []

        segments = self._split_legal_sections(text)
        chunks: List[str] = []

        for segment in segments:
            tokens = self.tokenizer.encode(segment, add_special_tokens=False)
            if not tokens:
                continue

            length = len(tokens)
            start = 0
            while start < length:
                end = min(start + self.chunk_size, length)
                chunk_tokens = tokens[start:end]
                if hasattr(self.tokenizer, "decode"):
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
                    if not chunk_text and isinstance(chunk_tokens[0], str):
                        chunk_text = " ".join(chunk_tokens).strip()
                else:
                    chunk_text = " ".join(str(tok) for tok in chunk_tokens).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                if end >= length:
                    break
                start = max(0, end - self.chunk_overlap)

        return chunks

    @staticmethod
    def _split_legal_sections(text: str) -> List[str]:
        pattern = re.compile(r"(?i)(?=§\s*\d|\bkapittel\b|\bledd\b)")
        positions = [match.start() for match in pattern.finditer(text)]
        if not positions:
            return [text]

        segments: List[str] = []
        last_pos = 0
        for pos in positions:
            if pos != last_pos:
                segments.append(text[last_pos:pos].strip())
            last_pos = pos
        segments.append(text[last_pos:].strip())
        return [seg for seg in segments if seg]

    def _load_or_build_embeddings(self) -> np.ndarray:
        embeddings_path = self.models_dir / "codex_rag_embeddings.npy"
        if embeddings_path.exists():
            return np.load(embeddings_path)

        if self.chunk_df.empty or self.embedding_model is None:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)

        texts = [f"passage: {text}" for text in self.chunk_df["text"].tolist()]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        np.save(embeddings_path, embeddings)
        return embeddings

    def _load_or_build_faiss(self):
        if self.embedding_model is None or self.embeddings.size == 0:
            return None

        import faiss

        index_path = self.models_dir / "codex_rag_hnsw.index"
        if index_path.exists():
            return faiss.read_index(str(index_path))

        dimension = self.embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(self.embeddings)
        faiss.write_index(index, str(index_path))
        return index

    def _build_bm25_index(self) -> Optional[BM25Okapi]:
        if self.chunk_df.empty:
            return None
        tokenized_corpus = [self._tokenize_bm25(text) for text in self.chunk_df["text"].tolist()]
        return BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize_bm25(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _encode_query(self, query: str) -> np.ndarray:
        if self.embedding_model is None:
            return np.zeros((1, self.embedding_dimension), dtype=np.float32)
        embedding = self.embedding_model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        return embedding

    def search(self, query: str, top_k: int = 10) -> List[ChunkRecord]:
        candidates = self._hybrid_candidates(query, max(top_k, self.rerank_top_k))
        return candidates[:top_k]

    def _hybrid_candidates(self, query: str, top_k: int) -> List[ChunkRecord]:
        if self.faiss_index is None or self.chunk_df.empty:
            return []

        dense_scores = self._dense_search(query, top_k)
        bm25_scores = self._bm25_scores(query)

        combined_scores: Dict[int, float] = {}
        for idx, score in dense_scores.items():
            combined_scores[idx] = combined_scores.get(idx, 0.0) + self.dense_weight * score

        if bm25_scores:
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 0.0
            for idx, score in bm25_scores.items():
                normalized = (score / max_bm25) if max_bm25 > 0 else 0.0
                combined_scores[idx] = combined_scores.get(idx, 0.0) + self.bm25_weight * normalized

        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        records: List[ChunkRecord] = []
        for idx, score in sorted_indices[:top_k]:
            row = self.chunk_df.iloc[idx]
            records.append(
                ChunkRecord(
                    chunk_id=row.get("chunk_id", str(idx)),
                    doc_id=row.get("doc_id", ""),
                    doc_title=row.get("doc_title", ""),
                    section_num=row.get("section_num"),
                    section_title=row.get("section_title"),
                    group=row.get("group", ""),
                    text=row.get("text", ""),
                    score=float(score),
                )
            )
        return records

    def _dense_search(self, query: str, top_k: int) -> Dict[int, float]:
        if self.faiss_index is None:
            return {}
        query_vector = self._encode_query(query)
        requested = max(top_k, self.rerank_top_k)
        distances, indices = self.faiss_index.search(query_vector, requested)
        scores = {}
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            scores[int(idx)] = float(score)
        return scores

    def _bm25_scores(self, query: str) -> Dict[int, float]:
        if self.bm25 is None:
            return {}
        tokenized_query = self._tokenize_bm25(query)
        scores = self.bm25.get_scores(tokenized_query)
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
        reranked.sort(key=lambda x: x[1], reverse=True)
        results: List[ChunkRecord] = []
        for record, score in reranked[:top_n]:
            record.score = float(score)
            results.append(record)
        return results

    def top_contexts(self, query: str) -> List[ChunkRecord]:
        candidates = self._hybrid_candidates(query, self.rerank_top_k)
        return self.rerank(query, candidates, self.rerank_keep)


def format_sources(records: List[ChunkRecord]) -> List[Dict[str, Optional[str]]]:
    formatted = []
    for record in records:
        formatted.append(
            {
                "doc_id": record.doc_id,
                "doc_title": record.doc_title,
                "section_num": record.section_num,
                "text": record.text,
                "score": record.score,
                "group": record.group,
            }
        )
    return formatted


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
