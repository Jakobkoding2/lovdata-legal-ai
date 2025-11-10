#!/usr/bin/env python3
import json
import logging
import os
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent))

from api.rag_pipeline import CodexRAGPipeline, ChunkRecord, compute_overlap  # noqa: E402
from lovdata_rag.bootstrap import ensure_assets_ready  # noqa: E402
from lovdata_rag.config import MODELS_DIR  # noqa: E402
from lovdata_rag.ft import resolve_active_model  # noqa: E402
from openai import OpenAI

DEFAULT_MODEL_NAME = "gpt-5-mini"
FALLBACK_MODEL_NAME = "gpt-4o-mini"

logger = logging.getLogger("lovdata_rag.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Lovdata Legal AI API",
    description="Hybrid RAG pipeline for Norwegian laws and forskrifter",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counts = {"search": 0, "ask_law": 0, "detect_overlap": 0}
        self._latency: Dict[str, List[float]] = {key: [] for key in self._counts}

    def record(self, endpoint: str, latency: float) -> None:
        with self._lock:
            if endpoint in self._counts:
                self._counts[endpoint] += 1
                self._latency[endpoint].append(latency)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                "counts": dict(self._counts),
                "latency_ms": {
                    key: (sum(values) / len(values) * 1000 if values else 0.0)
                    for key, values in self._latency.items()
                },
            }


metrics = MetricsCollector()
rag_pipeline: Optional[CodexRAGPipeline] = None
overlap_classifier = None


def _resolve_model_name() -> str:
    return resolve_active_model(DEFAULT_MODEL_NAME, FALLBACK_MODEL_NAME)


def _build_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="OPENAI_API_KEY must be set to call language models.",
        )
    client_kwargs: Dict[str, str] = {"api_key": api_key}
    base_url = os.getenv("MODEL_PROVIDER")
    if base_url and base_url.startswith("http"):
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(10, ge=1, le=50)
    min_similarity: float = Field(0.0, ge=0.0, le=1.0)
    filter_group: Optional[str] = Field(None, description="Filter by 'law' or 'regulation'")


class SearchResult(BaseModel):
    doc_id: str
    doc_title: str
    section_num: Optional[str]
    section_title: Optional[str]
    kapittel: Optional[str]
    ledd: Optional[str]
    text: str
    similarity: float
    group: str
    date: Optional[str]
    source_url: Optional[str]
    start_char: Optional[int]
    end_char: Optional[int]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int


class OverlapRequest(BaseModel):
    text1: str
    text2: str


class OverlapResponse(BaseModel):
    overlap_type: str
    similarity: float
    probabilities: Dict[str, float]
    explanation: str


class QARequest(BaseModel):
    question: str
    context_size: int = Field(5, ge=1, le=10)


class QAResponse(BaseModel):
    question: str
    answer: str
    sources: List[SearchResult]


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    corpus_size: int
    index_size: int


@app.on_event("startup")
async def load_models() -> None:
    global rag_pipeline, overlap_classifier
    ensure_assets_ready()
    logger.info("Loading RAG pipeline")
    rag_pipeline = CodexRAGPipeline()
    classifier_path = MODELS_DIR / "overlap_classifier.joblib"
    if classifier_path.exists():
        import joblib

        overlap_classifier = joblib.load(classifier_path)
        logger.info("Overlap classifier loaded")
    else:
        overlap_classifier = None
        logger.warning("Overlap classifier missing at %s", classifier_path)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Lovdata Legal AI API", "docs": "/docs"}


@app.get("/healthz")
async def healthz() -> Dict[str, object]:
    return {"ok": rag_pipeline is not None, "model": _resolve_model_name()}


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    models_loaded = {
        "rag_pipeline": rag_pipeline is not None,
        "faiss_index": bool(getattr(rag_pipeline, "faiss_index", None)),
        "bm25": bool(getattr(rag_pipeline, "bm25", None)),
        "embedder": bool(getattr(rag_pipeline, "embedding_model", None)),
        "overlap_classifier": overlap_classifier is not None,
    }
    corpus_size = len(rag_pipeline.chunk_df) if rag_pipeline else 0
    index_size = (
        rag_pipeline.faiss_index.ntotal if rag_pipeline and rag_pipeline.faiss_index is not None else 0
    )
    status = "healthy" if models_loaded["rag_pipeline"] else "degraded"
    return HealthResponse(status=status, models_loaded=models_loaded, corpus_size=corpus_size, index_size=index_size)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    start = time.perf_counter()
    if rag_pipeline is None or rag_pipeline.faiss_index is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    candidates = rag_pipeline.search(request.query, max(request.top_k, rag_pipeline.rerank_top_k))
    results: List[SearchResult] = []
    for record in candidates:
        if request.filter_group and record.group.lower() != request.filter_group.lower():
            continue
        if record.score < request.min_similarity:
            continue
        results.append(
            SearchResult(
                doc_id=record.doc_id,
                doc_title=record.doc_title,
                section_num=record.section_num,
                section_title=record.section_title,
                kapittel=record.kapittel,
                ledd=record.ledd,
                text=record.text[:500],
                similarity=float(record.score),
                group=record.group,
                date=record.date,
                source_url=record.source_url,
                start_char=record.start_char,
                end_char=record.end_char,
            )
        )
        if len(results) >= request.top_k:
            break
    metrics.record("search", time.perf_counter() - start)
    return SearchResponse(query=request.query, results=results, total_results=len(results))


@app.post("/detect_overlap", response_model=OverlapResponse)
async def detect_overlap_endpoint(request: OverlapRequest) -> OverlapResponse:
    start = time.perf_counter()
    if rag_pipeline is None or rag_pipeline.embedding_model is None or overlap_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    similarity = compute_overlap(rag_pipeline.embedding_model, request.text1, request.text2)
    len1 = len(request.text1)
    len2 = len(request.text2)
    words1 = set(request.text1.lower().split())
    words2 = set(request.text2.lower().split())

    features = {
        "similarity": similarity,
        "len_ratio": min(len1, len2) / max(len1, len2) if max(len1, len2) else 0,
        "len_diff": abs(len1 - len2),
        "word_overlap": len(words1 & words2) / len(words1 | words2) if words1 or words2 else 0,
        "same_doc": 0,
        "cross_group": 0,
        "avg_length": (len1 + len2) / 2,
        "max_length": max(len1, len2),
        "min_length": min(len1, len2),
    }

    classifier = overlap_classifier["classifier"]
    feature_names = overlap_classifier["feature_names"]
    X = np.array([[features[col] for col in feature_names]])
    prediction = classifier.predict(X)[0]
    probabilities = classifier.predict_proba(X)[0]
    prob_dict = dict(zip(classifier.classes_, [float(p) for p in probabilities]))

    explanations = {
        "duplicate": "Tekstene er nesten identiske og indikerer duplisering.",
        "subsumption": "En tekst innlemmer den andre og kan tyde pa subsumsjon.",
        "delegation": "Tekstene peker til hverandre eller fordeler ansvar.",
    }
    explanation = explanations.get(prediction, "Tekstene er semantisk ulike.")
    metrics.record("detect_overlap", time.perf_counter() - start)
    return OverlapResponse(overlap_type=prediction, similarity=similarity, probabilities=prob_dict, explanation=explanation)


@app.post("/ask_law", response_model=QAResponse)
async def ask_law(request: QARequest) -> QAResponse:
    start = time.perf_counter()
    if rag_pipeline is None or rag_pipeline.faiss_index is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    contexts = rag_pipeline.top_contexts(request.question)
    if not contexts or contexts[0].score < 0.2:
        metrics.record("ask_law", time.perf_counter() - start)
        raise HTTPException(status_code=404, detail="Ingen tilstrekkelig dokumentasjon funnet.")
    answer = await generate_answer(request.question, contexts)
    sources = [
        SearchResult(
            doc_id=record.doc_id,
            doc_title=record.doc_title,
            section_num=record.section_num,
            section_title=record.section_title,
            kapittel=record.kapittel,
            ledd=record.ledd,
            text=record.text[:500],
            similarity=float(record.score),
            group=record.group,
            date=record.date,
            source_url=record.source_url,
            start_char=record.start_char,
            end_char=record.end_char,
        )
        for record in contexts[: request.context_size]
    ]
    metrics.record("ask_law", time.perf_counter() - start)
    return QAResponse(question=request.question, answer=answer, sources=sources)


async def generate_answer(question: str, contexts: List[ChunkRecord]) -> str:
    source_map: Dict[str, ChunkRecord] = {}
    sections: List[str] = []
    for idx, record in enumerate(contexts, start=1):
        source_id = f"S{idx}"
        source_map[source_id] = record
        header = (
            f"{source_id} | doc_id={record.doc_id} | tittel={record.doc_title} | "
            f"paragraf={record.section_num or 'ukjent'}"
        )
        sections.append(f"{header}\n{record.text}")
    context_block = "\n\n".join(sections)
    system_prompt = (
        "Du er en ekspert pa norsk lov. Gi presise svar basert pa konteksten. "
        "Returner JSON med feltene 'answer' og 'citations'. 'citations' skal vere en liste med objekter "
        "som minst inneholder 'source_id' (S1, S2, ...). Legg til valgfri 'snippet'."
    )
    user_prompt = f"Kontekst:\n{context_block}\n\nSporsmal: {question}\n\nSvar med tydelige sitater."
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "legal_answer",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_id": {"type": "string"},
                                "snippet": {"type": "string"},
                            },
                            "required": ["source_id"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["answer", "citations"],
                "additionalProperties": False,
            },
        },
    }
    try:
        client = _build_openai_client()
        response = client.responses.create(
            model=_resolve_model_name(),
            input=[
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            temperature=0.2,
            max_output_tokens=800,
            response_format=response_format,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.error("LLM request failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    output_text = getattr(response, "output_text", "").strip()
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail="LLM returned malformed JSON.") from exc

    answer_text = str(parsed.get("answer", "")).strip()
    citation_lines: List[str] = []
    for citation in parsed.get("citations", []) or []:
        source_id = citation.get("source_id")
        record = source_map.get(source_id)
        if not record:
            continue
        label = record.doc_title or record.doc_id
        if record.section_num:
            label = f"{label} § {record.section_num}"
        snippet = citation.get("snippet", "")
        line = f"- {label} ({record.doc_id})"
        if snippet:
            line = f"{line}: {snippet.strip()}"
        citation_lines.append(line)

    if citation_lines:
        answer_text = f"{answer_text}\n\nKilder:\n" + "\n".join(citation_lines)
    if not answer_text:
        raise HTTPException(status_code=502, detail="LLM returned empty answer.")
    return answer_text


@app.get("/stats")
async def stats() -> Dict[str, Dict]:
    payload: Dict[str, Dict] = {}
    if rag_pipeline:
        payload["corpus"] = {"total_chunks": len(rag_pipeline.chunk_df)}
        if rag_pipeline.faiss_index is not None:
            payload["index"] = {"total_vectors": int(rag_pipeline.faiss_index.ntotal)}
    return payload


@app.get("/metrics")
async def metrics_endpoint() -> Dict[str, Dict[str, float]]:
    return metrics.snapshot()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.api_server:app", host="0.0.0.0", port=8000, reload=False)
