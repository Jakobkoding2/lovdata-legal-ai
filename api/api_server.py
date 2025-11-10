#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent))

from api.rag_pipeline import (  # noqa: E402
    CodexRAGPipeline,
    ChunkRecord,
    compute_overlap,
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Lovdata Legal AI API",
    description="Semantic search and Q&A for Norwegian legal texts",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline: Optional[CodexRAGPipeline] = None
overlap_classifier = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    filter_group: Optional[str] = Field(None, description="Filter by 'law' or 'regulation'")


class SearchResult(BaseModel):
    doc_id: str
    doc_title: str
    section_num: Optional[str]
    text: str
    similarity: float
    group: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int


class OverlapRequest(BaseModel):
    text1: str = Field(..., description="First legal text")
    text2: str = Field(..., description="Second legal text")


class OverlapResponse(BaseModel):
    overlap_type: str
    similarity: float
    probabilities: Dict[str, float]
    explanation: str


class QARequest(BaseModel):
    question: str = Field(..., description="Legal question in Norwegian")
    context_size: int = Field(5, ge=1, le=10, description="Number of context paragraphs")


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
    print("Loading Codex RAG pipeline...")
    try:
        rag_pipeline = CodexRAGPipeline(BASE_DIR)
        print("✓ Codex RAG pipeline loaded")
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: Failed to load Codex RAG pipeline: {exc}")
        rag_pipeline = None

    classifier_path = MODELS_DIR / "overlap_classifier.joblib"
    if classifier_path.exists():
        import joblib

        overlap_classifier = joblib.load(classifier_path)
        print("✓ Loaded overlap classifier")
    else:
        overlap_classifier = None
        print(f"WARNING: Overlap classifier not found at {classifier_path}")


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    return {
        "message": "Lovdata Legal AI API",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    models_loaded = {
        "rag_pipeline": rag_pipeline is not None,
        "faiss_index": getattr(rag_pipeline, "faiss_index", None) is not None,
        "bm25": getattr(rag_pipeline, "bm25", None) is not None,
        "embedder": getattr(rag_pipeline, "embedding_model", None) is not None,
        "reranker": getattr(rag_pipeline, "cross_encoder", None) is not None,
        "overlap_classifier": overlap_classifier is not None,
    }

    corpus_size = 0
    index_size = 0
    if rag_pipeline is not None:
        corpus_size = len(rag_pipeline.chunk_df)
        index_size = rag_pipeline.faiss_index.ntotal if rag_pipeline.faiss_index is not None else 0

    return HealthResponse(
        status="healthy" if models_loaded["rag_pipeline"] else "degraded",
        models_loaded=models_loaded,
        corpus_size=corpus_size,
        index_size=index_size,
    )


@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest) -> SearchResponse:
    if rag_pipeline is None or rag_pipeline.faiss_index is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    candidates = rag_pipeline.search(request.query, max(request.top_k, rag_pipeline.rerank_top_k))
    results: List[SearchResult] = []
    for record in candidates:
        if request.filter_group and record.group != request.filter_group:
            continue
        if record.score < request.min_similarity:
            continue
        results.append(
            SearchResult(
                doc_id=record.doc_id,
                doc_title=record.doc_title,
                section_num=record.section_num,
                text=record.text[:500],
                similarity=float(record.score),
                group=record.group,
            )
        )
        if len(results) >= request.top_k:
            break

    return SearchResponse(query=request.query, results=results, total_results=len(results))


@app.post("/detect_overlap", response_model=OverlapResponse)
async def detect_overlap(request: OverlapRequest) -> OverlapResponse:
    if rag_pipeline is None or rag_pipeline.embedding_model is None or overlap_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    similarity = compute_overlap(rag_pipeline.embedding_model, request.text1, request.text2)

    len1 = len(request.text1)
    len2 = len(request.text2)
    words1 = set(request.text1.lower().split())
    words2 = set(request.text2.lower().split())

    features = {
        "similarity": similarity,
        "len_ratio": min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0,
        "len_diff": abs(len1 - len2),
        "word_overlap": len(words1 & words2) / len(words1 | words2) if len(words1 | words2) > 0 else 0,
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

    if prediction == "duplicate":
        explanation = "Tekstene er nesten identiske. Dette kan indikere duplikasjon."
    elif prediction == "subsumption":
        explanation = "En tekst inneholder eller impliserer den andre. Dette kan indikere subsumpsjon."
    elif prediction == "delegation":
        explanation = "Tekstene viser til hverandre eller har delegasjonsforhold."
    else:
        explanation = "Tekstene er semantisk forskjellige."

    return OverlapResponse(
        overlap_type=prediction,
        similarity=similarity,
        probabilities=prob_dict,
        explanation=explanation,
    )


@app.post("/ask_law", response_model=QAResponse)
async def ask_legal_question(request: QARequest) -> QAResponse:
    if rag_pipeline is None or rag_pipeline.faiss_index is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    contexts = rag_pipeline.top_contexts(request.question)
    if not contexts:
        raise HTTPException(status_code=404, detail="Ingen relevant lovtekst funnet")

    answer = await generate_answer(request.question, contexts)

    sources: List[SearchResult] = []
    for record in contexts[: request.context_size]:
        sources.append(
            SearchResult(
                doc_id=record.doc_id,
                doc_title=record.doc_title,
                section_num=record.section_num,
                text=record.text[:500],
                similarity=float(record.score),
                group=record.group,
            )
        )

    return QAResponse(question=request.question, answer=answer, sources=sources)


async def generate_answer(question: str, contexts: List[ChunkRecord]) -> str:
    from openai import OpenAI

    context_strings = [
        f"{record.doc_title} § {record.section_num or 'N/A'}: {record.text}" for record in contexts
    ]
    context_block = "\n\n".join(context_strings)

    messages = [
        {
            "role": "system",
            "content": "Du er en ekspert på norsk lov. Svar presist og basert på den gitte konteksten.",
        },
        {
            "role": "user",
            "content": f"Kontekst:\n{context_block}\n\nSpørsmål: {question}\n\nSvar på norsk med tydelige referanser til relevante paragrafer.",
        },
    ]

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=800,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception:
        fallback_key = os.getenv("TOGETHER_API_KEY")
        if fallback_key:
            try:
                fallback_client = OpenAI(api_key=fallback_key, base_url="https://api.together.xyz/v1")
                response = fallback_client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.2,
                )
                return response.choices[0].message.content
            except Exception:
                pass

    return "Kunne ikke generere svar basert på tilgjengelige modeller. Vennligst prøv igjen senere."


@app.get("/stats")
async def get_statistics() -> Dict[str, Dict]:
    stats: Dict[str, Dict] = {}

    if rag_pipeline is not None:
        stats["corpus"] = {
            "total_chunks": len(rag_pipeline.chunk_df),
            "unique_documents": int(rag_pipeline.chunk_df["doc_id"].nunique()) if not rag_pipeline.chunk_df.empty else 0,
        }
        if rag_pipeline.embeddings is not None:
            stats["embeddings"] = {
                "shape": rag_pipeline.embeddings.shape,
                "dimension": int(rag_pipeline.embeddings.shape[1]) if rag_pipeline.embeddings.size else 0,
                "dtype": str(rag_pipeline.embeddings.dtype),
            }
        if rag_pipeline.faiss_index is not None:
            stats["index"] = {
                "total_vectors": int(rag_pipeline.faiss_index.ntotal),
                "efSearch": getattr(rag_pipeline.faiss_index.hnsw, "efSearch", None),
            }

    return stats


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
