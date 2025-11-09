#!/usr/bin/env python3
"""
Lovdata Legal AI API Server
Provides endpoints for semantic search, Q&A, and overlap detection.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Initialize FastAPI app
app = FastAPI(
    title="Lovdata Legal AI API",
    description="Semantic search and Q&A for Norwegian legal texts",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
corpus_df = None
embeddings = None
faiss_index = None
embedding_model = None
overlap_classifier = None


# Request/Response Models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    min_similarity: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
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
    context_size: int = Field(3, ge=1, le=10, description="Number of context paragraphs")


class QAResponse(BaseModel):
    question: str
    answer: str
    sources: List[SearchResult]


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    corpus_size: int
    index_size: int


# Startup event
@app.on_event("startup")
async def load_models():
    """Load all models and data on startup"""
    global corpus_df, embeddings, faiss_index, embedding_model, overlap_classifier
    
    print("Loading models and data...")
    
    # Load corpus
    corpus_path = PROCESSED_DIR / "lovdata_corpus.parquet"
    if corpus_path.exists():
        corpus_df = pd.read_parquet(corpus_path).head(20000)  # Limit to embedded subset
        print(f"✓ Loaded corpus: {len(corpus_df)} texts")
    else:
        print(f"WARNING: Corpus not found at {corpus_path}")
    
    # Load embeddings
    embeddings_path = MODELS_DIR / "lovdata_embeddings.npy"
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        print(f"✓ Loaded embeddings: {embeddings.shape}")
    else:
        print(f"WARNING: Embeddings not found at {embeddings_path}")
    
    # Load FAISS index
    index_path = MODELS_DIR / "lovdata_faiss.index"
    if index_path.exists():
        import faiss
        faiss_index = faiss.read_index(str(index_path))
        print(f"✓ Loaded FAISS index: {faiss_index.ntotal} vectors")
    else:
        print(f"WARNING: FAISS index not found at {index_path}")
    
    # Load embedding model
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print(f"✓ Loaded embedding model")
    except Exception as e:
        print(f"WARNING: Could not load embedding model: {e}")
    
    # Load overlap classifier
    classifier_path = MODELS_DIR / "overlap_classifier.joblib"
    if classifier_path.exists():
        import joblib
        model_data = joblib.load(classifier_path)
        overlap_classifier = model_data
        print(f"✓ Loaded overlap classifier")
    else:
        print(f"WARNING: Overlap classifier not found at {classifier_path}")
    
    print("Models loaded successfully!")


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Lovdata Legal AI API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "corpus": corpus_df is not None,
            "embeddings": embeddings is not None,
            "faiss_index": faiss_index is not None,
            "embedding_model": embedding_model is not None,
            "overlap_classifier": overlap_classifier is not None
        },
        corpus_size=len(corpus_df) if corpus_df is not None else 0,
        index_size=faiss_index.ntotal if faiss_index is not None else 0
    )


@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Semantic search endpoint"""
    
    if embedding_model is None or faiss_index is None or corpus_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Generate query embedding
    query_embedding = embedding_model.encode(
        [request.query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    
    # Search FAISS index
    distances, indices = faiss_index.search(query_embedding, request.top_k * 2)
    
    # Convert to similarities
    similarities = distances[0]
    
    # Filter and format results
    results = []
    for idx, similarity in zip(indices[0], similarities):
        if similarity < request.min_similarity:
            continue
        
        if idx >= len(corpus_df):
            continue
        
        row = corpus_df.iloc[idx]
        
        # Apply group filter
        if request.filter_group and row['group'] != request.filter_group:
            continue
        
        results.append(SearchResult(
            doc_id=row['doc_id'],
            doc_title=row['doc_title'],
            section_num=row.get('section_num'),
            text=row['text_clean'][:500],  # Limit text length
            similarity=float(similarity),
            group=row['group']
        ))
        
        if len(results) >= request.top_k:
            break
    
    return SearchResponse(
        query=request.query,
        results=results,
        total_results=len(results)
    )


@app.post("/detect_overlap", response_model=OverlapResponse)
async def detect_overlap(request: OverlapRequest):
    """Detect semantic overlap between two legal texts"""
    
    if embedding_model is None or overlap_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Generate embeddings
    embeddings_pair = embedding_model.encode(
        [request.text1, request.text2],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Compute similarity
    similarity = float(np.dot(embeddings_pair[0], embeddings_pair[1]))
    
    # Extract features
    len1 = len(request.text1)
    len2 = len(request.text2)
    words1 = set(request.text1.lower().split())
    words2 = set(request.text2.lower().split())
    
    features = {
        'similarity': similarity,
        'len_ratio': min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0,
        'len_diff': abs(len1 - len2),
        'word_overlap': len(words1 & words2) / len(words1 | words2) if len(words1 | words2) > 0 else 0,
        'same_doc': 0,
        'cross_group': 0,
        'avg_length': (len1 + len2) / 2,
        'max_length': max(len1, len2),
        'min_length': min(len1, len2)
    }
    
    # Predict overlap type
    classifier = overlap_classifier['classifier']
    feature_names = overlap_classifier['feature_names']
    
    X = np.array([[features[col] for col in feature_names]])
    
    prediction = classifier.predict(X)[0]
    probabilities = classifier.predict_proba(X)[0]
    prob_dict = dict(zip(classifier.classes_, [float(p) for p in probabilities]))
    
    # Generate explanation
    if prediction == 'duplicate':
        explanation = "Tekstene er nesten identiske. Dette kan indikere duplikasjon."
    elif prediction == 'subsumption':
        explanation = "En tekst inneholder eller impliserer den andre. Dette kan indikere subsumpsjon."
    elif prediction == 'delegation':
        explanation = "Tekstene viser til hverandre eller har delegasjonsforhold."
    else:
        explanation = "Tekstene er semantisk forskjellige."
    
    return OverlapResponse(
        overlap_type=prediction,
        similarity=similarity,
        probabilities=prob_dict,
        explanation=explanation
    )


@app.post("/ask_law", response_model=QAResponse)
async def ask_legal_question(request: QARequest):
    """Answer legal questions using semantic search + LLM"""
    
    if embedding_model is None or faiss_index is None or corpus_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # First, do semantic search to find relevant context
    query_embedding = embedding_model.encode(
        [request.question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    
    distances, indices = faiss_index.search(query_embedding, request.context_size)
    similarities = distances[0]
    
    # Collect context
    sources = []
    context_texts = []
    
    for idx, similarity in zip(indices[0], similarities):
        if idx >= len(corpus_df):
            continue
        
        row = corpus_df.iloc[idx]
        
        sources.append(SearchResult(
            doc_id=row['doc_id'],
            doc_title=row['doc_title'],
            section_num=row.get('section_num'),
            text=row['text_clean'][:500],
            similarity=float(similarity),
            group=row['group']
        ))
        
        context_texts.append(f"{row['doc_title']} § {row.get('section_num', 'N/A')}: {row['text_clean']}")
    
    # Generate answer using OpenAI API (if available)
    try:
        from openai import OpenAI
        client = OpenAI()  # Uses OPENAI_API_KEY from environment
        
        context = "\n\n".join(context_texts)
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Du er en ekspert på norsk lov. Svar basert på den gitte konteksten."},
                {"role": "user", "content": f"Kontekst:\n{context}\n\nSpørsmål: {request.question}\n\nSvar kort og presist basert på konteksten."}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
    except Exception as e:
        # Fallback: return context-based answer
        answer = f"Basert på relevante lovtekster:\n\n{context_texts[0] if context_texts else 'Ingen relevant lovtekst funnet.'}"
    
    return QAResponse(
        question=request.question,
        answer=answer,
        sources=sources
    )


@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    
    stats = {}
    
    if corpus_df is not None:
        stats['corpus'] = {
            'total_texts': len(corpus_df),
            'laws': int(corpus_df[corpus_df['group'] == 'law'].shape[0]),
            'regulations': int(corpus_df[corpus_df['group'] == 'regulation'].shape[0]),
            'unique_documents': int(corpus_df['doc_id'].nunique())
        }
    
    if embeddings is not None:
        stats['embeddings'] = {
            'shape': embeddings.shape,
            'dimension': int(embeddings.shape[1]),
            'dtype': str(embeddings.dtype)
        }
    
    if faiss_index is not None:
        stats['index'] = {
            'total_vectors': int(faiss_index.ntotal),
            'dimension': int(faiss_index.d)
        }
    
    return stats


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
