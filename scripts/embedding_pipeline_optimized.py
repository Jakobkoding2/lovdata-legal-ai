#!/usr/bin/env python3
"""
Optimized Embedding Pipeline for Lovdata Legal Texts
Processes in chunks to handle large datasets efficiently.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import gc

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Use lighter model for memory efficiency
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 32
CHUNK_SIZE = 5000  # Process 5K texts at a time
MAX_TEXTS = 20000  # Limit for initial run (representative sample)


def generate_embeddings_chunked(texts: List[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Generate embeddings in chunks to manage memory"""
    from sentence_transformers import SentenceTransformer
    
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"✓ Model loaded")
    
    all_embeddings = []
    
    # Process in chunks
    num_chunks = (len(texts) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"\nProcessing {len(texts)} texts in {num_chunks} chunks...")
    
    for i in range(0, len(texts), CHUNK_SIZE):
        chunk_texts = texts[i:i + CHUNK_SIZE]
        print(f"\nChunk {i//CHUNK_SIZE + 1}/{num_chunks}: {len(chunk_texts)} texts")
        
        embeddings = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Convert to FP16 to save memory
        embeddings = embeddings.astype(np.float16)
        all_embeddings.append(embeddings)
        
        # Clear memory
        del embeddings
        gc.collect()
    
    # Concatenate all chunks
    print("\nConcatenating embeddings...")
    final_embeddings = np.vstack(all_embeddings)
    
    print(f"✓ Generated embeddings with shape: {final_embeddings.shape}")
    
    return final_embeddings


def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS index"""
    import faiss
    
    dimension = embeddings.shape[1]
    n_vectors = embeddings.shape[0]
    
    print(f"\nBuilding FAISS index...")
    print(f"Dimension: {dimension}")
    print(f"Vectors: {n_vectors}")
    
    # Convert to FP32 for FAISS
    embeddings_fp32 = embeddings.astype(np.float32)
    
    # Use Flat index for smaller datasets, HNSW for larger
    if n_vectors < 100000:
        print("Using Flat index (exact search)")
        index = faiss.IndexFlatIP(dimension)  # Inner product (for normalized vectors = cosine)
    else:
        print("Using HNSW index (approximate search)")
        M = 32
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = 200
    
    print("Adding vectors to index...")
    index.add(embeddings_fp32)
    
    print(f"✓ Index built with {index.ntotal} vectors")
    
    return index


def analyze_similarities(index, embeddings: np.ndarray, df: pd.DataFrame, sample_size: int = 2000):
    """Analyze semantic similarities"""
    import faiss
    
    print(f"\nAnalyzing similarities for {sample_size} samples...")
    
    # Sample indices
    if sample_size > len(df):
        sample_size = len(df)
    
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    
    high_similarity_pairs = []
    
    for idx in tqdm(sample_indices):
        query_embedding = embeddings[idx:idx+1].astype(np.float32)
        
        # Search for top 10 similar
        distances, indices = index.search(query_embedding, 11)  # +1 for self
        
        # Convert to similarities (for normalized vectors with IP)
        similarities = distances[0]
        
        # Process results (skip first which is self)
        for sim_idx, similarity in zip(indices[0][1:], similarities[1:]):
            if similarity >= 0.7:  # High similarity threshold
                row1 = df.iloc[idx]
                row2 = df.iloc[sim_idx]
                
                high_similarity_pairs.append({
                    'idx1': idx,
                    'idx2': int(sim_idx),
                    'similarity': float(similarity),
                    'doc1_id': row1['doc_id'],
                    'doc2_id': row2['doc_id'],
                    'group1': row1['group'],
                    'group2': row2['group'],
                    'same_doc': row1['doc_id'] == row2['doc_id'],
                    'cross_group': row1['group'] != row2['group']
                })
    
    similarity_df = pd.DataFrame(high_similarity_pairs)
    print(f"✓ Found {len(similarity_df)} high-similarity pairs")
    
    return similarity_df


def compute_overlap_stats(similarity_df: pd.DataFrame) -> Dict:
    """Compute overlap statistics"""
    
    if similarity_df.empty:
        return {
            'total_high_similarity_pairs': 0,
            'cross_group_overlaps': 0,
            'potential_duplicates': 0,
            'law_regulation_overlaps': 0
        }
    
    duplicates = similarity_df[similarity_df['similarity'] > 0.95]
    cross_overlaps = similarity_df[
        (similarity_df['cross_group']) & 
        (similarity_df['similarity'] > 0.85)
    ]
    
    stats = {
        'total_high_similarity_pairs': len(similarity_df),
        'cross_group_overlaps': int(similarity_df['cross_group'].sum()),
        'potential_duplicates': len(duplicates),
        'law_regulation_overlaps': len(cross_overlaps),
        'avg_similarity': float(similarity_df['similarity'].mean()) if len(similarity_df) > 0 else 0.0,
        'max_similarity': float(similarity_df['similarity'].max()) if len(similarity_df) > 0 else 0.0
    }
    
    return stats


def main():
    """Main pipeline execution"""
    print("=" * 60)
    print("Lovdata Embedding Pipeline (Optimized)")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    corpus_path = PROCESSED_DIR / "lovdata_corpus.parquet"
    
    if not corpus_path.exists():
        print(f"ERROR: Dataset not found at {corpus_path}")
        sys.exit(1)
    
    df = pd.read_parquet(corpus_path)
    print(f"✓ Loaded {len(df)} legal text units")
    
    # Limit for initial run
    if len(df) > MAX_TEXTS:
        print(f"\nLimiting to first {MAX_TEXTS} texts for efficiency...")
        df = df.head(MAX_TEXTS)
    
    texts = df['text_clean'].tolist()
    
    # Generate embeddings
    print("\n[2/5] Generating embeddings...")
    embeddings = generate_embeddings_chunked(texts, EMBEDDING_MODEL, BATCH_SIZE)
    
    # Save embeddings
    embeddings_path = MODELS_DIR / "lovdata_embeddings.npy"
    np.save(embeddings_path, embeddings)
    size_mb = embeddings_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved embeddings to {embeddings_path} ({size_mb:.1f} MB)")
    
    # Build FAISS index
    print("\n[3/5] Building FAISS index...")
    index = build_faiss_index(embeddings)
    
    # Save index
    import faiss
    index_path = MODELS_DIR / "lovdata_faiss.index"
    faiss.write_index(index, str(index_path))
    size_mb = index_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved index to {index_path} ({size_mb:.1f} MB)")
    
    # Analyze similarities
    print("\n[4/5] Analyzing semantic similarities...")
    similarity_df = analyze_similarities(index, embeddings, df, sample_size=2000)
    
    # Save similarity pairs
    similarity_path = PROCESSED_DIR / "similarity_pairs.parquet"
    similarity_df.to_parquet(similarity_path, index=False)
    print(f"✓ Saved similarity pairs to {similarity_path}")
    
    # Compute stats
    print("\n[5/5] Computing overlap statistics...")
    stats = compute_overlap_stats(similarity_df)
    
    # Add embedding info
    stats['embedding_model'] = EMBEDDING_MODEL
    stats['embedding_dimension'] = int(embeddings.shape[1])
    stats['total_vectors'] = int(embeddings.shape[0])
    
    # Save stats
    stats_path = MODELS_DIR / "embedding_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Embedding Pipeline Complete!")
    print("=" * 60)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Index size: {index.ntotal} vectors")
    print(f"High-similarity pairs: {stats['total_high_similarity_pairs']}")
    print(f"Cross-group overlaps: {stats['cross_group_overlaps']}")
    print(f"Potential duplicates: {stats['potential_duplicates']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
