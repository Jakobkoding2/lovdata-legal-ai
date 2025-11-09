#!/usr/bin/env python3
"""
Embedding Pipeline for Lovdata Legal Texts
Generates embeddings using BGE-M3 and builds FAISS index for semantic search.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "BAAI/bge-m3"
BATCH_SIZE = 32
MAX_LENGTH = 512


class EmbeddingGenerator:
    """Generates embeddings for legal texts using BGE-M3"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = None):
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading embedding model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Set to FP16 for efficiency if using GPU
        if self.device == "cuda":
            self.model = self.model.half()
            print("✓ Model converted to FP16")
        
        print(f"✓ Model loaded: {model_name}")
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = BATCH_SIZE,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        
        print(f"\nGenerating embeddings for {len(texts)} texts...")
        print(f"Batch size: {batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        
        # Convert to FP16 to save space
        embeddings = embeddings.astype(np.float16)
        
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: Path):
        """Save embeddings to disk"""
        np.save(filepath, embeddings)
        
        # Calculate size
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ Saved embeddings to {filepath} ({size_mb:.1f} MB)")


class FAISSIndexBuilder:
    """Builds and manages FAISS index for semantic search"""
    
    def __init__(self):
        try:
            import faiss
            self.faiss = faiss
            print("✓ FAISS library loaded")
        except ImportError:
            print("ERROR: FAISS not installed. Install with: pip install faiss-cpu")
            sys.exit(1)
    
    def build_index(
        self, 
        embeddings: np.ndarray,
        use_gpu: bool = False
    ) -> 'faiss.Index':
        """Build FAISS HNSW index for efficient similarity search"""
        
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        print(f"\nBuilding FAISS index...")
        print(f"Dimension: {dimension}")
        print(f"Vectors: {n_vectors}")
        
        # Convert FP16 to FP32 for FAISS
        embeddings_fp32 = embeddings.astype(np.float32)
        
        # Use HNSW index for better performance
        # M = number of connections per layer (16-64 typical)
        # efConstruction = size of dynamic candidate list (40-500 typical)
        M = 32
        efConstruction = 200
        
        index = self.faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = efConstruction
        
        print(f"Index type: HNSW (M={M}, efConstruction={efConstruction})")
        
        # Add vectors to index
        print("Adding vectors to index...")
        index.add(embeddings_fp32)
        
        print(f"✓ Index built with {index.ntotal} vectors")
        
        return index
    
    def save_index(self, index: 'faiss.Index', filepath: Path):
        """Save FAISS index to disk"""
        self.faiss.write_index(index, str(filepath))
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ Saved index to {filepath} ({size_mb:.1f} MB)")
    
    def load_index(self, filepath: Path) -> 'faiss.Index':
        """Load FAISS index from disk"""
        index = self.faiss.read_index(str(filepath))
        print(f"✓ Loaded index from {filepath}")
        return index


class SemanticSimilarityAnalyzer:
    """Analyzes semantic similarity between legal texts"""
    
    def __init__(self, index, embeddings: np.ndarray, df: pd.DataFrame):
        self.index = index
        self.embeddings = embeddings
        self.df = df
        
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            print("ERROR: FAISS not installed")
            sys.exit(1)
    
    def find_similar(
        self, 
        query_idx: int, 
        k: int = 10,
        min_similarity: float = 0.7
    ) -> List[Tuple[int, float]]:
        """Find k most similar texts to the query"""
        
        query_embedding = self.embeddings[query_idx:query_idx+1].astype(np.float32)
        
        # Search index
        distances, indices = self.index.search(query_embedding, k + 1)
        
        # Convert distances to similarities (for normalized vectors, distance = 2*(1-similarity))
        similarities = 1 - (distances[0] / 2)
        
        # Filter results (skip first result which is the query itself)
        results = []
        for idx, sim in zip(indices[0][1:], similarities[1:]):
            if sim >= min_similarity:
                results.append((int(idx), float(sim)))
        
        return results
    
    def compute_similarity_matrix(
        self,
        sample_size: int = 1000,
        min_similarity: float = 0.8
    ) -> pd.DataFrame:
        """Compute pairwise similarities for a sample"""
        
        print(f"\nComputing similarity matrix for {sample_size} samples...")
        
        # Sample indices
        if sample_size > len(self.df):
            sample_size = len(self.df)
        
        sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
        
        high_similarity_pairs = []
        
        for idx in tqdm(sample_indices):
            similar = self.find_similar(idx, k=10, min_similarity=min_similarity)
            
            for sim_idx, similarity in similar:
                # Get metadata
                row1 = self.df.iloc[idx]
                row2 = self.df.iloc[sim_idx]
                
                high_similarity_pairs.append({
                    'idx1': idx,
                    'idx2': sim_idx,
                    'similarity': similarity,
                    'doc1_id': row1['doc_id'],
                    'doc2_id': row2['doc_id'],
                    'group1': row1['group'],
                    'group2': row2['group'],
                    'text1': row1['text_clean'][:100],
                    'text2': row2['text_clean'][:100],
                    'same_doc': row1['doc_id'] == row2['doc_id'],
                    'cross_group': row1['group'] != row2['group']
                })
        
        similarity_df = pd.DataFrame(high_similarity_pairs)
        
        print(f"✓ Found {len(similarity_df)} high-similarity pairs")
        
        return similarity_df
    
    def analyze_overlaps(self, similarity_df: pd.DataFrame) -> Dict:
        """Analyze semantic overlaps and potential duplicates"""
        
        print("\nAnalyzing semantic overlaps...")
        
        # Overall statistics
        total_pairs = len(similarity_df)
        cross_group_pairs = similarity_df['cross_group'].sum()
        same_doc_pairs = similarity_df['same_doc'].sum()
        
        # High similarity (potential duplicates)
        duplicates = similarity_df[similarity_df['similarity'] > 0.95]
        
        # Cross-group overlaps (law ↔ regulation)
        cross_overlaps = similarity_df[
            (similarity_df['cross_group']) & 
            (similarity_df['similarity'] > 0.85)
        ]
        
        stats = {
            'total_high_similarity_pairs': total_pairs,
            'cross_group_overlaps': int(cross_group_pairs),
            'same_document_pairs': int(same_doc_pairs),
            'potential_duplicates': len(duplicates),
            'law_regulation_overlaps': len(cross_overlaps),
            'avg_similarity': float(similarity_df['similarity'].mean()),
            'max_similarity': float(similarity_df['similarity'].max())
        }
        
        print(f"Total high-similarity pairs: {stats['total_high_similarity_pairs']}")
        print(f"Cross-group overlaps: {stats['cross_group_overlaps']}")
        print(f"Potential duplicates (>0.95): {stats['potential_duplicates']}")
        print(f"Law ↔ Regulation overlaps: {stats['law_regulation_overlaps']}")
        
        return stats


def main():
    """Main embedding pipeline execution"""
    print("=" * 60)
    print("Lovdata Embedding & FAISS Index Pipeline")
    print("=" * 60)
    
    # Load processed dataset
    print("\n[1/5] Loading dataset...")
    corpus_path = PROCESSED_DIR / "lovdata_corpus.parquet"
    
    if not corpus_path.exists():
        print(f"ERROR: Dataset not found at {corpus_path}")
        print("Please run data_pipeline.py first")
        sys.exit(1)
    
    df = pd.read_parquet(corpus_path)
    print(f"✓ Loaded {len(df)} legal text units")
    
    # Generate embeddings
    print("\n[2/5] Generating embeddings...")
    generator = EmbeddingGenerator()
    
    texts = df['text_clean'].tolist()
    embeddings = generator.generate_embeddings(texts, batch_size=BATCH_SIZE)
    
    # Save embeddings
    embeddings_path = MODELS_DIR / "lovdata_embeddings.npy"
    generator.save_embeddings(embeddings, embeddings_path)
    
    # Build FAISS index
    print("\n[3/5] Building FAISS index...")
    builder = FAISSIndexBuilder()
    index = builder.build_index(embeddings)
    
    # Save index
    index_path = MODELS_DIR / "lovdata_faiss.index"
    builder.save_index(index, index_path)
    
    # Analyze semantic similarities
    print("\n[4/5] Analyzing semantic similarities...")
    analyzer = SemanticSimilarityAnalyzer(index, embeddings, df)
    
    # Compute similarity matrix for sample
    similarity_df = analyzer.compute_similarity_matrix(
        sample_size=min(5000, len(df)),
        min_similarity=0.75
    )
    
    # Save similarity pairs
    similarity_path = PROCESSED_DIR / "similarity_pairs.parquet"
    similarity_df.to_parquet(similarity_path, index=False)
    print(f"✓ Saved similarity pairs to {similarity_path}")
    
    # Analyze overlaps
    print("\n[5/5] Computing overlap statistics...")
    stats = analyzer.analyze_overlaps(similarity_df)
    
    # Save statistics
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
    print(f"High-similarity pairs found: {stats['total_high_similarity_pairs']}")
    print(f"Cross-group overlaps: {stats['cross_group_overlaps']}")
    print(f"Potential duplicates: {stats['potential_duplicates']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
