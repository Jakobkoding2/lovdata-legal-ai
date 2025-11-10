#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from api.rag_pipeline import CodexRAGPipeline


def main() -> None:
    base_dir = Path(__file__).parent.parent
    pipeline = CodexRAGPipeline(base_dir)
    print("Codex RAG pipeline assets generated")
    print(f"Total chunks: {len(pipeline.chunk_df)}")
    if pipeline.embeddings is not None:
        print(f"Embeddings shape: {pipeline.embeddings.shape}")
    if pipeline.faiss_index is not None:
        print(f"Index size: {pipeline.faiss_index.ntotal}")


if __name__ == "__main__":
    main()
