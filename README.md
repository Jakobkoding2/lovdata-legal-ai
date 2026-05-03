# Lovdata Legal AI

Autonomous AI system for training and deploying a Norwegian legal chatbot using the Lovdata API.

## Overview

Fetches and indexes Norwegian legal texts via the Lovdata API, fine-tunes a language model on Norwegian law, and deploys an interactive chatbot for legal Q&A and semantic search across legislation and case law.

## Features

- **Automated data collection** from Lovdata API (laws and regulations)
- **Hybrid RAG pipeline** combining FAISS vector search, BM25 keyword search, and cross-encoder reranking
- **Semantic search** across legislation and case law
- **Interactive Q&A** with source citations
- **Overlap detection** for identifying duplicate or related legal texts
- **REST API** built with FastAPI for easy integration
- **Gradio UI** for interactive chatbot experience

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Lovdata    │────▶│  Data        │────▶│  Chunking   │
│  API        │     │  Pipeline    │     │  & Embeds   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                       ┌────────────────────────┘
                       ▼
              ┌─────────────────┐
              │  FAISS + BM25   │
              │  Hybrid Index   │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌─────────┐  ┌──────────┐  ┌──────────┐
    │ Search  │  │  Q&A     │  │ Overlap  │
    │ API     │  │  API     │  │ Detection│
    └────┬────┘  └────┬─────┘  └────┬─────┘
         │            │             │
         └────────────┼─────────────┘
                      ▼
               ┌─────────────┐
               │  Gradio UI  │
               └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- 8GB+ RAM (16GB recommended for embedding generation)
- OpenAI API key (for LLM responses)

### Installation

```bash
# Clone the repository
git clone https://github.com/Jakobkoding2/lovdata-legal-ai.git
cd lovdata-legal-ai

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Start the API server (downloads data and builds index on first run)
python api/api_server.py
```

The server will:
1. Download Norwegian laws and regulations from Lovdata
2. Parse and chunk the legal texts
3. Generate embeddings using `intfloat/multilingual-e5-large`
4. Build FAISS and BM25 indexes
5. Start the FastAPI server on `http://localhost:8000`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/healthz` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/search` | POST | Semantic search with filters |
| `/ask_law` | POST | Legal Q&A with citations |
| `/detect_overlap` | POST | Detect text overlaps |
| `/stats` | GET | Corpus statistics |
| `/metrics` | GET | Performance metrics |

### Example Usage

```bash
# Search for legal texts
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "arbeidsmiljølov", "top_k": 5}'

# Ask a legal question
curl -X POST http://localhost:8000/ask_law \
  -H "Content-Type: application/json" \
  -d '{"question": "Hva er arbeidsgivers plikter etter arbeidsmiljøloven?"}'
```

### Gradio UI

```bash
# In another terminal
python app.py
```

Open `http://localhost:7860` in your browser for the interactive chatbot interface.

## Testing

```bash
# Run the test suite
pytest test_imports.py test_functional.py -v
```

Tests verify:
- All module imports work correctly
- Text utilities (sentence/section splitting)
- BM25 ranking
- API request/response models
- Configuration paths

## Tech Stack

- **Python 3.10+**
- **FastAPI** - REST API framework
- **FAISS** - Vector similarity search
- **BM25** - Keyword search ranking
- **Sentence Transformers** - Text embeddings (`intfloat/multilingual-e5-large`)
- **OpenAI API** - Language model for Q&A
- **Gradio** - Interactive UI
- **Pandas/NumPy** - Data processing

## Project Structure

```
lovdata-legal-ai/
├── api/
│   ├── api_server.py      # FastAPI server
│   ├── bm25.py            # BM25 implementation
│   └── rag_pipeline.py     # RAG pipeline logic
├── lovdata_rag/
│   ├── bootstrap.py        # Asset initialization
│   ├── chunking.py         # Text chunking
│   ├── config.py           # Configuration
│   ├── data_pipeline.py    # Data processing
│   ├── embeddings.py       # Embedding generation
│   ├── ft.py               # Fine-tuning utilities
│   ├── logging_utils.py    # Logging setup
│   ├── text_utils.py       # Text processing
│   └── update.py           # Data updates
├── scripts/
│   └── eval_rag.py         # RAG evaluation
├── app.py                  # Gradio UI
├── requirements.txt        # Dependencies
├── Dockerfile              # Container setup
├── docker-compose.yml      # Docker Compose config
└── README.md               # This file
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Background

Norwegian legal texts are publicly available via Lovdata but require legal expertise to navigate effectively. This project makes that content accessible through natural language queries, combining modern RAG techniques with domain-specific Norwegian legal data.
