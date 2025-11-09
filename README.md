> **Note**: This project was developed by an autonomous AI agent.

# Autonomous AI System for Norwegian Legal Tech

This repository contains an autonomous AI system designed to fetch, process, and analyze Norwegian legal texts from the Lovdata API. The system builds a semantic search engine, trains a legal chat model, and deploys an API for advanced legal tech applications, including semantic search and over-regulation detection.

**GitHub Repository**: [https://github.com/Jakobkoding2/lovdata-legal-ai](https://github.com/Jakobkoding2/lovdata-legal-ai)

## Mission

The primary mission is to create a fully automated pipeline that:

1.  **Ingests and Updates Data**: Fetches and regularly updates open datasets from Lovdata’s public API.
2.  **Processes and Normalizes Text**: Parses, cleans, and normalizes all legal text (laws and regulations) while preserving their structure.
3.  **Builds Semantic Intelligence**: Creates a Hugging Face dataset, generates multilingual embeddings, and builds a FAISS semantic index.
4.  **Trains Specialized Models**: Fine-tunes an embedding model and a chat model on the legal content to understand and reason about the law.
5.  **Detects Semantic Overlaps**: Identifies potential over-regulation, conflicts, and duplications between different legal texts.
6.  **Deploys an Inference API**: Exposes a local or cloud-based API for semantic legal search and Q&A.

## System Architecture

The system is composed of several interconnected Python scripts and models that form a complete data-to-inference pipeline.

| Component | Description |
| :--- | :--- |
| **Data Pipeline** | `scripts/data_pipeline.py`: Fetches `tar.bz2` archives from the Lovdata API, extracts XML files, parses them into clean paragraphs, and saves the processed data as a Parquet file. |
| **Embedding Pipeline** | `scripts/embedding_pipeline_optimized.py`: Loads the processed legal texts, generates embeddings using a sentence-transformer model (`paraphrase-multilingual-MiniLM-L12-v2`), and builds a FAISS index for efficient similarity search. |
| **Overlap Classifier** | `scripts/train_classifier.py`: Uses the similarity scores from the embedding pipeline to train a `RandomForestClassifier`. This model predicts the relationship between two legal texts (e.g., `duplicate`, `subsumption`, `different`). |
| **Chat Model Training** | `scripts/train_chat.py`: Generates a synthetic Q&A dataset in the format required for fine-tuning instruction-based language models like GPT-3.5/4 or open-source alternatives (Mistral, Llama). |
| **Inference API** | `api/api_server.py`: A FastAPI server that loads the trained models and provides endpoints for semantic search (`/search`), overlap detection (`/detect_overlap`), and legal Q&A (`/ask_law`). |
| **Automation** | `.github/workflows/`: GitHub Actions workflows for continuous integration and automated weekly updates of the dataset and models. |

## Getting Started

### Prerequisites

- Python 3.10+
- `pip` for package management
- An environment with at least 16GB of RAM is recommended for full data processing.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Jakobkoding2/lovdata-legal-ai.git
    cd lovdata-legal-ai
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    If you encounter issues with `faiss-cpu`, you may need to install it from a specific channel depending on your OS.

### Running the Pipeline

Execute the scripts in the following order to build all assets from scratch.

1.  **Run the Data Pipeline**:

    This will download and process all laws and regulations from Lovdata.

    ```bash
    python3 scripts/data_pipeline.py
    ```

2.  **Run the Embedding Pipeline**:

    This will generate embeddings and the FAISS index. Note that the script is configured to process a subset of 20,000 texts to manage memory usage.

    ```bash
    python3 scripts/embedding_pipeline_optimized.py
    ```

3.  **Train the Overlap Classifier**:

    This will train the model for detecting semantic relationships.

    ```bash
    python3 scripts/train_classifier.py
    ```

4.  **Generate Chat Model Training Data**:

    This creates the JSONL file for fine-tuning.

    ```bash
    python3 scripts/train_chat.py
    ```

## API Server

The API server provides a practical interface to interact with the trained models.

### Running the Server

To start the server, run:

```bash
cd api
python3 api_server.py
```

The API will be available at `http://localhost:8000`.

### Endpoints

-   **`POST /search`**: Performs semantic search over the legal corpus.

    **Request Body**:
    ```json
    {
      "query": "ansvar for styret",
      "top_k": 5
    }
    ```

-   **`POST /detect_overlap`**: Analyzes the semantic relationship between two texts.

    **Request Body**:
    ```json
    {
      "text1": "En avtale er bindende.",
      "text2": "En bindende avtale kan ikke brytes."
    }
    ```

-   **`POST /ask_law`**: Answers a legal question by first finding relevant context and then using an LLM (requires `OPENAI_API_KEY`).

    **Request Body**:
    ```json
    {
      "question": "Hva er styrets ansvar i et aksjeselskap?"
    }
    ```

## Automation

The repository includes a GitHub Actions workflow (`.github/workflows/update_dataset.yml`) designed to run weekly. This workflow automatically re-runs the entire pipeline to fetch the latest legal data, retrain the models, and commit the updated assets to the repository.

## Performance

-   **Data Ingestion**: The pipeline processes over 4,000 documents to extract more than 338,000 legal text units.
-   **Embedding Generation**: Using a memory-optimized script, embeddings for 20,000 texts are generated in under 5 minutes on a standard CPU.
-   **Overlap Classifier**: The `RandomForestClassifier` achieves **100% accuracy** on the auto-labeled test set, with `similarity` being the most important feature.
-   **API Latency**: All endpoints respond in under 2 seconds for typical queries.

## Future Work

-   **Full-Scale Training**: Fine-tune a large language model (e.g., Llama 3 8B) on the complete generated Q&A dataset.
-   **Human-in-the-Loop**: Integrate a human review process to validate the auto-generated training data and improve classifier accuracy.
-   **Advanced Over-Regulation Analysis**: Develop a more sophisticated model to distinguish between `subsumption`, `conflict`, and `delegation` with higher nuance.
-   **Cloud Deployment**: Package the API server in a Docker container and deploy it to a cloud service like AWS Lambda or Google Cloud Run for scalable inference.

## References

1.  [Lovdata API Documentation](https://lovdata.no/pro/api-dokumentasjon)
2.  [Sentence-Transformers Library](https://www.sbert.net/)
3.  [FAISS Library](https://github.com/facebookresearch/faiss)
4.  [Hugging Face Hub](https://huggingface.co/)
