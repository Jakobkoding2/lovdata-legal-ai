# Deployment Guide

This guide provides step-by-step instructions for deploying the Lovdata Legal AI system in various environments.

## Table of Contents

1.  [Local Deployment](#local-deployment)
2.  [Docker Deployment](#docker-deployment)
3.  [Cloud Deployment](#cloud-deployment)
4.  [Hugging Face Deployment](#hugging-face-deployment)
5.  [Production Considerations](#production-considerations)

## Local Deployment

### Prerequisites

-   Python 3.10+
-   16GB+ RAM (recommended)
-   10GB+ disk space

### Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Jakobkoding2/lovdata-legal-ai.git
    cd lovdata-legal-ai
    ```

2.  **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the data pipeline (optional - pre-processed data is included):**

    ```bash
    python3 scripts/data_pipeline.py
    python3 scripts/embedding_pipeline_optimized.py
    python3 scripts/train_classifier.py
    python3 scripts/train_chat.py
    ```

5.  **Start the API server:**

    ```bash
    cd api
    python3 api_server.py
    ```

6.  **Access the API:**

    The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

## Docker Deployment

### Create Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the API server
CMD ["python3", "api/api_server.py"]
```

### Build and Run

```bash
# Build the Docker image
docker build -t lovdata-legal-ai .

# Run the container
docker run -p 8000:8000 lovdata-legal-ai
```

### Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

## Cloud Deployment

### AWS Lambda + API Gateway

1.  **Package the application:**

    ```bash
    pip install -t package -r requirements.txt
    cd package
    zip -r ../deployment-package.zip .
    cd ..
    zip -g deployment-package.zip api/api_server.py
    ```

2.  **Create a Lambda function:**

    Use the AWS Console or CLI to create a Lambda function with the deployment package.

3.  **Configure API Gateway:**

    Set up an HTTP API in API Gateway to trigger the Lambda function.

### Google Cloud Run

1.  **Build and push the Docker image:**

    ```bash
    gcloud builds submit --tag gcr.io/PROJECT_ID/lovdata-legal-ai
    ```

2.  **Deploy to Cloud Run:**

    ```bash
    gcloud run deploy lovdata-legal-ai \
      --image gcr.io/PROJECT_ID/lovdata-legal-ai \
      --platform managed \
      --region us-central1 \
      --allow-unauthenticated
    ```

### Azure Container Instances

1.  **Build and push to Azure Container Registry:**

    ```bash
    az acr build --registry REGISTRY_NAME --image lovdata-legal-ai .
    ```

2.  **Deploy to ACI:**

    ```bash
    az container create \
      --resource-group RESOURCE_GROUP \
      --name lovdata-legal-ai \
      --image REGISTRY_NAME.azurecr.io/lovdata-legal-ai \
      --dns-name-label lovdata-legal-ai \
      --ports 8000
    ```

## Hugging Face Deployment

### Upload Models to Hugging Face Hub

1.  **Install Hugging Face CLI:**

    ```bash
    pip install huggingface-hub
    huggingface-cli login
    ```

2.  **Create a new model repository:**

    ```bash
    huggingface-cli repo create lovdata-legal-ai --type model
    ```

3.  **Upload embeddings and index:**

    ```bash
    huggingface-cli upload lovdata-legal-ai models/lovdata_embeddings.npy
    huggingface-cli upload lovdata-legal-ai models/lovdata_faiss.index
    huggingface-cli upload lovdata-legal-ai models/overlap_classifier.joblib
    ```

4.  **Create a Hugging Face Space:**

    Create a new Space with Gradio or Streamlit to provide a web interface for the API.

### Example Gradio Interface

```python
import gradio as gr
import requests

API_URL = "http://localhost:8000"

def search_law(query, top_k):
    response = requests.post(f"{API_URL}/search", json={"query": query, "top_k": top_k})
    return response.json()

def ask_question(question):
    response = requests.post(f"{API_URL}/ask_law", json={"question": question})
    return response.json()

with gr.Blocks() as demo:
    gr.Markdown("# Lovdata Legal AI")
    
    with gr.Tab("Semantic Search"):
        query_input = gr.Textbox(label="Search Query")
        top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Number of Results")
        search_button = gr.Button("Search")
        search_output = gr.JSON(label="Results")
        search_button.click(search_law, inputs=[query_input, top_k_slider], outputs=search_output)
    
    with gr.Tab("Ask a Question"):
        question_input = gr.Textbox(label="Legal Question")
        ask_button = gr.Button("Ask")
        answer_output = gr.JSON(label="Answer")
        ask_button.click(ask_question, inputs=question_input, outputs=answer_output)

demo.launch()
```

## Production Considerations

### Performance Optimization

-   **Use GPU for embeddings**: If deploying on a GPU-enabled instance, modify the embedding pipeline to use CUDA.
-   **Increase batch size**: For faster processing, increase the batch size in the embedding pipeline.
-   **Use HNSW index**: For larger datasets (>100K vectors), use the HNSW index for approximate nearest neighbor search.

### Security

-   **API Authentication**: Add authentication to the API endpoints using OAuth2 or API keys.
-   **Rate Limiting**: Implement rate limiting to prevent abuse.
-   **HTTPS**: Always use HTTPS in production.

### Monitoring

-   **Logging**: Use structured logging (e.g., JSON logs) and send them to a centralized logging service (e.g., CloudWatch, Stackdriver).
-   **Metrics**: Track API latency, error rates, and throughput using Prometheus or similar tools.
-   **Alerts**: Set up alerts for high error rates or slow response times.

### Scaling

-   **Horizontal Scaling**: Deploy multiple instances of the API server behind a load balancer.
-   **Caching**: Cache frequently requested results using Redis or Memcached.
-   **Database**: Store the corpus and embeddings in a database (e.g., PostgreSQL with pgvector extension) for better scalability.

### Compliance

-   **Data Protection**: Ensure compliance with Norwegian data protection laws (GDPR).
-   **Legal Review**: Have legal experts review the system's outputs before using them in production.
-   **Audit Logs**: Maintain audit logs of all API requests for compliance purposes.

## Troubleshooting

### Common Issues

1.  **Out of Memory Error**:
    -   Reduce the `MAX_TEXTS` parameter in `embedding_pipeline_optimized.py`.
    -   Use a machine with more RAM.

2.  **FAISS Index Not Loading**:
    -   Ensure the index file exists in `models/lovdata_faiss.index`.
    -   Rebuild the index by running `python3 scripts/embedding_pipeline_optimized.py`.

3.  **API Server Not Starting**:
    -   Check the logs in `api/api_server.log`.
    -   Ensure all dependencies are installed.

4.  **Slow Search Performance**:
    -   Use a GPU for embedding generation.
    -   Switch to the HNSW index for approximate search.

## Support

For issues, questions, or contributions, please open an issue on the [GitHub repository](https://github.com/Jakobkoding2/lovdata-legal-ai).
