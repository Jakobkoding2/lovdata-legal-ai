FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime environment variables are provided externally (e.g., GitHub Secrets):
# - OPENAI_API_KEY (required)
# - MODEL_PROVIDER (optional base URL)
# - MODEL_NAME (optional override, defaults to gpt-5-mini)

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
