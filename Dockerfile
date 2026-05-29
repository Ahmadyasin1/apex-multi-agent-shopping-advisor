# ─────────────────────────────────────────────────────────────────────────────
# APEX — Intelligent Shopping Advisor
# Multi-stage Docker build for Railway / Render / Fly.io deployment
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies needed by faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source (excludes files in .dockerignore)
COPY . .

# Pre-build the FAISS index so the first request is fast
RUN python -c "
import os, sys
sys.path.insert(0, '.')
os.environ.setdefault('GEMINI_API_KEY', 'placeholder')
from data_layer.vector_db import VectorDBStore
db = VectorDBStore(
    data_path='data_layer/dataset.json',
    index_file='data_layer/products.index',
    metadata_file='data_layer/metadata.json'
)
db.load_and_index()
print('FAISS index built successfully')
" || echo "FAISS pre-build skipped — will build on first request"

# Expose the application port
EXPOSE 8000

# Start uvicorn (single worker — stateful FAISS index lives in memory)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
