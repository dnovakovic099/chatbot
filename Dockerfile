# Use Python 3.11 slim image (has pre-built wheels for most packages)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence (ChromaDB for RAG)
RUN mkdir -p /app/chroma_db

# Expose port (Railway sets PORT env var)
EXPOSE 8000

# Run the application - Railway provides PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
