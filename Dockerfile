# Python 3.11 slim base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for sentence-transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8000

# Run the app
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}