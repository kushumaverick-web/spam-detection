# Build and run SpamGuard AI with Python 3.11-slim for minimal image size and fast startup
FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure trained models are present (runs training script if model files missing)
RUN python -c "import os; os.system('python train_model.py') if not os.path.exists('model/spam_classifier.joblib') else None"

# Healthcheck to verify container status
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

EXPOSE 8000

# Start Uvicorn ASGI server with production concurrency
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 2
