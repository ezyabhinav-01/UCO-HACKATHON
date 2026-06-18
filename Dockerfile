# ============================================================
# PhaseGuard - Layer 2 Backend (FastAPI + SpeechBrain ECAPA-TDNN)
# ============================================================
FROM python:3.10-slim

# System dependencies required by torchaudio / librosa / soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY .env.example .env

# Runtime directories
RUN mkdir -p /app/data/temp_uploads /app/logs /app/pretrained_models/ecapa

EXPOSE 8000

# Healthcheck hits the FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
