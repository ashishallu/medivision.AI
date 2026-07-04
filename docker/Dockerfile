# Dockerfile
# Builds the MediVision AI Streamlit application container.
# Ollama runs as a separate service (see docker-compose.yml) so the
# LLM stays swappable and doesn't bloat this image.

FROM python:3.11-slim

WORKDIR /app

# System deps required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY python/ python/
COPY data/clinical_guidelines/ data/clinical_guidelines/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "python/app/streamlit_app.py", "--server.address=0.0.0.0"]
