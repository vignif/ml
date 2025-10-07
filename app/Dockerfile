FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

# Install only runtime libs needed for Pillow (jpeg/png). Avoid build-essential to keep image slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libjpeg-dev \
       zlib1g-dev \
       libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BASE_PATH="" \
    TORCH_HOME=/app/.cache/torch

# Copy requirements first for better build cache utilization
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && gunicorn --version

# Copy source code and assets
COPY src ./src
COPY static ./static
COPY templates ./templates

# Create runtime directories
RUN mkdir -p uploads job_state && chmod 755 uploads job_state
VOLUME ["/app/uploads"]

EXPOSE 8000

# Pre-fetch VGG19 weights to avoid first-request latency
RUN python -c "import torchvision.models as m; m.vgg19(weights=m.VGG19_Weights.DEFAULT)"

# Optional simple healthcheck hitting root path (BASE_PATH handled at runtime)
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python - <<'PY' || exit 1\nimport os,urllib.request,sys; base=os.environ.get('BASE_PATH','').rstrip('/');\nurl=f"http://127.0.0.1:8000{base if base else ''}/";\ntry: urllib.request.urlopen(url,timeout=3);\nexcept Exception as e: sys.exit(1)\nPY

# Use gunicorn; fallback to uvicorn only if gunicorn unexpectedly fails.
CMD ["/bin/sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -w ${ST_WORKERS:-1} -b 0.0.0.0:8000 src.server:app --timeout 300 || uvicorn src.server:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300"]