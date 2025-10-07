FROM python:3.11-slim AS base
ARG DEBIAN_FRONTEND=noninteractive

# Minimal runtime libs for Pillow & torchvision image ops
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libjpeg-dev libpng-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BASE_PATH="" \
    TORCH_HOME=/app/.cache/torch

WORKDIR /app

# Build args / toggles
ARG TORCH_CPU_ONLY=1
ARG PRELOAD_WEIGHTS=0

# Copy requirements early
COPY requirements.txt ./requirements.txt

# Separate torch install path to allow better layer caching & optional CPU-only slim wheels
RUN set -eux; \
    pip install --no-cache-dir --upgrade pip setuptools wheel; \
    # Filter out torch & torchvision from requirements (will install explicitly next)
    grep -vi '^torchvision==' requirements.txt | grep -vi '^torch==' > req.no-torch.txt; \
    pip install --no-cache-dir -r req.no-torch.txt; \
    if [ "$TORCH_CPU_ONLY" = "1" ]; then \
        pip install --no-cache-dir torch==2.3.1+cpu torchvision==0.18.1+cpu --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1; \
    fi; \
    gunicorn --version; \
    python - <<'PY' || true\nimport torch,os,platform\nprint('Torch build:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'Device count:', torch.cuda.device_count())\nPY

# Copy source code
COPY src ./src
COPY static ./static
COPY templates ./templates

RUN mkdir -p uploads job_state && chmod 755 uploads job_state
VOLUME ["/app/uploads"]

EXPOSE 8000

# Optionally pre-download VGG19 weights (skipped if PRELOAD_WEIGHTS=0 to save build space/time)
RUN if [ "$PRELOAD_WEIGHTS" = "1" ]; then python -c "import torchvision.models as m; m.vgg19(weights=m.VGG19_Weights.DEFAULT)"; fi

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python - <<'PY' || exit 1\nimport os,urllib.request,sys\nbase=os.environ.get('BASE_PATH','').rstrip('/')\nurl=f"http://127.0.0.1:8000{base if base else ''}/"\ntry: urllib.request.urlopen(url,timeout=3)\nexcept Exception: sys.exit(1)\nPY

CMD ["/bin/sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -w ${ST_WORKERS:-1} -b 0.0.0.0:8000 src.server:app --timeout 300 || uvicorn src.server:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300"]