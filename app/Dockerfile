FROM python:3.11-slim

# Install system deps (optional: libjpeg for Pillow performance)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    which gunicorn && gunicorn --version

# Copy source
COPY src ./src
COPY static ./static
COPY templates ./templates

# Create uploads directory (volume mount optional)
RUN mkdir -p uploads
VOLUME ["/app/uploads"]

ENV PYTHONUNBUFFERED=1 \
    BASE_PATH="" \
    TORCH_HOME=/app/.cache/torch

EXPOSE 8000

# Download VGG19 weights at build (optional; speeds first request)
RUN python -c "import torchvision.models as m; m.vgg19(weights=m.VGG19_Weights.DEFAULT)"

CMD ["/bin/sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8000 src.server:app --timeout 300 || uvicorn src.server:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300"]