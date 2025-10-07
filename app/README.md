# Neural Style Transfer FastAPI App

A minimal FastAPI web application to perform neural style transfer using a VGG19-based optimization loop. Suitable for a demo deployment on a small VPS (with CPU it will be slow; GPU recommended for faster results).

## Features
- Upload content + style images via web form.
- Adjustable steps and style weight.
- Returns stylized image directly.
- Simple, dependency-light (no database).

## Tech Stack
- Python, FastAPI, Uvicorn
- PyTorch + TorchVision (VGG19 features)
- Frontend: vanilla HTML/CSS/JS

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # (macOS/Linux)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If running on a CPU-only VPS you may want to install the CPU wheels for torch explicitly (see https://pytorch.org for the right command). The pinned versions here target recent PyTorch.

## Run (Dev)
```bash
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
```
Open: http://localhost:8000

## Production Suggestions
Use `gunicorn` with Uvicorn workers behind Nginx:
```bash
pip install gunicorn
GUNICORN_CMD_ARGS="--timeout 180" gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8000 src.server:app
```
Add Nginx reverse proxy for TLS and static caching.

### Caddy Reverse Proxy Under a Subpath (/ml)
If you serve multiple apps on one domain and want this app at `https://apps.example.com/ml/`:

1. Run the app on an internal port (e.g. 8006):
```bash
BASE_PATH=/ml GUNICORN_CMD_ARGS="--timeout 300" gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 127.0.0.1:8006 src.server:app
```

2. In your Caddyfile:
```
apps.example.com {
	handle /ml/* {
		uri strip_prefix /ml
		reverse_proxy 127.0.0.1:8006 {
			header_up X-Forwarded-Prefix /ml
		}
	}
}
```

3. All internal links/scripts are generated using BASE_PATH so the form posts to `/ml/stylize` and static assets load from `/ml/static/style.css`.

Note: If you use `systemd`, set `Environment=BASE_PATH=/ml` in the service file.

## Docker
Build:
```bash
docker build -t style-transfer-app:latest .
```

Run (root path):
```bash
docker run --rm -p 8000:8000 style-transfer-app:latest
```

Run under a base path (/ml):
```bash
docker run --rm -e BASE_PATH=/ml -p 8000:8000 style-transfer-app:latest
```

Bind a host volume for uploads (optional temporary files):
```bash
docker run --rm -p 8000:8000 -v "$PWD/uploads":/app/uploads style-transfer-app:latest
```

Pre-pulling VGG19 weights occurs at build; to disable remove the corresponding RUN line in `Dockerfile`.

For heavier traffic or faster inference, replace the iterative optimization with a pre-trained fast style transfer network (e.g. Johnson et al. 2016) and precompute multiple style models.

## Environment Variables
None required currently.

## Notes
- Iterative optimization (default 200 steps) on CPU may take several minutes. Reduce steps or deploy a GPU instance.
- The current implementation processes one request fully in memory. For concurrency, you could queue jobs (e.g., with Celery or asyncio task queue) and stream progress via Server-Sent Events or WebSockets.

## License
MIT (add a LICENSE file if you want explicit licensing).
