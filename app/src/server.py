import asyncio
import os
import time
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .style_transfer import stylize_with_progress

# Allow serving under a path prefix (e.g. /ml) behind Caddy or another reverse proxy.
# In Caddy you will typically:
#   handle /ml/* {
#       uri strip_prefix /ml
#       reverse_proxy 127.0.0.1:8006 {
#           header_up X-Forwarded-Prefix /ml
#       }
#   }
# Set BASE_PATH=/ml in the service environment so generated links include the prefix.
BASE_PATH = os.getenv("BASE_PATH", "").rstrip("/")
if BASE_PATH and not BASE_PATH.startswith("/"):
        BASE_PATH = "/" + BASE_PATH

# We avoid setting root_path because we support two deployment modes:
# 1. Reverse proxy that strips the prefix (passes X-Forwarded-Prefix header)
# 2. Direct access where the prefix is kept (BASE_PATH set and no strip)
# Using root_path plus manual prefix handling can produce double-prefix static issues.
app = FastAPI(title="Neural Style Transfer")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
JOBS_DIR = BASE_DIR / "job_state"
JOBS_DIR.mkdir(exist_ok=True)
PERSIST_JOBS = os.getenv("ST_PERSIST_JOBS", "1") not in {"0", "false", "False"}
PERSIST_STEP_INTERVAL = int(os.getenv("ST_PERSIST_STEP_INTERVAL", "5"))

static_dir = StaticFiles(directory=str(BASE_DIR / "static"))
# Always mount base /static (works when proxy strips prefix)
app.mount("/static", static_dir, name="static")
# Also mount prefixed static if BASE_PATH is provided and requests come without stripping
if BASE_PATH:
    app.mount(f"{BASE_PATH}/static", static_dir, name="static_prefixed")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

def _determine_prefix(request: Request) -> str:
    # Priority: X-Forwarded-Prefix header (from proxy) else BASE_PATH env else empty
    hdr = request.headers.get("x-forwarded-prefix") or request.headers.get("x-forwarded-path")
    prefix = hdr or BASE_PATH or ""
    if prefix and not prefix.startswith("/"):
        prefix = "/" + prefix
    return prefix.rstrip("/")


@app.get("/", response_class=HTMLResponse)
async def index_root(request: Request):
    prefix = _determine_prefix(request)
    return templates.TemplateResponse("index.html", {"request": request, "prefix": prefix})

if BASE_PATH:
    @app.get(f"{BASE_PATH}/", response_class=HTMLResponse)
    async def index_prefixed(request: Request):  # type: ignore
        prefix = _determine_prefix(request)
        return templates.TemplateResponse("index.html", {"request": request, "prefix": prefix})


_JOBS: dict[str, dict[str, Any]] = {}
_JOB_LOCK = asyncio.Lock()
_JOB_QUEUE: "asyncio.Queue[tuple[str, Path, Path, int, float]]" = asyncio.Queue()
WORKER_COUNT = int(os.getenv("ST_WORKERS", "1"))

# ---------------- Persistence Helpers ---------------- #

def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id

def _job_meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "meta.json"

def _job_result_path(job_id: str) -> Path:
    return _job_dir(job_id) / "result.jpg"

def _job_preview_path(job_id: str) -> Path:
    return _job_dir(job_id) / "preview.jpg"

def _persist_job(job_id: str):
    if not PERSIST_JOBS:
        return
    job = _JOBS.get(job_id)
    if not job:
        return
    d = _job_dir(job_id)
    d.mkdir(exist_ok=True, parents=True)
    data = {k: v for k, v in job.items() if k not in {"result", "preview"}}
    tmp = d / "meta.tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f)
    tmp.rename(_job_meta_path(job_id))
    if 'preview' in job:
        with open(_job_preview_path(job_id), 'wb') as pf:
            pf.write(job['preview'])

def _load_job_meta(job_id: str) -> Optional[dict]:
    path = _job_meta_path(job_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def _ensure_job_loaded(job_id: str) -> Optional[dict]:
    job = _JOBS.get(job_id)
    if job:
        return job
    meta = _load_job_meta(job_id)
    if not meta:
        return None
    # If meta says running/queued but process restarted, mark lost.
    if meta.get('status') in {'queued', 'running'}:
        meta['status'] = 'lost'
        _JOBS[job_id] = meta
        _persist_job(job_id)
        return meta
    _JOBS[job_id] = meta
    return meta

async def _save_upload(file: UploadFile, path: Path):
    data = await file.read()
    with open(path, 'wb') as f:
        f.write(data)

def _run_style_job(job_id: str, content_path: Path, style_path: Path, steps: int, style_weight: float):
    import time
    job = _JOBS[job_id]
    start = time.time()
    last_step_time = None

    def cb(info):
        nonlocal last_step_time
        now = time.time()
        if last_step_time is not None:
            job['avg_step_seconds'] = (job.get('avg_step_seconds', 0) * (info['step']-2) + (now - last_step_time)) / max(1, info['step']-1)
        last_step_time = now
        remaining = info['total_steps'] - info['step']
        eta = remaining * job.get('avg_step_seconds', 0) if 'avg_step_seconds' in job else None
        if '_preview_image' in info:
            # Store only a JPEG thumbnail to reduce memory
            buf_prev = BytesIO()
            info['_preview_image'].save(buf_prev, format='JPEG', quality=70)
            buf_prev.seek(0)
            job['preview'] = buf_prev.getvalue()
        job.update({
            'status': 'running',
            'step': info['step'],
            'total_steps': info['total_steps'],
            'content_loss': info['content_loss'],
            'style_loss': info['style_loss'],
            'total_loss': info['total_loss'],
            'elapsed_seconds': info['elapsed_seconds'],
            'eta_seconds': eta,
            'preview_step': info.get('preview_step'),
        })
        if PERSIST_JOBS and (info['step'] % PERSIST_STEP_INTERVAL == 0 or info.get('preview_step')):
            _persist_job(job_id)
    try:
        img = stylize_with_progress(
            content_path,
            style_path,
            steps=steps,
            style_weight=style_weight,
            progress_cb=cb,
            callback_every=1,
            should_cancel=lambda: _JOBS.get(job_id, {}).get('status') == 'cancelling',
            preview_every=25,
        )
        if _JOBS.get(job_id, {}).get('status') == 'cancelling':
            job['status'] = 'cancelled'
            _persist_job(job_id)
            return
        buf = BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        job['result'] = buf
        job['status'] = 'finished'
        job['elapsed_total'] = time.time() - start
        if PERSIST_JOBS:
            with open(_job_result_path(job_id), 'wb') as rf:
                rf.write(buf.getvalue())
            _persist_job(job_id)
    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)
        _persist_job(job_id)
    finally:
        try:
            content_path.unlink(missing_ok=True)
            style_path.unlink(missing_ok=True)
        except Exception:
            pass


async def _job_worker():
    while True:
        spec = await _JOB_QUEUE.get()
        if spec is None:  # shutdown signal
            _JOB_QUEUE.task_done()
            break
        job_id, content_path, style_path, steps, style_weight = spec
        # Execute heavy work in thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, _run_style_job, job_id, content_path, style_path, steps, style_weight)
        finally:
            _JOB_QUEUE.task_done()


@app.on_event("startup")
async def _startup():
    # Launch worker coroutines
    for _ in range(max(1, WORKER_COUNT)):
        asyncio.create_task(_job_worker())

@app.post("/stylize")
async def enqueue_stylize(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    steps: int = Form(200),
    style_weight: float = Form(1_000_000.0),
):
    timestamp = int(time.time()*1000)
    content_path = UPLOAD_DIR / f"content_{timestamp}.png"
    style_path = UPLOAD_DIR / f"style_{timestamp}.png"
    await _save_upload(content_image, content_path)
    await _save_upload(style_image, style_path)

    import uuid
    job_id = uuid.uuid4().hex
    async with _JOB_LOCK:
        _JOBS[job_id] = {
            'status': 'queued',
            'step': 0,
            'total_steps': steps,
            'style_weight': style_weight,
            'created_ts': time.time(),
        }
        _persist_job(job_id)
    await _JOB_QUEUE.put((job_id, content_path, style_path, steps, style_weight))
    return {"job_id": job_id}

@app.get("/stylize/{job_id}")
async def get_job(job_id: str):
    job = _ensure_job_loaded(job_id)
    if not job:
        return HTMLResponse(status_code=404, content="Job not found")
    if job.get('status') == 'finished':
        # Return image
        if 'result' in job:
            buf: BytesIO = job['result']
            return StreamingResponse(BytesIO(buf.getvalue()), media_type='image/jpeg')
        # load from disk
        rp = _job_result_path(job_id)
        if rp.exists():
            return StreamingResponse(open(rp, 'rb'), media_type='image/jpeg')
    # If preview available and client wants it, we can embed base64 or separate endpoint.
    resp = {k: v for k, v in job.items() if k not in {'result', 'preview'} and not isinstance(v, (bytes, bytearray))}
    resp['preview_available'] = 'preview' in job or _job_preview_path(job_id).exists()
    return resp

@app.get("/stylize/{job_id}/preview")
async def get_preview(job_id: str):
    job = _ensure_job_loaded(job_id)
    if not job:
        return HTMLResponse(status_code=404, content="Job not found")
    if 'preview' not in job:
        pp = _job_preview_path(job_id)
        if pp.exists():
            return StreamingResponse(open(pp, 'rb'), media_type='image/jpeg')
        return HTMLResponse(status_code=204, content="")
    return StreamingResponse(BytesIO(job['preview']), media_type='image/jpeg')

@app.post("/stylize/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = _ensure_job_loaded(job_id)
    if not job:
        return HTMLResponse(status_code=404, content="Job not found")
    if job.get('status') in { 'finished', 'error', 'cancelled' }:
        return { 'status': job['status'] }
    job['status'] = 'cancelling'
    _persist_job(job_id)
    return { 'status': 'cancelling' }

if BASE_PATH:
    @app.post(f"{BASE_PATH}/stylize")
    async def enqueue_stylize_prefixed(
        content_image: UploadFile = File(...),
        style_image: UploadFile = File(...),
        steps: int = Form(200),
        style_weight: float = Form(1_000_000.0),
    ):  # type: ignore
        return await enqueue_stylize(content_image, style_image, steps, style_weight)

    @app.get(f"{BASE_PATH}/stylize/{{job_id}}")
    async def get_job_prefixed(job_id: str):  # type: ignore
        return await get_job(job_id)



# Uvicorn entrypoint helper
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
