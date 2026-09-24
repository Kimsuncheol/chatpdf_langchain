import logging
from contextlib import asynccontextmanager

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi.errors import RateLimitExceeded

import firebase_io
from celery_app import celery_app
from chat import router as chat_router
from ratelimit import ANALYZE_LIMIT, limiter, rate_limit_handler
from security import get_current_user
from tasks import process_document_task

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    firebase_io.init_firebase()  # once at startup, not per request
    yield


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.include_router(chat_router, dependencies=[Depends(get_current_user)])


@app.get("/health")
def health():
    return {"status": "ok"}  # the only unauthenticated route (besides FastAPI's /docs)


@app.get("/", dependencies=[Depends(get_current_user)])
def read_root():
    return {"message": "Hello, FastAPI!"}


class AnalyzeRequest(BaseModel):
    docId: str = Field(min_length=1)
    storagePath: str = Field(min_length=1)


@app.post("/analyze", status_code=202, dependencies=[Depends(get_current_user)])
@limiter.limit(ANALYZE_LIMIT)  # per uid
def analyze(request: Request, req: AnalyzeRequest):
    # Never parse here: always hand off to the worker.
    try:
        task = process_document_task.delay(req.docId, req.storagePath)
    except Exception:
        logger.exception("Could not enqueue task for %s", req.docId)
        raise HTTPException(status_code=503, detail="Job queue unavailable")
    return {"jobId": task.id, "status": "queued"}


@app.get("/jobs/{job_id}/status", dependencies=[Depends(get_current_user)])
def job_status(job_id: str):
    result = AsyncResult(job_id, app=celery_app)
    state = result.state
    if state == "FAILURE":
        # Only PipelineError messages are stored (see tasks.py); never expose result.traceback.
        return {"state": state, "error": str(result.result)}
    body = {"state": state}
    if state == "STARTED":
        info = result.info if isinstance(result.info, dict) else {}
        body["progressPercent"] = info.get("progressPercent", 0)
    elif state == "SUCCESS":
        body["progressPercent"] = 100
    return body
