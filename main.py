import logging

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from celery_app import celery_app
from tasks import process_document_task

logger = logging.getLogger(__name__)
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


class AnalyzeRequest(BaseModel):
    docId: str = Field(min_length=1)
    storagePath: str = Field(min_length=1)


@app.post("/analyze", status_code=202)
def analyze(req: AnalyzeRequest):
    # Never parse here: always hand off to the worker.
    try:
        task = process_document_task.delay(req.docId, req.storagePath)
    except Exception:
        logger.exception("Could not enqueue task for %s", req.docId)
        raise HTTPException(status_code=503, detail="Job queue unavailable")
    return {"jobId": task.id, "status": "queued"}


@app.get("/jobs/{job_id}/status")
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
