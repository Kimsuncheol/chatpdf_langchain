import logging

import firebase_io
from celery_app import celery_app
from pipeline import PipelineError, run_pipeline

logger = logging.getLogger(__name__)

FIRESTORE_STEP_PERCENT = 5  # throttle intra-phase Firestore writes


@celery_app.task(bind=True, name="process_document_task")
def process_document_task(self, docId: str, storagePath: str) -> dict:
    last = {"stage": None, "percent": -100}

    def report(stage: str, percent: int) -> None:
        self.update_state(state="STARTED", meta={"stage": stage, "progressPercent": percent})
        if stage != last["stage"] or percent - last["percent"] >= FIRESTORE_STEP_PERCENT:
            firebase_io.update_status(docId, stage, percent)
            last.update(stage=stage, percent=percent)

    try:
        run_pipeline(docId, storagePath, report)
    except Exception as e:
        logger.exception("process_document_task failed for docId=%s", docId)  # full trace, server-side only
        message = str(e) if isinstance(e, PipelineError) else "Processing failed"
        firebase_io.update_status(docId, "error", 0, error=message)
        raise PipelineError(message) from None
    return {"docId": docId}
