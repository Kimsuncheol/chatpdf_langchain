"""Thin Firestore/Storage adapter. Replace with the project's existing admin-SDK helpers.

Status writes never raise: the GET /jobs/{id}/status endpoint is the fallback,
so a Firestore outage must not fail the document pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

COLLECTION = os.environ.get("FIRESTORE_DOCS_COLLECTION", "documents")


def init_firebase() -> None:
    """Initialise the Admin SDK once per process (called at app/worker startup); idempotent."""
    import firebase_admin

    if not firebase_admin._apps:
        firebase_admin.initialize_app()


def _db():
    from firebase_admin import firestore

    init_firebase()  # no-op after startup; keeps Celery workers self-sufficient
    return firestore.client()


def update_status(doc_id: str, stage: str, percent: int, error: Optional[str] = None) -> None:
    fields = {"processingStage": stage, "progressPercent": percent}
    if error is not None:
        fields.update({"status": "error", "errorMessage": error})
    try:
        _db().collection(COLLECTION).document(doc_id).set(fields, merge=True)
    except Exception:
        logger.exception("Firestore status write failed for %s (%s)", doc_id, stage)


def download_pdf(storage_path: str, dest: str) -> None:
    from firebase_admin import storage

    _db()  # ensures the app is initialised
    storage.bucket(os.environ.get("STORAGE_BUCKET")).blob(storage_path).download_to_filename(dest)
