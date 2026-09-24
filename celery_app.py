import os

from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("chatpdf", broker=REDIS_URL, backend=REDIS_URL, include=["tasks"])
celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    result_expires=24 * 3600,
)
