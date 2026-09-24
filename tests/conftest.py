import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from celery_app import celery_app  # noqa: E402

# Set once, before any test can touch the (lazily created) result backend.
celery_app.conf.update(
    task_always_eager=True, task_store_eager_result=True, result_backend="cache+memory://"
)
