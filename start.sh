#!/bin/sh
# Runs the Celery worker in the background and the API in the foreground.
celery -A celery_app.celery_app worker --loglevel=info --concurrency=1 &
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
