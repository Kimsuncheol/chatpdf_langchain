import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ["RATELIMIT_STORAGE_URI"] = "memory://"

import pytest  # noqa: E402
from celery_app import celery_app  # noqa: E402

# Set once, before any test can touch the (lazily created) result backend.
celery_app.conf.update(
    task_always_eager=True, task_store_eager_result=True, result_backend="cache+memory://"
)


@pytest.fixture
def client(monkeypatch):
    """TestClient with lifespan; tokens are their own uid, "bad"/"expired" are rejected."""
    from fastapi.testclient import TestClient
    from firebase_admin import auth

    import firebase_io
    import main
    from ratelimit import limiter

    def verify(token):
        if token == "expired":
            raise auth.ExpiredIdTokenError("expired", cause=None)
        if token == "bad":
            raise auth.InvalidIdTokenError("bad")
        return {"uid": token}

    inits = []
    monkeypatch.setattr(firebase_io, "init_firebase", lambda: inits.append(1))
    monkeypatch.setattr(auth, "verify_id_token", verify)
    limiter.reset()
    with TestClient(main.app, headers={"Authorization": "Bearer user-a"}) as c:
        c.inits = inits
        yield c
    main.app.dependency_overrides.clear()
