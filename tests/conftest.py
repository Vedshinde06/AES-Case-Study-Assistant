import os
import pytest
import app.session as session_module

os.environ.setdefault("GOOGLE_API_KEY", "test-key-for-testing")
os.environ.setdefault("HF_TOKEN", "test-hf-token-for-testing")


@pytest.fixture(autouse=True)
def clear_session_store():
    session_module._store.clear()
    yield
    session_module._store.clear()
