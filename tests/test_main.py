import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.main import app  # safe at import time; lifespan only runs inside TestClient context


async def _fake_astream_ok(*args, **kwargs):
    yield AIMessageChunk(content="Hello")
    yield AIMessageChunk(content=" world")


async def _fake_astream_empty(*args, **kwargs):
    yield AIMessageChunk(content="")
    yield AIMessageChunk(content="text")


async def _fake_astream_error(*args, **kwargs):
    raise RuntimeError("Gemini timeout")
    yield  # marks this as an async generator so the exception is raised during iteration


@pytest.fixture
def client():
    mock_chain = MagicMock()
    mock_chain.astream = _fake_astream_ok

    with patch("app.main.os.path.exists", return_value=True), \
         patch("app.main.build_chain", return_value=mock_chain):
        with TestClient(app) as c:
            yield c, mock_chain


def test_health_returns_ok(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_streams_tokens(client):
    c, _ = client
    with c.stream("POST", "/api/chat", json={"session_id": "s1", "question": "What is AES?"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [line for line in resp.iter_lines() if line]

    assert 'data: {"token": "Hello"}' in lines
    assert 'data: {"token": " world"}' in lines
    assert "data: [DONE]" in lines


def test_chat_skips_empty_content_chunks(client):
    c, mock_chain = client
    mock_chain.astream = _fake_astream_empty

    with c.stream("POST", "/api/chat", json={"session_id": "s2", "question": "Test"}) as resp:
        lines = [line for line in resp.iter_lines() if line]

    assert 'data: {"token": ""}' not in lines
    assert 'data: {"token": "text"}' in lines


def test_chat_streams_error_on_exception(client):
    c, mock_chain = client
    mock_chain.astream = _fake_astream_error

    with c.stream("POST", "/api/chat", json={"session_id": "s3", "question": "Test"}) as resp:
        lines = [line for line in resp.iter_lines() if line]

    assert any('"error"' in line for line in lines)


def test_chat_empty_question_returns_422(client):
    c, _ = client
    resp = c.post("/api/chat", json={"session_id": "s1", "question": ""})
    assert resp.status_code == 422


def test_clear_existing_session(client):
    c, _ = client
    import app.session as s
    s.get_session_history("to-clear")
    resp = c.delete("/api/chat/to-clear")
    assert resp.status_code == 200
    assert resp.json() == {"session_id": "to-clear", "cleared": True}


def test_clear_nonexistent_session(client):
    c, _ = client
    resp = c.delete("/api/chat/nonexistent-999")
    assert resp.status_code == 200
    assert resp.json() == {"session_id": "nonexistent-999", "cleared": False}


def test_startup_fails_without_chroma_db():
    with patch("app.main.os.path.exists", return_value=False):
        with pytest.raises(RuntimeError, match="Run: python ingest.py"):
            with TestClient(app):
                pass
