import pytest
from pydantic import ValidationError
from app.schemas import ChatRequest, ClearResponse


def test_chat_request_valid():
    req = ChatRequest(session_id="abc-123", question="What is AES?")
    assert req.session_id == "abc-123"
    assert req.question == "What is AES?"


def test_chat_request_empty_question_raises():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="abc-123", question="")


def test_chat_request_whitespace_question_raises():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="abc-123", question="   ")


def test_chat_request_missing_session_id_raises():
    with pytest.raises(ValidationError):
        ChatRequest(question="What is AES?")


def test_chat_request_missing_question_raises():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="abc-123")


def test_clear_response_cleared_true():
    resp = ClearResponse(session_id="abc-123", cleared=True)
    assert resp.session_id == "abc-123"
    assert resp.cleared is True


def test_clear_response_cleared_false():
    resp = ClearResponse(session_id="abc-123", cleared=False)
    assert resp.cleared is False
