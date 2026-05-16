from app.session import get_session_history, clear_session, _store


def test_get_session_history_creates_new_entry():
    history = get_session_history("session-001")
    assert "session-001" in _store
    assert history is _store["session-001"]


def test_get_session_history_returns_same_object():
    h1 = get_session_history("session-002")
    h2 = get_session_history("session-002")
    assert h1 is h2


def test_clear_existing_session_returns_true():
    get_session_history("session-003")
    result = clear_session("session-003")
    assert result is True
    assert "session-003" not in _store


def test_clear_nonexistent_session_returns_false():
    result = clear_session("nonexistent-999")
    assert result is False


def test_session_store_is_isolated_between_tests():
    # conftest.autouse fixture clears _store; this should find it empty
    assert len(_store) == 0
