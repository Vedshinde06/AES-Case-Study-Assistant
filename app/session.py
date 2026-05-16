from langchain_community.chat_message_histories import ChatMessageHistory

_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def clear_session(session_id: str) -> bool:
    if session_id in _store:
        del _store[session_id]
        return True
    return False
