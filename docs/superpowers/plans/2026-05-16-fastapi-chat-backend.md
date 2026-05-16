# FastAPI Chat Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a streaming FastAPI RAG chat backend over AES engineering PDF case studies using LangChain LCEL, ChromaDB, and Google Gemini.

**Architecture:** A separate `ingest.py` script loads PDFs, chunks them, and persists a ChromaDB vector index. The FastAPI server loads that index on startup and exposes a streaming SSE endpoint. Conversation history is kept per session in an in-memory dict using LangChain's `RunnableWithMessageHistory`.

**Tech Stack:** Python, FastAPI, LangChain LCEL, `langchain-chroma`, `langchain-google-genai`, `sse-starlette`, ChromaDB, Google Gemini (`gemini-2.0-flash`), pytest, httpx

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `requirements.txt` | Add fastapi, uvicorn, sse-starlette, pytest, httpx |
| Create | `app/__init__.py` | Package marker |
| Create | `app/schemas.py` | Pydantic request/response models |
| Create | `app/session.py` | In-memory session store |
| Create | `app/rag.py` | LCEL chain builder |
| Create | `app/main.py` | FastAPI app, lifespan, endpoints |
| Create | `ingest.py` | One-shot PDF ingestion script |
| Create | `tests/__init__.py` | Test package marker |
| Create | `tests/conftest.py` | Shared fixtures, env setup |
| Create | `tests/test_schemas.py` | Pydantic model validation tests |
| Create | `tests/test_session.py` | Session store unit tests |
| Create | `tests/test_rag.py` | RAG chain unit tests (mocked) |
| Create | `tests/test_main.py` | Endpoint integration tests (mocked) |

---

## Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Replace the full file content with:

```
langchain-community
langchain-text-splitters
langchain-google-genai
langchain-chroma
pypdf
python-dotenv
fastapi
uvicorn[standard]
sse-starlette
pytest
httpx
```

- [ ] **Step 2: Install dependencies**

Run:
```
venv\Scripts\pip install -r requirements.txt
```

Expected: All packages install without error. Confirm fastapi appears in output.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add fastapi, uvicorn, sse-starlette, pytest, httpx"
```

---

## Task 2: App Package + Schemas

**Files:**
- Create: `app/__init__.py`
- Create: `app/schemas.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/conftest.py`:

```python
import os
import pytest
import app.session as session_module

os.environ.setdefault("GOOGLE_API_KEY", "test-key-for-testing")


@pytest.fixture(autouse=True)
def clear_session_store():
    session_module._store.clear()
    yield
    session_module._store.clear()
```

Create `tests/test_schemas.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
venv\Scripts\pytest tests/test_schemas.py -v
```

Expected: `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Create app package files**

Create `app/__init__.py` (empty):
```python
```

Create `tests/__init__.py` (empty):
```python
```

Create `app/schemas.py`:

```python
from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    session_id: str
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v


class ClearResponse(BaseModel):
    session_id: str
    cleared: bool
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
venv\Scripts\pytest tests/test_schemas.py -v
```

Expected: 7 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add app/__init__.py app/schemas.py tests/__init__.py tests/conftest.py tests/test_schemas.py
git commit -m "feat: add Pydantic schemas with validation"
```

---

## Task 3: Session Store

**Files:**
- Create: `app/session.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_session.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
venv\Scripts\pytest tests/test_session.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.session'`

- [ ] **Step 3: Implement session.py**

Create `app/session.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
venv\Scripts\pytest tests/test_session.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add app/session.py tests/test_session.py
git commit -m "feat: add in-memory session store"
```

---

## Task 4: LCEL RAG Chain

**Files:**
- Create: `app/rag.py`
- Create: `tests/test_rag.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rag.py`:

```python
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.rag import format_docs, build_chain


def test_format_docs_joins_with_double_newline():
    docs = [Document(page_content="Alpha"), Document(page_content="Beta")]
    result = format_docs(docs)
    assert result == "Alpha\n\nBeta"


def test_format_docs_single_doc():
    docs = [Document(page_content="Only one")]
    assert format_docs(docs) == "Only one"


def test_format_docs_empty_list():
    assert format_docs([]) == ""


def test_build_chain_returns_runnable():
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()

    with patch("app.rag.GoogleGenerativeAIEmbeddings") as mock_emb_cls, \
         patch("app.rag.Chroma") as mock_chroma_cls, \
         patch("app.rag.ChatGoogleGenerativeAI"):

        mock_chroma_cls.return_value = mock_vectorstore
        chain = build_chain("./fake_db")

    assert chain is not None
    mock_chroma_cls.assert_called_once_with(
        persist_directory="./fake_db",
        embedding_function=mock_emb_cls.return_value,
    )
    mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
venv\Scripts\pytest tests/test_rag.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.rag'`

- [ ] **Step 3: Implement rag.py**

Create `app/rag.py`:

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.session import get_session_history

_SYSTEM_PROMPT = (
    "You are an expert assistant for AES (Applied Engineering Solutions), "
    "a company specialising in engineering design, simulation, and analysis. "
    "Answer questions based strictly on the following case studies and documents. "
    "If the answer is not found in the context, say you don't have that information.\n\n"
    "Context:\n{context}"
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(persist_directory: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
venv\Scripts\pytest tests/test_rag.py -v
```

Expected: 4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add app/rag.py tests/test_rag.py
git commit -m "feat: add LCEL RAG chain with message history"
```

---

## Task 5: FastAPI App & Endpoints

**Files:**
- Create: `app/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_main.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
venv\Scripts\pytest tests/test_main.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Implement main.py**

Create `app/main.py`:

```python
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.rag import build_chain
from app.schemas import ChatRequest, ClearResponse
from app.session import clear_session


@asynccontextmanager
async def lifespan(app: FastAPI):
    chroma_path = "./chroma_db"
    if not os.path.exists(chroma_path):
        raise RuntimeError(
            "ChromaDB index not found. Run: python ingest.py"
        )
    app.state.chain = build_chain(chroma_path)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    async def token_stream():
        try:
            async for chunk in app.state.chain.astream(
                {"question": request.question},
                config={"configurable": {"session_id": request.session_id}},
            ):
                if chunk.content:
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@app.delete("/api/chat/{session_id}")
async def clear_chat(session_id: str):
    cleared = clear_session(session_id)
    return ClearResponse(session_id=session_id, cleared=cleared)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
venv\Scripts\pytest tests/test_main.py -v
```

Expected: 9 tests PASSED

- [ ] **Step 5: Run the full test suite**

Run:
```
venv\Scripts\pytest tests/ -v
```

Expected: All 25 tests PASSED across all 4 test files.

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: add FastAPI app with streaming chat and session endpoints"
```

---

## Task 6: Ingestion Script

**Files:**
- Create: `ingest.py`

No unit test here — this script's correctness is verified by the server starting successfully after it runs.

- [ ] **Step 1: Create ingest.py**

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PDF_DIR = "./case-studies"
PERSIST_DIR = "./chroma_db"


def ingest(pdf_dir: str = PDF_DIR, persist_dir: str = PERSIST_DIR) -> None:
    print(f"Loading PDFs from {pdf_dir} ...")
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()
    print(f"  Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("  Embedding and persisting to ChromaDB ...")
    Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    print(f"Done. Index saved to {persist_dir}/")


if __name__ == "__main__":
    ingest()
```

- [ ] **Step 2: Verify the script is syntactically valid**

Run:
```
venv\Scripts\python -m py_compile ingest.py && echo OK
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add ingest.py
git commit -m "feat: add PDF ingestion script"
```

---

## Task 7: End-to-End Smoke Test

Prerequisites: `GOOGLE_API_KEY` set in `.env`.

- [ ] **Step 1: Copy the example env file if `.env` doesn't exist**

```
copy .env.example .env
```

Edit `.env` and replace the placeholder value with your real `GOOGLE_API_KEY`.

- [ ] **Step 2: Run the ingestion script**

```
venv\Scripts\python ingest.py
```

Expected output:
```
Loading PDFs from ./case-studies ...
  Loaded <N> pages
  Split into <M> chunks
  Embedding and persisting to ChromaDB ...
Done. Index saved to ./chroma_db/
```

Confirm `./chroma_db/` directory was created.

- [ ] **Step 3: Start the server**

```
venv\Scripts\uvicorn app.main:app --reload
```

Expected: Server starts on `http://127.0.0.1:8000` with no errors.

- [ ] **Step 4: Check the health endpoint**

In a new terminal:
```
curl http://127.0.0.1:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 5: Test a streaming chat request**

```
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"test-001\", \"question\": \"What is the AES smart wearables project?\"}"
```

Expected: A stream of `data: {"token": "..."}` lines followed by `data: [DONE]`.

- [ ] **Step 6: Test session continuity (follow-up question)**

```
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"test-001\", \"question\": \"What technology did that project use?\"}"
```

Expected: Answer references the previous question context without re-asking.

- [ ] **Step 7: Test session clear**

```
curl -X DELETE http://127.0.0.1:8000/api/chat/test-001
```

Expected: `{"session_id":"test-001","cleared":true}`

- [ ] **Step 8: Final commit**

```bash
git add .
git commit -m "chore: end-to-end verified"
```
