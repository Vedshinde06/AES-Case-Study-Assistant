# Swap Chat LLM to HuggingFace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `ChatGoogleGenerativeAI` (gemini-2.0-flash) with `ChatHuggingFace` + `HuggingFaceEndpoint` (openai/gpt-oss-120b) in `app/rag.py`, keeping Gemini embeddings and the ChromaDB index untouched.

**Architecture:** `HuggingFaceEndpoint` wraps the HF Inference API call; `ChatHuggingFace` adapts it to LangChain's chat interface. The rest of the RAG chain (retriever, prompt, history) is unchanged.

**Tech Stack:** `langchain-huggingface`, `langchain-google-genai` (embeddings only), `langchain-chroma`, FastAPI, ChromaDB

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Add `langchain-huggingface` |
| `app/rag.py` | Modify | Swap LLM imports and instantiation |
| `tests/conftest.py` | Modify | Stub `HF_TOKEN` env var for tests |
| `tests/test_rag.py` | Modify | Patch new classes instead of old |
| `.env` | Modify (manual) | Add `HF_TOKEN=<your_token>` |

---

## Task 1: Add dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add `langchain-huggingface` to requirements**

`requirements.txt` should look like:

```
langchain-community
langchain-text-splitters
langchain-google-genai
langchain-huggingface
langchain-chroma
pypdf
python-dotenv
fastapi
uvicorn[standard]
sse-starlette
pytest
httpx
```

- [ ] **Step 2: Install the new dependency**

```
pip install langchain-huggingface
```

Expected: installs `langchain-huggingface` and its deps (`huggingface-hub`, `tokenizers`, etc.) with no errors.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add langchain-huggingface dependency"
```

---

## Task 2: Update test to patch new LLM classes

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_rag.py`

- [ ] **Step 1: Add `HF_TOKEN` stub to conftest**

Replace the contents of `tests/conftest.py` with:

```python
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
```

- [ ] **Step 2: Update `test_build_chain_returns_runnable` to patch the new classes**

Replace the `test_build_chain_returns_runnable` function in `tests/test_rag.py` with:

```python
def test_build_chain_returns_runnable():
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()

    with patch("app.rag.GoogleGenerativeAIEmbeddings") as mock_emb_cls, \
         patch("app.rag.Chroma") as mock_chroma_cls, \
         patch("app.rag.HuggingFaceEndpoint") as mock_endpoint_cls, \
         patch("app.rag.ChatHuggingFace") as mock_chat_cls:

        mock_chroma_cls.return_value = mock_vectorstore
        mock_chat_cls.return_value = MagicMock()
        chain = build_chain("./fake_db")

    assert chain is not None
    mock_chroma_cls.assert_called_once_with(
        persist_directory="./fake_db",
        embedding_function=mock_emb_cls.return_value,
        collection_name="aes-case-studies",
    )
    mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
    mock_endpoint_cls.assert_called_once_with(
        repo_id="openai/gpt-oss-120b",
        huggingfacehub_api_token="test-hf-token-for-testing",
        streaming=True,
    )
    mock_chat_cls.assert_called_once_with(llm=mock_endpoint_cls.return_value)
```

Also update the imports at the top of `tests/test_rag.py` to remove `ChatGoogleGenerativeAI` reference (it is no longer patched):

```python
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.rag import format_docs, build_chain
```

(The imports are already correct — no change needed here since `ChatGoogleGenerativeAI` was only referenced inside the `patch` call, not imported directly in the test file.)

- [ ] **Step 3: Run test to verify it fails**

```
pytest tests/test_rag.py::test_build_chain_returns_runnable -v
```

Expected: FAIL — `app.rag` still imports `ChatGoogleGenerativeAI` and doesn't have `HuggingFaceEndpoint` or `ChatHuggingFace`, so the patch targets won't resolve.

---

## Task 3: Implement the LLM swap in `app/rag.py`

**Files:**
- Modify: `app/rag.py`

- [ ] **Step 1: Rewrite `app/rag.py`**

Replace the full contents of `app/rag.py` with:

```python
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

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
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="aes-case-studies",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    endpoint = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        streaming=True,
    )
    llm = ChatHuggingFace(llm=endpoint)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    async def _retrieve(x: dict) -> str:
        docs = await retriever.ainvoke(x["question"])
        return format_docs(docs)

    rag_chain = (
        RunnablePassthrough.assign(context=_retrieve)
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

- [ ] **Step 2: Run tests to verify they pass**

```
pytest tests/test_rag.py -v
```

Expected output:
```
tests/test_rag.py::test_format_docs_joins_with_double_newline PASSED
tests/test_rag.py::test_format_docs_single_doc PASSED
tests/test_rag.py::test_format_docs_empty_list PASSED
tests/test_rag.py::test_build_chain_returns_runnable PASSED
```

- [ ] **Step 3: Run full test suite to check for regressions**

```
pytest -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add app/rag.py tests/test_rag.py tests/conftest.py
git commit -m "feat: swap Gemini chat model for HuggingFace openai/gpt-oss-120b"
```

---

## Task 4: Add `HF_TOKEN` to `.env` (manual step)

**Files:**
- Modify: `.env` (not committed — contains secrets)

- [ ] **Step 1: Add `HF_TOKEN` to your `.env` file**

Open `.env` and add:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get your token from https://huggingface.co/settings/tokens. The token needs **read** access. If `openai/gpt-oss-120b` requires a Pro subscription, ensure your account has one and the token belongs to that account.

- [ ] **Step 2: Smoke-test the running server**

```
uvicorn app.main:app --reload
```

Then in a separate terminal:

```
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AES?", "session_id": "smoke-test"}'
```

Expected: SSE stream of `data: {"token": "..."}` lines ending with `data: [DONE]`.

If you get a 403 or model-not-found error, the model may require HF Pro — verify at https://huggingface.co/openai/gpt-oss-120b.
