# FastAPI Chat/Query Backend — Design Spec

**Date:** 2026-05-16  
**Project:** AES-RAG  
**Status:** Approved

---

## Overview

A FastAPI backend that exposes a streaming RAG chat API over AES engineering PDF case studies. The system uses LangChain LCEL, ChromaDB for vector storage, and Google Gemini for generation. Conversation history is maintained per session in memory.

---

## Architecture

```
AES-RAG/
├── case-studies/          # source PDFs (existing)
├── chroma_db/             # persisted ChromaDB index (created by ingest.py)
├── ingest.py              # one-shot ingestion script
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app, lifespan, router
│   ├── rag.py             # LCEL chain with RunnableWithMessageHistory
│   ├── session.py         # in-memory session store
│   └── schemas.py         # Pydantic request/response models
├── .env                   # GOOGLE_API_KEY
└── requirements.txt       # updated with fastapi, uvicorn, sse-starlette
```

**Data flow:**

```
POST /api/chat { session_id, question }
  → session.py: fetch or create ChatMessageHistory for session_id
  → rag.py: ChromaDB retriever (top 5 docs)
  → rag.py: build prompt (history + context + question)
  → ChatGoogleGenerativeAI.astream()
  → StreamingResponse: SSE tokens → data: [DONE]
```

---

## Components

### `ingest.py` — Ingestion Script

Run once before starting the server. Not part of the server process.

- **Loader:** `PyPDFDirectoryLoader("./case-studies")`
- **Splitter:** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
- **Embeddings:** `GoogleGenerativeAIEmbeddings(model="models/embedding-001")`
- **Store:** `Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")`

### `app/session.py` — Session Store

```python
_store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory
def clear_session(session_id: str) -> bool
```

In-memory only. Sessions are lost on server restart. No TTL. Suitable for hosted deployments where restarts are infrequent.

### `app/rag.py` — LCEL Chain

- **Retriever:** `Chroma(persist_directory="./chroma_db").as_retriever(search_kwargs={"k": 5})`
- **Model:** `ChatGoogleGenerativeAI(model="gemini-2.0-flash", streaming=True)`
- **Prompt:** System message sets AES assistant persona; template injects `{context}`, `{chat_history}`, `{question}`
- **Chain:** `RunnableWithMessageHistory` wraps `retriever | format_docs | prompt | llm`
- History input key: `"question"`, history key: `"chat_history"`

### `app/schemas.py` — Pydantic Models

```python
class ChatRequest(BaseModel):
    session_id: str   # client-generated UUID
    question: str     # must be non-empty

class ClearResponse(BaseModel):
    session_id: str
    cleared: bool
```

### `app/main.py` — FastAPI App

- Lifespan: load ChromaDB on startup, raise `RuntimeError` with "Run ingest.py first" if `./chroma_db` is missing
- `POST /api/chat` → `StreamingResponse(media_type="text/event-stream")`
- `DELETE /api/chat/{session_id}` → `ClearResponse`
- `GET /health` → `{"status": "ok"}`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Stream answer to a question |
| `DELETE` | `/api/chat/{session_id}` | Clear session history |
| `GET` | `/health` | Liveness check |

### SSE Wire Format

```
data: {"token": "Hello"}\n\n
data: {"token": " world"}\n\n
data: [DONE]\n\n
```

Error mid-stream (HTTP status already sent):
```
data: {"error": "Gemini API error: ..."}\n\n
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Empty `question` | `422` Unprocessable Entity (Pydantic) |
| Missing ChromaDB index | `500` — "Run ingest.py first" |
| Gemini API error mid-stream | Yield `{"error": "..."}` SSE event, close stream |
| Unknown `session_id` on DELETE | Return `{"cleared": false}` |

---

## Dependencies Added

```
fastapi
uvicorn[standard]
sse-starlette
```

---

## Out of Scope

- Session persistence across restarts (no Redis/DB)
- CORS (same-origin deployment)
- Authentication
- Rate limiting
- Re-ranking retrieved documents
