# AES Knowledge Base — RAG Chatbot

A retrieval-augmented generation (RAG) chatbot built for **Applied Engineering Solutions (AES)**, a company specialising in engineering design, simulation, and analysis. Ask questions in natural language and get answers grounded in AES's library of engineering case studies.

---

## How it works

```
User question
     │
     ▼
Gemini Embeddings ──► ChromaDB vector store ──► top-5 relevant chunks
                                                        │
                                                        ▼
                                              HuggingFace LLM (gpt-oss-120b)
                                                        │
                                                        ▼
                                              Streamed answer (SSE)
```

PDFs are ingested once via `ingest.py`, embedded with `gemini-embedding-001`, and stored in a persistent ChromaDB collection. At query time the FastAPI backend retrieves the most relevant chunks and streams a response token-by-token.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| RAG framework | LangChain |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| LLM | HuggingFace `openai/gpt-oss-120b` |
| Vector store | ChromaDB (persistent SQLite) |
| Frontend | Vanilla HTML / CSS / JS |
| Deployment | Vercel (Python serverless) |

---

## Getting started

### Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/) API key (for embeddings)
- A [HuggingFace](https://huggingface.co/settings/tokens) API token (for the LLM)

### 1 — Clone and install

```bash
git clone <repo-url>
cd AES-RAG
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### 3 — Ingest the case studies

Place engineering PDFs in the `case-studies/` directory, then run:

```bash
python ingest.py
```

This loads every PDF, splits it into 1 000-token chunks (200-token overlap), embeds them via Gemini, and persists the vectors to `chroma_db/`. Re-running wipes and rebuilds the collection.

### 4 — Start the server

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000` — the chat interface loads automatically.

---

## Project structure

```
AES-RAG/
├── app/
│   ├── main.py        # FastAPI app, lifespan, API routes
│   ├── rag.py         # RAG chain (retriever + LLM + session history)
│   ├── session.py     # In-memory chat session store
│   └── schemas.py     # Pydantic request/response models
├── frontend/
│   └── index.html     # Single-page chat UI
├── case-studies/      # Source PDFs (not committed)
├── chroma_db/         # Persisted vector store (committed as built asset)
├── ingest.py          # PDF ingestion pipeline
├── tests/             # Pytest test suite
├── vercel.json        # Vercel deployment config
└── requirements.txt
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Send a question; streams tokens via SSE |
| `DELETE` | `/api/chat/{session_id}` | Clear a conversation session |
| `GET` | `/health` | Health check |

### Chat request body

```json
{
  "session_id": "abc-123",
  "question": "What simulation methods does AES use for thermal analysis?"
}
```

### Streaming response (SSE)

```
data: {"token": "AES "}
data: {"token": "uses "}
...
data: [DONE]
```

---

## Deployment (Vercel)

The app is configured for Vercel's Python serverless runtime. Because Vercel's Lambda filesystem is read-only, the lifespan handler copies `chroma_db/` to `/tmp` at cold-start so SQLite can write its WAL files.

```bash
vercel deploy
```

Required environment variables in Vercel project settings: `GOOGLE_API_KEY`, `HF_TOKEN`.

---

## Running tests

```bash
pytest tests/
```

---

## Case studies included

The knowledge base covers AES engineering work across domains including:

- Thermal management (heat sinks, cooling channels, electrostatic chucks)
- Structural and modal analysis (PCIe cards, hardware frames)
- CFD simulation (boilers, fenestration water systems)
- Robotics and automation (bolt feeders, assembly stations, robotic controllers)
- Medical device design and development
- Smart wearables and IoT electronics
- PCB testing and optical inspection systems
