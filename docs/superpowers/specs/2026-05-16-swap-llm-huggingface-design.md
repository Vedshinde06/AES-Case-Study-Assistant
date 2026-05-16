# Spec: Swap Chat LLM from Gemini to Hugging Face

**Date:** 2026-05-16  
**Status:** Approved

## Overview

Replace the Gemini chat model (`gemini-2.0-flash`) with `openai/gpt-oss-120b` served via the Hugging Face Inference API. Gemini embeddings (`gemini-embedding-001`) and the ChromaDB index are unchanged — no re-ingestion required.

## Architecture

No structural changes to the RAG pipeline. The `build_chain` function in `app/rag.py` constructs:

1. Embeddings → ChromaDB retriever (unchanged)
2. Prompt template (unchanged)
3. LLM → swapped from `ChatGoogleGenerativeAI` to `ChatHuggingFace`
4. `RunnableWithMessageHistory` wrapper (unchanged)

## Components Changed

### `app/rag.py`

- Remove import: `ChatGoogleGenerativeAI` from `langchain_google_genai`
- Add imports: `HuggingFaceEndpoint`, `ChatHuggingFace` from `langchain_huggingface`
- Replace LLM instantiation:

```python
endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    streaming=True,
)
llm = ChatHuggingFace(llm=endpoint)
```

### `requirements.txt`

- Add: `langchain-huggingface`
- Keep: `langchain-google-genai` (still required for `GoogleGenerativeAIEmbeddings`)

### `.env`

- Add: `HF_TOKEN=<your_huggingface_token>`
- Keep: `GOOGLE_API_KEY` (still required for embeddings)

## Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini embeddings | Yes |
| `HF_TOKEN` | HF Inference API for chat model | Yes |

## Constraints and Notes

- `openai/gpt-oss-120b` is a 120B parameter model. The HF Inference API free tier typically serves models ≤13B. A **HF Pro subscription** or a **dedicated Inference Endpoint** may be required. If the API returns a 403 or model-not-found error, the model ID or tier should be verified.
- Streaming is passed to `HuggingFaceEndpoint`. LangChain's `ChatHuggingFace` propagates streaming tokens through `astream`, so the SSE response in `app/main.py` is unaffected.

## Out of Scope

- Swapping embeddings model
- Re-ingesting documents
- Any changes to session management, schemas, or API routes
