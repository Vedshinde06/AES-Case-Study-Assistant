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
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/chat/{session_id}")
async def clear_chat(session_id: str):
    cleared = clear_session(session_id)
    return ClearResponse(session_id=session_id, cleared=cleared)


@app.get("/health")
async def health():
    return {"status": "ok"}
