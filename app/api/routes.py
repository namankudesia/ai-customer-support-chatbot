"""FastAPI routes: chat, stream, session management, document ingestion, health."""
from __future__ import annotations
import uuid
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter()

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    stream: bool = False

class ChatResponse(BaseModel):
    reply: str
    session_id: str

class IngestRequest(BaseModel):
    directory: str = "data/docs"

def _engine(session_id: str):
    from openai import OpenAI
    from app.core.config import get_settings
    from app.rag.vector_store import VectorStore
    from app.rag.retriever import Retriever
    from app.core.memory import ConversationMemory
    from app.core.chat_engine import ChatEngine
    s = get_settings()
    c = OpenAI(api_key=s.openai_api_key)
    return ChatEngine(Retriever(VectorStore(index_path=s.vector_store_path), c), ConversationMemory(session_id), c)

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    engine = _engine(sid)
    if req.stream:
        async def gen():
            async for c in engine.stream_chat(req.message):
                yield f"data: {c}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return ChatResponse(reply=engine.chat(req.message), session_id=sid)

@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    from app.core.memory import ConversationMemory
    ConversationMemory(session_id).clear()
    return {"cleared": session_id}

@router.post("/ingest")
async def ingest(req: IngestRequest, bg: BackgroundTasks):
    bg.add_task(_ingest_bg, req.directory)
    return {"status": "started", "directory": req.directory}

@router.get("/health")
async def health():
    from app.rag.vector_store import VectorStore
    from app.core.config import get_settings
    s = get_settings()
    vs = VectorStore(index_path=s.vector_store_path)
    return {"status": "ok", "docs": len(vs.documents), "model": s.llm_model}

def _ingest_bg(directory: str):
    from openai import OpenAI
    from app.core.config import get_settings
    from app.rag.document_loader import DocumentLoader
    from app.rag.vector_store import VectorStore
    s = get_settings()
    c = OpenAI(api_key=s.openai_api_key)
    loader = DocumentLoader(s.chunk_size, s.chunk_overlap)
    docs = loader.load_directory(directory)
    vs = VectorStore(index_path=s.vector_store_path)
    embeddings = [c.embeddings.create(model=s.embedding_model, input=d.content).data[0].embedding for d in docs]
    vs.add_documents(docs, embeddings)
    vs.save()
