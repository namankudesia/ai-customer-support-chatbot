"""Retriever with query rewriting for improved semantic search accuracy."""
from __future__ import annotations
from typing import List, Optional
from app.rag.document_loader import Document
from app.rag.vector_store import VectorStore
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

class Retriever:
    def __init__(self, vector_store: VectorStore, llm_client=None):
        self.vs = vector_store
        self.llm = llm_client

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        k = k or settings.top_k_retrieval
        search_q = self._rewrite(query) if self.llm else query
        emb = self._embed(search_q)
        if not emb:
            return []
        results = self.vs.similarity_search(emb, k=k)
        return [d for d, score in results if score > 0.25]

    def _rewrite(self, query: str) -> str:
        try:
            r = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content":
                    f"Rewrite this search query to be more specific. Return only the rewritten query.\nQuery: {query}"}],
                max_tokens=80, temperature=0)
            return r.choices[0].message.content.strip()
        except Exception:
            return query

    def _embed(self, text: str) -> Optional[List[float]]:
        if not self.llm:
            return [0.0] * 1536
        try:
            r = self.llm.embeddings.create(model=settings.embedding_model, input=text)
            return r.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
