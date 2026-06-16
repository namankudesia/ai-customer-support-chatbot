"""FAISS vector store with persistence, cosine similarity, and metadata filtering."""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from app.rag.document_loader import Document

class VectorStore:
    def __init__(self, embedding_dim=1536, index_path="data/faiss_index"):
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path)
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
        self._index = None
        self._init_faiss()
        self._load_if_exists()

    def _init_faiss(self):
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.embedding_dim)
        except ImportError:
            pass

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        if self._index and embeddings:
            import faiss, numpy as np
            arr = np.array(embeddings, dtype="float32")
            faiss.normalize_L2(arr)
            self._index.add(arr)

    def similarity_search(self, q_emb: List[float], k=5,
                          filter_meta: Optional[Dict]=None) -> List[Tuple[Document, float]]:
        if not self.documents:
            return []
        if self._index:
            import faiss, numpy as np
            q = np.array([q_emb], dtype="float32")
            faiss.normalize_L2(q)
            scores, idx = self._index.search(q, min(k*3, len(self.documents)))
            results = [(self.documents[i], float(scores[0][j]))
                       for j, i in enumerate(idx[0]) if i >= 0]
        else:
            import math
            def cos(a, b):
                d = sum(x*y for x,y in zip(a,b))
                return d / (math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b)) + 1e-9)
            results = sorted(
                [(doc, cos(q_emb, emb)) for doc, emb in zip(self.documents, self.embeddings)],
                key=lambda x: -x[1])
        if filter_meta:
            results = [(d,s) for d,s in results
                       if all(d.metadata.get(k)==v for k,v in filter_meta.items())]
        return results[:k]

    def save(self):
        self.index_path.mkdir(parents=True, exist_ok=True)
        if self._index:
            import faiss
            faiss.write_index(self._index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "docs.pkl", "wb") as f:
            pickle.dump({"documents": self.documents, "embeddings": self.embeddings}, f)

    def _load_if_exists(self):
        p = self.index_path / "docs.pkl"
        if not p.exists():
            return
        with open(p, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        if self._index:
            import faiss
            ip = self.index_path / "index.faiss"
            if ip.exists():
                self._index = faiss.read_index(str(ip))
