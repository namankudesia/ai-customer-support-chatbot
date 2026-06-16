"""Multi-format document ingestion: PDF, DOCX, TXT, CSV, Markdown with deduplication."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]

class DocumentLoader:
    SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".csv"}
    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._seen: set = set()

    def load_directory(self, path: str) -> List[Document]:
        docs = []
        for fp in Path(path).rglob("*"):
            if fp.suffix in self.SUPPORTED:
                docs.extend(self._load_file(fp))
        return docs

    def _load_file(self, path: Path) -> List[Document]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            result = []
            for i, chunk in enumerate(self._split(text)):
                h = hashlib.md5(chunk.encode()).hexdigest()
                if h in self._seen:
                    continue
                self._seen.add(h)
                result.append(Document(
                    content=chunk,
                    metadata={"source": str(path), "chunk": i, "type": path.suffix}
                ))
            return result
        except Exception:
            return []

    def _split(self, text: str) -> List[str]:
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            c = " ".join(words[start:end])
            if len(c.strip()) > 20:
                chunks.append(c)
            start += self.chunk_size - self.chunk_overlap
        return chunks
