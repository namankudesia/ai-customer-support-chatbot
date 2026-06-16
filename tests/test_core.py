import pytest
from app.rag.document_loader import DocumentLoader, Document

def test_chunking():
    loader = DocumentLoader(chunk_size=10, chunk_overlap=2)
    chunks = loader._split("word " * 30)
    assert len(chunks) > 1

def test_dedup(tmp_path):
    (tmp_path / "a.txt").write_text("hello world test document with sufficient words to form a proper chunk here now")
    loader = DocumentLoader()
    d1 = loader.load_directory(str(tmp_path))
    d2 = loader.load_directory(str(tmp_path))
    assert len(d2) == 0

def test_doc_id():
    d = Document(content="test content here for hashing")
    assert len(d.doc_id) == 12
