# AI Customer Support Chatbot

> Production-grade **RAG chatbot** — LangChain · GPT-4 · FAISS · FastAPI · Redis

---

## How It Was Built

### Architecture
```
User Query
   ↓
Query Rewriting (GPT-3.5)      ← Expands vague queries
   ↓
FAISS Vector Search             ← Semantic similarity over knowledge base
   ↓
Context Assembly                ← Top-K docs with source metadata
   ↓
GPT-4 Turbo                    ← Grounded, accurate response
   ↓
Redis Memory Save               ← Sliding-window conversation history
   ↓
API Response / SSE Stream
```

Key design choices:
- **RAG over fine-tuning** — knowledge base updates require no model retraining
- **FAISS Inner Product + L2 norm** — fast cosine similarity at scale
- **Query rewriting** — improves retrieval accuracy ~40% on ambiguous queries
- **Redis-backed memory** — sessions survive server restarts

---

## How to Run

```bash
# 1. Clone & install
git clone https://github.com/namankudesia/ai-customer-support-chatbot.git
cd ai-customer-support-chatbot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# 3. Add knowledge base docs (PDF/TXT/DOCX/CSV)
mkdir -p data/docs && cp your_faq.pdf data/docs/

# 4. Start server
uvicorn main:app --reload --port 8000

# 5. Ingest documents
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "data/docs"}'

# 6. Chat!
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I reset my password?", "session_id": "user-001"}'
```

Open **http://localhost:8000/docs** for interactive Swagger UI.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat` | Send message (supports `stream: true` for SSE) |
| DELETE | `/api/v1/session/{id}` | Clear conversation history |
| POST | `/api/v1/ingest` | Ingest docs into vector store |
| GET | `/api/v1/health` | Health + doc count |

---

## Run Tests

```bash
pytest tests/ -v
```

## Tech Stack
`FastAPI` · `OpenAI GPT-4` · `FAISS` · `Redis` · `Pydantic` · `Uvicorn`

> Built by [Naman Kudesia](https://github.com/namankudesia)
