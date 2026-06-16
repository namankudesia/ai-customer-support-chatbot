from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "AI Customer Support Chatbot"
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-4-turbo-preview"
    llm_temperature: float = 0.2
    max_tokens: int = 1024
    vector_store_path: str = "data/faiss_index"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 5
    redis_url: str = "redis://localhost:6379"
    session_ttl: int = 3600
    max_history_turns: int = 10
    log_level: str = "INFO"
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
