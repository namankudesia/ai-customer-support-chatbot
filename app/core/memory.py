"""Redis-backed sliding-window conversation memory with in-memory fallback."""
from __future__ import annotations
import json, time
from typing import List, Dict
from dataclasses import dataclass, asdict
from app.core.config import get_settings
settings = get_settings()

@dataclass
class Turn:
    role: str
    content: str
    timestamp: float = 0.0
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

class ConversationMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.max_turns = settings.max_history_turns
        self._turns: List[Turn] = []
        self._redis = self._connect()
        self._load()

    def add(self, role: str, content: str):
        self._turns.append(Turn(role=role, content=content))
        if len(self._turns) > self.max_turns * 2:
            self._turns = self._turns[-(self.max_turns * 2):]
        self._save()

    def get_messages(self) -> List[Dict]:
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def clear(self):
        self._turns = []
        if self._redis:
            self._redis.delete(f"session:{self.session_id}")

    def _connect(self):
        try:
            import redis
            r = redis.from_url(settings.redis_url, decode_responses=True)
            r.ping()
            return r
        except Exception:
            return None

    def _save(self):
        if self._redis:
            self._redis.setex(f"session:{self.session_id}", settings.session_ttl,
                              json.dumps([asdict(t) for t in self._turns]))

    def _load(self):
        if self._redis:
            raw = self._redis.get(f"session:{self.session_id}")
            if raw:
                self._turns = [Turn(**t) for t in json.loads(raw)]
