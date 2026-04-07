import json
import math
import time
from collections import defaultdict
from typing import Dict, List

from app.core.config import settings

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    redis = None

try:
    import chromadb  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    chromadb = None


class HybridMemoryManager:
    """Short-term Redis memory + long-term vector memory with safe local fallback."""

    def __init__(self) -> None:
        self._redis_client = None
        self._chroma_collection = None
        self._local_turns: Dict[str, List[dict]] = defaultdict(list)
        self._local_long_term: Dict[str, List[str]] = defaultdict(list)
        self._init_redis()
        self._init_chroma()

    def _init_redis(self) -> None:
        if not redis:
            return
        try:
            client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            client.ping()
            self._redis_client = client
        except Exception:
            self._redis_client = None

    def _init_chroma(self) -> None:
        if not chromadb:
            return
        try:
            client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
            self._chroma_collection = client.get_or_create_collection(name="user_long_term_memory")
        except Exception:
            self._chroma_collection = None

    def _to_embedding(self, text: str, dims: int = 64) -> List[float]:
        """Lightweight deterministic embedding that avoids external model dependency."""
        vec = [0.0] * dims
        for token in text.lower().split():
            idx = hash(token) % dims
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def save_turn(self, user_id: str, user_message: str, assistant_message: str, keep: int = 12) -> None:
        record = {
            "ts": int(time.time()),
            "user": user_message,
            "assistant": assistant_message,
        }

        if self._redis_client is not None:
            key = f"chat:short:{user_id}"
            payload = json.dumps(record, ensure_ascii=False)
            self._redis_client.lpush(key, payload)
            self._redis_client.ltrim(key, 0, keep - 1)
            return

        self._local_turns[user_id].insert(0, record)
        self._local_turns[user_id] = self._local_turns[user_id][:keep]

    def get_recent_turns(self, user_id: str, limit: int = 4) -> List[dict]:
        if self._redis_client is not None:
            key = f"chat:short:{user_id}"
            raw_items = self._redis_client.lrange(key, 0, max(limit - 1, 0))
            turns = []
            for item in raw_items:
                try:
                    turns.append(json.loads(item))
                except json.JSONDecodeError:
                    continue
            return list(reversed(turns))

        turns = self._local_turns.get(user_id, [])[:limit]
        return list(reversed(turns))

    def add_long_term_memory(self, user_id: str, text: str) -> None:
        if not text.strip():
            return

        if self._chroma_collection is not None:
            doc_id = f"{user_id}:{int(time.time() * 1000)}"
            self._chroma_collection.upsert(
                ids=[doc_id],
                documents=[text],
                embeddings=[self._to_embedding(text)],
                metadatas=[{"user_id": user_id}],
            )
            return

        self._local_long_term[user_id].append(text)
        self._local_long_term[user_id] = self._local_long_term[user_id][-20:]

    def recall_long_term_memory(self, user_id: str, query_text: str, top_k: int = 2) -> List[str]:
        if self._chroma_collection is not None:
            result = self._chroma_collection.query(
                query_embeddings=[self._to_embedding(query_text)],
                n_results=top_k,
                where={"user_id": user_id},
            )
            docs = result.get("documents", [])
            if docs and docs[0]:
                return docs[0]
            return []

        items = self._local_long_term.get(user_id, [])
        return items[-top_k:]


hybrid_memory = HybridMemoryManager()
