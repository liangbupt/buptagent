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
        self._local_long_term: Dict[str, List[dict]] = defaultdict(list)
        self._long_term_ttl_sec = 60 * 60 * 24 * 30
        self._short_term_ttl_sec = 60 * 60 * 24 * 7
        self._init_redis()
        self._init_chroma()

    def _now(self) -> int:
        return int(time.time())

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _text_hash(self, text: str) -> str:
        return str(abs(hash(self._normalize_text(text))))

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
            "ts": self._now(),
            "user": user_message,
            "assistant": assistant_message,
        }

        if self._redis_client is not None:
            key = f"chat:short:{user_id}"
            payload = json.dumps(record, ensure_ascii=False)
            self._redis_client.lpush(key, payload)
            self._redis_client.ltrim(key, 0, keep - 1)
            self._redis_client.expire(key, self._short_term_ttl_sec)
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

    def score_long_term_signal(self, message: str) -> float:
        text = (message or "").strip().lower()
        if not text:
            return 0.0

        score = 0.0
        keyword_weights = {
            "喜欢": 1.0,
            "偏好": 1.0,
            "不吃": 1.0,
            "爱吃": 0.9,
            "预算": 0.9,
            "习惯": 0.8,
            "常去": 0.8,
            "长期": 0.8,
            "prefer": 1.0,
            "favorite": 1.0,
            "usually": 0.8,
            "budget": 0.9,
        }
        for token, weight in keyword_weights.items():
            if token in text:
                score += weight

        if len(text) > 20:
            score += 0.15
        if "我" in text or "i " in text:
            score += 0.15
        return score

    def should_store_long_term(self, message: str, threshold: float = 1.0) -> bool:
        return self.score_long_term_signal(message) >= threshold

    def save_route_audit(
        self,
        user_id: str,
        route: str,
        rationale: str,
        user_message: str,
        confidence: float,
    ) -> None:
        record = {
            "ts": self._now(),
            "route": route,
            "rationale": rationale,
            "user": user_message,
            "confidence": confidence,
        }
        if self._redis_client is not None:
            key = f"chat:route_audit:{user_id}"
            payload = json.dumps(record, ensure_ascii=False)
            self._redis_client.lpush(key, payload)
            self._redis_client.ltrim(key, 0, 49)
            self._redis_client.expire(key, self._short_term_ttl_sec)
            return

        self._local_turns[f"route_audit::{user_id}"].insert(0, record)
        self._local_turns[f"route_audit::{user_id}"] = self._local_turns[f"route_audit::{user_id}"][:50]

    def get_route_audit(self, user_id: str, limit: int = 20) -> List[dict]:
        if self._redis_client is not None:
            key = f"chat:route_audit:{user_id}"
            raw_items = self._redis_client.lrange(key, 0, max(limit - 1, 0))
            audits: List[dict] = []
            for item in raw_items:
                try:
                    audits.append(json.loads(item))
                except json.JSONDecodeError:
                    continue
            return list(reversed(audits))

        items = self._local_turns.get(f"route_audit::{user_id}", [])[:limit]
        return list(reversed(items))

    def add_long_term_memory(self, user_id: str, text: str) -> None:
        if not text.strip():
            return

        now = self._now()
        expires_at = now + self._long_term_ttl_sec
        normalized_hash = self._text_hash(text)

        if self._chroma_collection is not None:
            existing = self._chroma_collection.get(where={"user_id": user_id}, include=["metadatas"])
            metadatas = existing.get("metadatas") or []
            for meta in metadatas:
                if not isinstance(meta, dict):
                    continue
                if meta.get("text_hash") == normalized_hash and int(meta.get("expires_at", 0)) > now:
                    return

            doc_id = f"{user_id}:{int(time.time() * 1000)}"
            self._chroma_collection.upsert(
                ids=[doc_id],
                documents=[text],
                embeddings=[self._to_embedding(text)],
                metadatas=[
                    {
                        "user_id": user_id,
                        "created_at": now,
                        "expires_at": expires_at,
                        "text_hash": normalized_hash,
                    }
                ],
            )
            return

        items = self._local_long_term[user_id]
        if any(x.get("text_hash") == normalized_hash and int(x.get("expires_at", 0)) > now for x in items):
            return

        items.append(
            {
                "text": text,
                "created_at": now,
                "expires_at": expires_at,
                "text_hash": normalized_hash,
            }
        )
        self._local_long_term[user_id] = items[-20:]

    def recall_long_term_memory(self, user_id: str, query_text: str, top_k: int = 2) -> List[str]:
        now = self._now()
        if self._chroma_collection is not None:
            result = self._chroma_collection.query(
                query_embeddings=[self._to_embedding(query_text)],
                n_results=max(top_k * 3, top_k),
                where={"user_id": user_id},
                include=["documents", "metadatas"],
            )
            docs = result.get("documents", [])
            metas = result.get("metadatas", [])
            if docs and docs[0]:
                filtered: List[str] = []
                meta_row = metas[0] if metas else []
                for idx, doc in enumerate(docs[0]):
                    meta = meta_row[idx] if idx < len(meta_row) else {}
                    expires_at = int((meta or {}).get("expires_at", now + 1))
                    if expires_at > now:
                        filtered.append(doc)
                    if len(filtered) >= top_k:
                        break
                return filtered
            return []

        items = self._local_long_term.get(user_id, [])
        alive = [x for x in items if int(x.get("expires_at", 0)) > now]
        self._local_long_term[user_id] = alive[-20:]
        return [x.get("text", "") for x in alive[-top_k:]]

    def delete_user_memory(self, user_id: str, scope: str = "all") -> None:
        scope = scope.lower().strip()
        if self._redis_client is not None:
            if scope in {"all", "short"}:
                self._redis_client.delete(f"chat:short:{user_id}")
            if scope in {"all", "route", "audit"}:
                self._redis_client.delete(f"chat:route_audit:{user_id}")

        if scope in {"all", "short"}:
            self._local_turns.pop(user_id, None)
        if scope in {"all", "route", "audit"}:
            self._local_turns.pop(f"route_audit::{user_id}", None)

        if scope in {"all", "long"}:
            self._local_long_term.pop(user_id, None)
            if self._chroma_collection is not None:
                found = self._chroma_collection.get(where={"user_id": user_id}, include=[])
                ids = found.get("ids", [])
                if ids:
                    self._chroma_collection.delete(ids=ids)


hybrid_memory = HybridMemoryManager()
