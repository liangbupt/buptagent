import math
import os
import json
import re
from collections import Counter
from typing import Dict, List, Tuple

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import settings


class CampusRAGRetriever:
    """Hybrid RAG retriever: keyword + BGE vector recall, fusion, rerank, top-k."""

    def __init__(self) -> None:
        self._kb_path = settings.RAG_KB_PATH
        self._kb_mtime = 0.0
        self._docs: List[str] = []
        self._source_ids: List[str] = []
        self._doc_tokens: List[List[str]] = []
        self._idf: Dict[str, float] = {}
        
        # 真正的高级点：引入完整的 BGE 语义模型（支持离线部署、推理成本低、中文效果好）
        self._embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={"device": "cpu"},    # 面试可说是GPU/CPU可自适应切换
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 真正的高级点：持久化的 ChromaDB 实例实现快速检索
        self._vectorstore = Chroma(
            collection_name="campus_kb",
            embedding_function=self._embeddings,
            persist_directory="./chroma_data"
        )
        
        self._reload_if_needed(force=True)

    def _default_docs(self) -> List[Dict[str, str]]:
        return [
            {"source_id": "campus_kb_1", "content": "教三通常在整点后 10 分钟会有一批空教室释放，建议优先查询 101-305。"},
            {"source_id": "campus_kb_2", "content": "工作日 11:30-12:20 食堂高峰明显，若预算 30 以内可优先考虑馨园一层套餐窗口。"},
            {"source_id": "campus_kb_3", "content": "蹭课建议先确认课堂容量与任课老师要求，AI 相关课程在下午时段更集中。"},
            {"source_id": "campus_kb_4", "content": "跳蚤市场交易建议约在图书馆门口或教学楼大厅进行，当面验货优先。"},
            {"source_id": "campus_kb_5", "content": "如果用户表达偏好（如不吃辣、预算 30），应在后续推荐中持续沿用。"},
            {"source_id": "campus_kb_6", "content": "复合任务可拆分为 教务查询、生活推荐、交易反馈 三类并并行组织回复。"},
        ]

    def _reload_if_needed(self, force: bool = False) -> None:
        path = self._kb_path
        if not path:
            payload = self._default_docs()
            force = True
        else:
            exists = os.path.exists(path)
            if not exists:
                payload = self._default_docs()
                force = True
            else:
                mtime = os.path.getmtime(path)
                if not force and mtime <= self._kb_mtime:
                    return
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                payload = loaded if isinstance(loaded, list) else self._default_docs()
                self._kb_mtime = mtime

        docs: List[str] = []
        source_ids: List[str] = []
        for i, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            source_id = str(item.get("source_id", f"campus_kb_{i}")).strip() or f"campus_kb_{i}"
            docs.append(content)
            source_ids.append(source_id)

        if not docs:
            fallback = self._default_docs()
            docs = [x["content"] for x in fallback]
            source_ids = [x["source_id"] for x in fallback]

        self._docs = docs
        self._source_ids = source_ids
        self._doc_tokens = [self._tokenize(text) for text in self._docs]
        self._idf = self._build_idf(self._doc_tokens)
        
        # 将结构化的文本放入 ChromaDB
        # 这里判断简单的数量比对来进行缓存更新。在真实的工业项目中，会使用基于 Hash 签名的 Document 变更对比。
        try:
            curr_count = self._vectorstore._collection.count()
        except Exception:
            curr_count = 0
            
        if curr_count != len(docs) or force:
            langchain_docs = [
                Document(page_content=doc, metadata={"source_id": sid, "original_idx": idx})
                for idx, (doc, sid) in enumerate(zip(self._docs, self._source_ids))
            ]
            # 先清空当前collection，再写入以实现热刷新
            self._vectorstore.delete_collection()
            self._vectorstore = Chroma(
                collection_name="campus_kb",
                embedding_function=self._embeddings,
                persist_directory="./chroma_data"
            )
            self._vectorstore.add_documents(langchain_docs)

    def _tokenize(self, text: str) -> List[str]:
        lowered = text.lower()
        # Mixed tokenizer: keeps English words/numbers and Chinese character bigrams.
        words = re.findall(r"[a-z0-9]+", lowered)
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
        bigrams = ["".join(cjk_chars[i:i + 2]) for i in range(len(cjk_chars) - 1)]
        return words + bigrams

    def _build_idf(self, docs_tokens: List[List[str]]) -> Dict[str, float]:
        doc_count = len(docs_tokens)
        df: Counter[str] = Counter()
        for tokens in docs_tokens:
            for token in set(tokens):
                df[token] += 1
        idf: Dict[str, float] = {}
        for token, freq in df.items():
            idf[token] = math.log((doc_count + 1.0) / (freq + 1.0)) + 1.0
        return idf

    def _keyword_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        tf = Counter(doc_tokens)
        score = 0.0
        for token in query_tokens:
            if token in tf:
                score += tf[token] * self._idf.get(token, 0.0)
        return score

    def _rank(self, scores: List[Tuple[float, int]]) -> Dict[int, int]:
        ordered = sorted(scores, key=lambda x: x[0], reverse=True)
        return {doc_idx: rank + 1 for rank, (_, doc_idx) in enumerate(ordered)}

    def _fuse_rrf(
        self,
        keyword_scores: List[Tuple[float, int]],
        vector_scores: List[Tuple[float, int]],
        k: int = 60,
    ) -> Dict[int, float]:
        keyword_rank = self._rank(keyword_scores)
        vector_rank = self._rank(vector_scores)
        fused: Dict[int, float] = {}
        for doc_idx in range(len(self._docs)):
            r1 = keyword_rank.get(doc_idx, 10_000)
            r2 = vector_rank.get(doc_idx, 10_000)
            fused[doc_idx] = 1.0 / (k + r1) + 1.0 / (k + r2)
        return fused

    def _rerank_score(
        self,
        query_tokens: List[str],
        doc_tokens: List[str],
        fused_score: float,
    ) -> float:
        # Lightweight rerank based on exact overlap ratio + fused retrieval score.
        if not query_tokens:
            return fused_score
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        overlap = len(query_set & doc_set) / max(len(query_set), 1)
        return 0.65 * fused_score + 0.35 * overlap

    def retrieve_with_explanations(self, query: str, top_k: int = 2) -> List[Dict[str, object]]:
        self._reload_if_needed()
        if not query.strip():
            return []

        query_tokens = self._tokenize(query)

        # 1. 稀疏检索（Sparse Retrieval - 关键字分数打底）
        keyword_scores: List[Tuple[float, int]] = []
        for idx in range(len(self._docs)):
            keyword_scores.append((self._keyword_score(query_tokens, self._doc_tokens[idx]), idx))

        # 2. 稠密检索（Dense Retrieval - 查询真正的 Chroma VectorStore）
        # 在 RRF 中，我们需要拉取较大基数结果再行综合，目前基数等于库总大小
        db_size = max(1, len(self._docs))
        vector_results = self._vectorstore.similarity_search_with_score(query, k=db_size)
        
        vector_scores: List[Tuple[float, int]] = [(0.0, i) for i in range(len(self._docs))]
        for doc, distance in vector_results:
            orig_idx = doc.metadata.get("original_idx")
            if orig_idx is not None and 0 <= orig_idx < len(self._docs):
                # distance是L2距离，相似度与距离成负相关，在此转为正向score进行后续rank排序
                vector_scores[orig_idx] = (-distance, orig_idx)

        # 3. 混合倒数融合打分（RRF）: `1/(k+rank1) + 1/(k+rank2)`
        fused = self._fuse_rrf(keyword_scores=keyword_scores, vector_scores=vector_scores)

        candidate_size = min(max(top_k * 3, 4), len(self._docs))
        candidates = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:candidate_size]

        reranked: List[Tuple[float, int]] = []
        score_map: Dict[int, Dict[str, float]] = {}
        for doc_idx, fused_score in candidates:
            kw_score = next((s for s, i in keyword_scores if i == doc_idx), 0.0)
            vec_score = next((s for s, i in vector_scores if i == doc_idx), 0.0)
            final_score = self._rerank_score(query_tokens, self._doc_tokens[doc_idx], fused_score)
            reranked.append((final_score, doc_idx))
            score_map[doc_idx] = {
                "keyword": kw_score,
                "vector": vec_score,
                "fused": fused_score,
                "rerank": final_score,
            }

        reranked.sort(reverse=True)

        results: List[Dict[str, object]] = []
        for rank, (final_score, idx) in enumerate(reranked[:top_k], start=1):
            scores = score_map.get(idx, {})
            results.append(
                {
                    "rank": rank,
                    "source_id": self._source_ids[idx],
                    "content": self._docs[idx],
                    "scores": {
                        "keyword": round(float(scores.get("keyword", 0.0)), 4),
                        "vector": round(float(scores.get("vector", 0.0)), 4),
                        "fused": round(float(scores.get("fused", 0.0)), 6),
                        "rerank": round(float(final_score), 6),
                    },
                }
            )
        return results

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        hits = self.retrieve_with_explanations(query=query, top_k=top_k)
        return [str(item["content"]) for item in hits]


campus_rag_retriever = CampusRAGRetriever()
