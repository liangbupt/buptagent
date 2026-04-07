from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.graph import build_graph
from app.core.config import settings
from langchain_core.messages import HumanMessage, SystemMessage
from app.memory.hybrid_memory import hybrid_memory
from app.rag.campus_rag import campus_rag_retriever

router = APIRouter()

graph_cache = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str
    api_key: str | None = None
    api_base: str | None = None
    model: str | None = None

class ChatResponse(BaseModel):
    reply: str


def _get_graph_for_request(api_key: str | None):
    return _get_graph_for_request_with_base(api_key=api_key, api_base=None, model=None)


def _get_graph_for_request_with_base(api_key: str | None, api_base: str | None, model: str | None):
    resolved_key = (api_key or settings.OPENAI_API_KEY or "").strip()
    resolved_base = (api_base or settings.OPENAI_BASE_URL or "").strip()
    resolved_model = (model or settings.LLM_MODEL or "").strip()
    if not resolved_key:
        return None

    cache_key = f"{resolved_base}::{resolved_model}::{resolved_key}"
    if cache_key not in graph_cache:
        graph_cache[cache_key] = build_graph(
            api_key=resolved_key,
            base_url=resolved_base,
            model=resolved_model,
        )

    return graph_cache[cache_key]


def _build_memory_context(user_id: str, user_message: str, rag_hits: list[dict]) -> str:
    recent_turns = hybrid_memory.get_recent_turns(user_id=user_id, limit=3)
    long_term = hybrid_memory.recall_long_term_memory(user_id=user_id, query_text=user_message, top_k=2)

    lines = ["Conversation memory context:"]
    if recent_turns:
        lines.append("Recent turns:")
        for turn in recent_turns:
            lines.append(f"- user: {turn.get('user', '')}")
            lines.append(f"- assistant: {turn.get('assistant', '')}")

    if long_term:
        lines.append("Long-term profile hints:")
        for item in long_term:
            lines.append(f"- {item}")

    if rag_hits:
        lines.append("Campus knowledge snippets:")
        for hit in rag_hits:
            source_id = hit.get("source_id", "unknown")
            content = hit.get("content", "")
            scores = hit.get("scores", {})
            lines.append(
                f"- [{source_id}] {content} "
                f"(kw={scores.get('keyword', 0)}, vec={scores.get('vector', 0)}, "
                f"fused={scores.get('fused', 0)}, rerank={scores.get('rerank', 0)})"
            )

    return "\n".join(lines)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    graph = _get_graph_for_request_with_base(request.api_key, request.api_base, request.model)
    if not graph:
        raise HTTPException(
            status_code=400,
            detail="Missing API key. Please provide api_key in request or configure OPENAI_API_KEY in .env",
        )
        
    try:
        rag_hits = campus_rag_retriever.retrieve_with_explanations(query=request.message, top_k=2)
        memory_context = _build_memory_context(request.user_id, request.message, rag_hits)
        inputs = {
            "messages": [
                SystemMessage(content=memory_context),
                HumanMessage(content=request.message),
            ]
        }
        # Thread_id is used by MemorySaver for session management
        config = {"configurable": {"thread_id": request.user_id}}
        
        result = graph.invoke(inputs, config=config)
        final_message = result["messages"][-1].content
        route_rationale = str(result.get("route_rationale", ""))
        final_route = str(result.get("next_node", ""))
        route_confidence = float(result.get("route_confidence", 0.5))

        if rag_hits:
            refs = [f"{idx + 1}. [{h.get('source_id')}] {h.get('content')}" for idx, h in enumerate(rag_hits)]
            final_message = final_message + "\n\n参考片段来源:\n" + "\n".join(refs)

        hybrid_memory.save_turn(
            user_id=request.user_id,
            user_message=request.message,
            assistant_message=final_message,
        )
        hybrid_memory.save_route_audit(
            user_id=request.user_id,
            route=final_route,
            rationale=route_rationale,
            user_message=request.message,
            confidence=route_confidence,
        )

        if hybrid_memory.should_store_long_term(request.message):
            hybrid_memory.add_long_term_memory(
                user_id=request.user_id,
                text=f"User preference: {request.message}",
            )

        return ChatResponse(reply=final_message)
    except Exception as e:
        detail = str(e)
        if "Connection error" in detail or "handshake" in detail.lower() or "ssl" in detail.lower():
            detail = (
                "Connection error: 当前网络无法直连 OpenAI。"
                "请在前端填写可用的 API Base URL（OpenAI 兼容网关），"
                "或在 .env 配置 OPENAI_BASE_URL 后重试。"
            )
        elif "InvalidEndpointOrModel.NotFound" in detail or "does not exist" in detail:
            detail = (
                "模型不可用：当前网关不支持该模型。"
                "请在前端的模型输入框改成该网关可用模型（例如 deepseek-chat / qwen-plus 等）。"
            )
        raise HTTPException(status_code=500, detail=detail)


@router.delete("/memory/{user_id}")
async def delete_user_memory(user_id: str, scope: str = "all"):
    try:
        hybrid_memory.delete_user_memory(user_id=user_id, scope=scope)
        return {"ok": True, "user_id": user_id, "scope": scope}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {exc}")


@router.get("/audit/{user_id}")
async def get_route_audit(user_id: str, limit: int = 20):
    try:
        rows = hybrid_memory.get_route_audit(user_id=user_id, limit=limit)
        return {"user_id": user_id, "count": len(rows), "items": rows}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch route audit: {exc}")
