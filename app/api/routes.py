from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.graph import build_graph
from app.core.config import settings
from langchain_core.messages import HumanMessage, SystemMessage
from app.memory.hybrid_memory import hybrid_memory

router = APIRouter()

graph_cache = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str
    api_key: str | None = None

class ChatResponse(BaseModel):
    reply: str


def _get_graph_for_request(api_key: str | None):
    resolved_key = (api_key or settings.OPENAI_API_KEY or "").strip()
    if not resolved_key:
        return None

    if resolved_key not in graph_cache:
        graph_cache[resolved_key] = build_graph(api_key=resolved_key)

    return graph_cache[resolved_key]


def _build_memory_context(user_id: str, user_message: str) -> str:
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

    return "\n".join(lines)


def _should_store_long_term(message: str) -> bool:
    keywords = [
        "喜欢", "偏好", "不吃", "爱吃", "预算", "习惯", "长期", "常去",
        "prefer", "favorite", "usually", "budget",
    ]
    lower_msg = message.lower()
    return any(k in message or k in lower_msg for k in keywords)

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    graph = _get_graph_for_request(request.api_key)
    if not graph:
        raise HTTPException(
            status_code=400,
            detail="Missing API key. Please provide api_key in request or configure OPENAI_API_KEY in .env",
        )
        
    try:
        memory_context = _build_memory_context(request.user_id, request.message)
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

        hybrid_memory.save_turn(
            user_id=request.user_id,
            user_message=request.message,
            assistant_message=final_message,
        )
        if _should_store_long_term(request.message):
            hybrid_memory.add_long_term_memory(
                user_id=request.user_id,
                text=f"User preference: {request.message}",
            )

        return ChatResponse(reply=final_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
