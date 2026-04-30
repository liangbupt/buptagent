from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from app.agents.supervisor import create_supervisor_agent, AgentState
from app.agents.workers import create_academic_agent, create_life_agent, create_interaction_agent
from app.core.config import settings
from app.memory.hybrid_memory import hybrid_memory

def build_graph(api_key: str | None = None, base_url: str | None = None, model: str | None = None):
    resolved_key = (api_key or settings.OPENAI_API_KEY or "").strip()
    resolved_base_url = (base_url or settings.OPENAI_BASE_URL or "").strip()
    resolved_model = (model or settings.LLM_MODEL or "").strip()
    llm = (
        ChatOpenAI(
            model=resolved_model,
            api_key=resolved_key,
            base_url=resolved_base_url or None,
            timeout=45,
        )
        if resolved_key and resolved_model
        else None
    )
    
    if not llm:
        return None

    # ✨ 第17题优化：前置的“语义缓存拦截”节点
    def semantic_cache_node(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1].content
        
        # 尝试检查之前是否已经有非常相似的问题回答了
        cached_response = hybrid_memory.check_semantic_cache(query_text=last_message, threshold=0.95)
        
        # 缓存命中，无需调后面的大模型，在0.1s内短路返回！
        if cached_response:
            return {"next_node": "FINISH", "messages": [AIMessage(content=cached_response, name="semantic_plugin")]}
            
        # 没有命中，走常规的 Supervisor 节点
        return {"next_node": "supervisor"}
        
    supervisor_node = create_supervisor_agent(llm)
    academic_agent = create_academic_agent(llm)
    life_agent = create_life_agent(llm)
    interaction_agent = create_interaction_agent(llm)
    
    # 我们拦截处理结果以回写 semantic cache
    def _save_to_cache(original_query: str, answer: str):
        hybrid_memory.save_semantic_cache(original_query, answer)

    def academic_node(state: AgentState):
        query = state["messages"][0].content
        result = academic_agent.invoke({"messages": state["messages"]})
        msg = AIMessage(content=result["messages"][-1].content, name="academic_agent")
        _save_to_cache(query, msg.content)
        return {"messages": [msg]}
        
    def life_node(state: AgentState):
        query = state["messages"][0].content
        result = life_agent.invoke({"messages": state["messages"]})
        msg = AIMessage(content=result["messages"][-1].content, name="life_agent")
        _save_to_cache(query, msg.content)
        return {"messages": [msg]}

    def interaction_node(state: AgentState):
        query = state["messages"][0].content
        result = interaction_agent.invoke({"messages": state["messages"]})
        msg = AIMessage(content=result["messages"][-1].content, name="interaction_agent")
        _save_to_cache(query, msg.content)
        return {"messages": [msg]}

    workflow = StateGraph(AgentState)
    
    workflow.add_node("semantic_cache", semantic_cache_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("academic_agent", academic_node)
    workflow.add_node("life_agent", life_node)
    workflow.add_node("interaction_agent", interaction_node)
    
    workflow.add_conditional_edges(
        "semantic_cache",
        lambda state: state["next_node"],
        {
            "FINISH": END,            # 命中缓存，直接结束
            "supervisor": "supervisor" # 未命中，把请求传给 Supervisor 正常处理
        }
    )
    
    workflow.add_edge("academic_agent", "supervisor")
    workflow.add_edge("life_agent", "supervisor")
    workflow.add_edge("interaction_agent", "supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_node"],
        {
            "academic_agent": "academic_agent",
            "life_agent": "life_agent",
            "interaction_agent": "interaction_agent",
            "FINISH": END
        }
    )
    
    workflow.set_entry_point("semantic_cache")
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

graph = build_graph()
