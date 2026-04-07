from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from app.agents.supervisor import create_supervisor_agent, AgentState
from app.agents.workers import create_academic_agent, create_life_agent, create_interaction_agent
from app.core.config import settings

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
        
    supervisor_node = create_supervisor_agent(llm)
    academic_agent = create_academic_agent(llm)
    life_agent = create_life_agent(llm)
    interaction_agent = create_interaction_agent(llm)
    
    def academic_node(state: AgentState):
        result = academic_agent.invoke({"messages": state["messages"]})
        msg = AIMessage(content=result["messages"][-1].content, name="academic_agent")
        return {"messages": [msg]}
        
    def life_node(state: AgentState):
        result = life_agent.invoke({"messages": state["messages"]})
        msg = AIMessage(content=result["messages"][-1].content, name="life_agent")
        return {"messages": [msg]}

    def interaction_node(state: AgentState):
        result = interaction_agent.invoke({"messages": state["messages"]})
        msg = AIMessage(content=result["messages"][-1].content, name="interaction_agent")
        return {"messages": [msg]}

    workflow = StateGraph(AgentState)
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("academic_agent", academic_node)
    workflow.add_node("life_agent", life_node)
    workflow.add_node("interaction_agent", interaction_node)
    
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
    
    workflow.set_entry_point("supervisor")
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

graph = build_graph()
