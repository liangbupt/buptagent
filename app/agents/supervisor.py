from typing import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_node: str

class RouteDecision(BaseModel):
    next_node: str
    reason: str

SUPERVISOR_PROMPT = """You are the Supervisor Agent for the BUPT Campus Smart Life Assistant.
Your job is to analyze the user's request and route it to the appropriate expert agent.
Available expert agents:
- 'academic_agent': Handle matters related to classes, classrooms, grades, exams.
- 'life_agent': Handle matters related to food, dining halls, campus life, second-hand market.
- 'interaction_agent': Handle matters related to second-hand trading and campus feedback.
- 'FINISH': If the user's request has been fully addressed or no further routing is needed.

Based on the conversation history, decide the next expert agent to route to.
"""

def create_supervisor_agent(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        ("messages", "{messages}")
    ])
    
    # We use structured output to enforce the route
    chain = prompt | llm.with_structured_output(RouteDecision)
    
    def supervisor_node(state: AgentState):
        decision = chain.invoke(state)
        return {"next_node": decision.next_node}
    
    return supervisor_node
