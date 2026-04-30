from typing import Annotated, Sequence, TypedDict, Literal, Optional
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_node: str
    route_rationale: str
    route_confidence: float
    extracted_slots: Optional[dict]

class RouteDecision(BaseModel):
    """Routing decision for the BUPT Campus Smart Life Assistant Supervisor."""
    route: Literal["academic_agent", "life_agent", "interaction_agent", "FINISH"] = Field(
        ..., description="The next agent to route to, or FINISH if the task is complete."
    )
    rationale: str = Field(
        ..., description="A short sentence explaining the rationale for the routing decision."
    )
    confidence: float = Field(
        ..., description="Confidence score between 0.0 and 1.0 for the routing decision."
    )
    extracted_slots: dict = Field(
        default_factory=dict, description="Extracted structured entities (e.g., location, time, preference, budget). Empty dict if none."
    )

SUPERVISOR_PROMPT = """You are the Supervisor Agent for the BUPT Campus Smart Life Assistant.
Your job is to analyze the user's request and route it to the appropriate expert agent.
Available expert agents:
- 'academic_agent': Handle matters related to classes, classrooms, grades, exams.
- 'life_agent': Handle matters related to food, dining halls, campus life, second-hand market.
- 'interaction_agent': Handle matters related to second-hand trading and campus feedback.
- 'FINISH': If the user's request has been fully addressed or no further routing is needed.

Follow this structured decision process internally:
1) Extract user intent entities (location, time, preference, budget) into extracted_slots.
2) Match intent to the most suitable expert agent to determine the route.
3) Provide a rationale and a confidence score.
"""

def create_supervisor_agent(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        MessagesPlaceholder(variable_name="messages")
    ])

    # 强制大模型按照 Pydantic Schema 输出结构化 JSON
    structured_llm = llm.with_structured_output(RouteDecision)
    chain = prompt | structured_llm

    def supervisor_node(state: AgentState):
        decision: RouteDecision = chain.invoke(state)
        
        return {
            "next_node": decision.route, 
            "route_rationale": decision.rationale, 
            "route_confidence": decision.confidence,
            "extracted_slots": decision.extracted_slots
        }
    
    return supervisor_node
