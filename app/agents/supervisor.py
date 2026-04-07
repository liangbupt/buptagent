from typing import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_node: str
    route_rationale: str
    route_confidence: float

SUPERVISOR_PROMPT = """You are the Supervisor Agent for the BUPT Campus Smart Life Assistant.
Your job is to analyze the user's request and route it to the appropriate expert agent.
Available expert agents:
- 'academic_agent': Handle matters related to classes, classrooms, grades, exams.
- 'life_agent': Handle matters related to food, dining halls, campus life, second-hand market.
- 'interaction_agent': Handle matters related to second-hand trading and campus feedback.
- 'FINISH': If the user's request has been fully addressed or no further routing is needed.

Follow this structured decision process internally:
1) Extract user intent entities (location, time, preference, budget).
2) Match intent to the most suitable expert agent.
3) Check whether the conversation appears completed.

Return exactly in this format:
ROUTE: <academic_agent|life_agent|interaction_agent|FINISH>
RATIONALE: <one short sentence>
CONFIDENCE: <0.0-1.0>
"""

def create_supervisor_agent(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        MessagesPlaceholder(variable_name="messages")
    ])

    chain = prompt | llm

    def _extract_route_reason_confidence(text: str) -> tuple[str, str, float]:
        value = (text or "").strip().lower()
        route = "academic_agent"
        reason = "Fallback to academic routing due to unrecognized route output."
        confidence = 0.5

        for line in (text or "").splitlines():
            lower_line = line.lower().strip()
            if lower_line.startswith("route:"):
                token = lower_line.replace("route:", "").strip()
                if token in {"academic_agent", "life_agent", "interaction_agent", "finish"}:
                    route = "FINISH" if token == "finish" else token
            if lower_line.startswith("rationale:"):
                reason = line.split(":", 1)[1].strip() or reason
            if lower_line.startswith("confidence:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    parsed = float(raw)
                    confidence = max(0.0, min(1.0, parsed))
                except ValueError:
                    confidence = confidence

        if "academic_agent" in value or "academic" in value or "class" in value or "教室" in value:
            route = "academic_agent"
        elif "life_agent" in value or "dining" in value or "canteen" in value or "食堂" in value:
            route = "life_agent"
        elif "interaction_agent" in value or "flea" in value or "反馈" in value or "交易" in value:
            route = "interaction_agent"
        elif "finish" in value:
            route = "FINISH"

        return route, reason, confidence
    
    def supervisor_node(state: AgentState):
        decision = chain.invoke(state)
        next_node, rationale, confidence = _extract_route_reason_confidence(getattr(decision, "content", ""))
        return {"next_node": next_node, "route_rationale": rationale, "route_confidence": confidence}
    
    return supervisor_node
