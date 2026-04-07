from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from app.tools.mcp_tools import academic_tools, life_tools, interaction_tools

def create_academic_agent(llm: ChatOpenAI):
    prompt = "You are the Academic Expert Agent. Use tools to help students with academic and classroom queries."
    return create_react_agent(llm, tools=academic_tools, state_modifier=prompt)

def create_life_agent(llm: ChatOpenAI):
    prompt = "You are the Life Expert Agent. Use tools to help students with campus life and food queries."
    return create_react_agent(llm, tools=life_tools, state_modifier=prompt)


def create_interaction_agent(llm: ChatOpenAI):
    prompt = (
        "You are the Interaction Expert Agent. Use tools to help with flea market and "
        "campus feedback queries."
    )
    return create_react_agent(llm, tools=interaction_tools, state_modifier=prompt)
