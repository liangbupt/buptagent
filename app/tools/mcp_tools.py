from langchain_core.tools import tool
from app.tools.mcp_client import mcp_client


def _local_get_free_classrooms(building: str, time_slot: str) -> str:
    return f"Mock: In {building} at {time_slot}, classrooms 101, 102 are free."


def _local_get_audit_course_suggestion(topic: str, preferred_time: str = "afternoon") -> str:
    return (
        f"Mock: For topic '{topic}', suggested audit course is 'AI Fundamentals' "
        f"with open seats around {preferred_time}."
    )


def _local_get_dining_recommendation(preference: str) -> str:
    if "spicy" in preference.lower() or "辣" in preference:
        return "Mock: Recommend the spicy hotpot on the 2nd floor of Student Dining Hall."
    return "Mock: Recommend the light set meal on the 1st floor of Xinyuan Dining Hall."


def _local_get_flea_market_items(keyword: str, budget: str = "不限") -> str:
    return (
        f"Mock: Found flea market items for '{keyword}' under budget '{budget}': "
        "used monitor 300 CNY, desk lamp 45 CNY."
    )


def _local_get_campus_feedback(topic: str) -> str:
    return f"Mock: Recent campus feedback on '{topic}' is mostly positive with suggestions on queue time."


def _mcp_call(name: str, arguments: dict, fallback):
    try:
        return mcp_client.call_tool(name=name, arguments=arguments)
    except Exception:
        return fallback(**arguments)

@tool
def get_free_classrooms(building: str, time_slot: str) -> str:
    """Find empty classrooms in a given building and time slot."""
    return _mcp_call(
        "get_free_classrooms",
        {"building": building, "time_slot": time_slot},
        _local_get_free_classrooms,
    )


@tool
def get_audit_course_suggestion(topic: str, preferred_time: str = "afternoon") -> str:
    """Recommend audit courses based on topic and preferred time."""
    return _mcp_call(
        "get_audit_course_suggestion",
        {"topic": topic, "preferred_time": preferred_time},
        _local_get_audit_course_suggestion,
    )

@tool
def get_dining_recommendation(preference: str) -> str:
    """Get dining hall or food recommendation based on user preference."""
    return _mcp_call(
        "get_dining_recommendation",
        {"preference": preference},
        _local_get_dining_recommendation,
    )


@tool
def get_flea_market_items(keyword: str, budget: str = "不限") -> str:
    """Search flea market items by keyword and budget."""
    return _mcp_call(
        "get_flea_market_items",
        {"keyword": keyword, "budget": budget},
        _local_get_flea_market_items,
    )


@tool
def get_campus_feedback(topic: str) -> str:
    """Get summarized campus feedback for a topic such as canteen or study spaces."""
    return _mcp_call(
        "get_campus_feedback",
        {"topic": topic},
        _local_get_campus_feedback,
    )

academic_tools = [get_free_classrooms, get_audit_course_suggestion]
life_tools = [get_dining_recommendation]
interaction_tools = [get_flea_market_items, get_campus_feedback]
