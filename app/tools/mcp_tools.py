from langchain_core.tools import tool

@tool
def get_free_classrooms(building: str, time_slot: str) -> str:
    """Find empty classrooms in a given building and time slot."""
    return f"Mock: In {building} at {time_slot}, classrooms 101, 102 are free."


@tool
def get_audit_course_suggestion(topic: str, preferred_time: str = "afternoon") -> str:
    """Recommend audit courses based on topic and preferred time."""
    return (
        f"Mock: For topic '{topic}', suggested audit course is 'AI Fundamentals' "
        f"with open seats around {preferred_time}."
    )

@tool
def get_dining_recommendation(preference: str) -> str:
    """Get dining hall or food recommendation based on user preference."""
    if "spicy" in preference.lower() or "辣" in preference:
        return "Mock: Recommend the spicy hotpot on the 2nd floor of Student Dining Hall."
    return "Mock: Recommend the light set meal on the 1st floor of Xinyuan Dining Hall."


@tool
def get_flea_market_items(keyword: str, budget: str = "不限") -> str:
    """Search flea market items by keyword and budget."""
    return (
        f"Mock: Found flea market items for '{keyword}' under budget '{budget}': "
        "used monitor 300 CNY, desk lamp 45 CNY."
    )


@tool
def get_campus_feedback(topic: str) -> str:
    """Get summarized campus feedback for a topic such as canteen or study spaces."""
    return f"Mock: Recent campus feedback on '{topic}' is mostly positive with suggestions on queue time."

academic_tools = [get_free_classrooms, get_audit_course_suggestion]
life_tools = [get_dining_recommendation]
interaction_tools = [get_flea_market_items, get_campus_feedback]
