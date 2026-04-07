from mcp.server.fastmcp import FastMCP
from app.tools.data_provider import provider

mcp = FastMCP(name="buptagent-local-mcp")


@mcp.tool()
def get_free_classrooms(building: str, time_slot: str) -> str:
    """Find empty classrooms in a given building and time slot."""
    return provider.get_free_classrooms(building=building, time_slot=time_slot)


@mcp.tool()
def get_audit_course_suggestion(topic: str, preferred_time: str = "afternoon") -> str:
    """Recommend audit courses based on topic and preferred time."""
    return provider.get_audit_course_suggestion(topic=topic, preferred_time=preferred_time)


@mcp.tool()
def get_dining_recommendation(preference: str) -> str:
    """Get dining hall or food recommendation based on user preference."""
    return provider.get_dining_recommendation(preference=preference)


@mcp.tool()
def get_flea_market_items(keyword: str, budget: str = "不限") -> str:
    """Search flea market items by keyword and budget."""
    return provider.get_flea_market_items(keyword=keyword, budget=budget)


@mcp.tool()
def get_campus_feedback(topic: str) -> str:
    """Get summarized campus feedback for a topic."""
    return provider.get_campus_feedback(topic=topic)


if __name__ == "__main__":
    mcp.run(transport="stdio")
