from langgraph.graph import StateGraph, START

from .types import State
from .nodes import (
    code_node,
    url_to_markdown_node,
    browser_node,
    supervisor_node,
    reporter_node,
    planner_node,
)

# 这里不链接边嘛？
def build_graph():
    """Build and return the agent workflow graph."""
    builder = StateGraph(State)
    builder.add_edge(START, "planner")
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("url_to_markdown", url_to_markdown_node)
    builder.add_node("coder", code_node)
    builder.add_node("browser", browser_node)
    builder.add_node("reporter", reporter_node)
    return builder.compile()


