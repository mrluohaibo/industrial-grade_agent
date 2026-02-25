from langgraph.prebuilt import create_react_agent

from bz_agent.prompts import apply_prompt_template
from bz_agent.tools import (
    bash_tool,
    page_html_tool,
    python_repl_tool,
    browser_tool,
    read_file_tool,
    rag_knowledge_retrieval_async,
)

from .llm import get_llm_by_type
from bz_agent.config.agents_map import AGENT_LLM_MAP

# Create agents using configured LLM types
# research_agent = create_react_agent(
#     get_llm_by_type(AGENT_LLM_MAP["researcher"]),
#     tools=[tavily_tool, crawl_tool],
#     prompt=lambda state: apply_prompt_template("researcher", state),
# )

coder_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["coder"]),
    tools=[python_repl_tool, bash_tool],
    prompt=lambda state: apply_prompt_template("coder", state),
)

browser_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["browser"]),
    tools=[browser_tool],
    prompt=lambda state: apply_prompt_template("browser", state),
)


url_to_markdown_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["url_to_markdown"]),
    tools=[page_html_tool,read_file_tool],
    prompt=lambda state: apply_prompt_template("url_to_markdown", state),
)

# RAG Agent for knowledge retrieval
rag_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["rag"]),
    tools=[rag_knowledge_retrieval_async],
    prompt=lambda state: apply_prompt_template("rag", state),
)
