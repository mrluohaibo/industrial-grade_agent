import logging
import json
import time
from copy import deepcopy
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import END

from bz_agent.agents import coder_agent, browser_agent, url_to_markdown_agent, rag_agent
from bz_agent.agents.llm import get_llm_by_type
from bz_agent.config import TEAM_MEMBERS
from bz_agent.config.agents_map import AGENT_LLM_MAP
from bz_agent.prompts.template import apply_prompt_template
from .types import State, Router

logger = logging.getLogger(__name__)

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"


# def research_node(state: State) -> Command[Literal["supervisor"]]:
#     """Node for the researcher agent that performs research tasks."""
#     logger.info("Research agent starting task")
#     result = research_agent.invoke(state)
#     logger.info("Research agent completed task")
#     logger.debug(f"Research agent response: {result['messages'][-1].content}")
#     return Command(
#         update={
#             "messages": [
#                 HumanMessage(
#                     content=RESPONSE_FORMAT.format(
#                         "researcher", result["messages"][-1].content
#                     ),
#                     name="researcher",
#                 )
#             ]
#         },
#         goto="supervisor",
#     )


def code_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the coder agent that executes Python code."""
    logger.info("Code agent starting task")
    result = coder_agent.invoke(state)
    logger.info("Code agent completed task")
    logger.debug(f"Code agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "coder", result["messages"][-1].content
                    ),
                    name="coder",
                )
            ]
        },
        goto="supervisor",
    )


def browser_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the browser agent that performs web browsing tasks."""
    logger.info("Browser agent starting task")
    result = browser_agent.invoke(state)
    logger.info("Browser agent completed task")
    logger.debug(f"Browser agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "browser", result["messages"][-1].content
                    ),
                    name="browser",
                )
            ]
        },
        goto="supervisor",
    )


def supervisor_node(state: State) -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
    """Supervisor node that decides which agent should act next."""
    logger.info("Supervisor evaluating next action")
    messages = apply_prompt_template("supervisor", state)
    response = (
        get_llm_by_type(AGENT_LLM_MAP["supervisor"])
        .with_structured_output(Router)
        .invoke(messages)
    )
    goto = response["next"]
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Supervisor response: {response}")

    if goto == "FINISH":
        goto = "__end__"
        logger.info("Workflow completed")
    else:
        logger.info(f"Supervisor delegating to: {goto}")

    return Command(goto=goto, update={"next": goto})


def planner_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan")
    messages = apply_prompt_template("planner", state)
    # whether to enable deep thinking mode
    llm = get_llm_by_type("basic")
    if state.get("deep_thinking_mode"):
        llm = get_llm_by_type("reasoning")
    # Initialize with search_before_planning enabled, meaning direct third-party call results before planning
    # if state.get("search_before_planning"):
    #     searched_content = tavily_tool.invoke({"query": state["messages"][-1].content})
    #     messages = deepcopy(messages)
    #     messages[
    #         -1
    #     ].content += f"\n\n# Relative Search Results\n\n{json.dumps([{'title': elem['title'], 'content': elem['content']} for elem in searched_content], ensure_ascii=False)}"
    stream = llm.stream(messages)
    full_response = ""
    for chunk in stream:
        full_response += chunk.content
    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {full_response}")

    if full_response.startswith("```json"):
        full_response = full_response.removeprefix("```json")

    if full_response.endswith("```"):
        full_response = full_response.removesuffix("```")

    goto = "supervisor"
    try:
        json.loads(full_response)
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        goto = "__end__"

    return Command(
        update={
            "messages": [HumanMessage(content=full_response, name="planner")],
            "full_plan": full_response,
        },
        goto=goto,
    )

# state is the context from beginning to end
# def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]:
#     """Coordinator node that communicates with customers."""
#     logger.info("Coordinator talking.")
#     messages = apply_prompt_template("coordinator", state)
#     response = get_llm_by_type(AGENT_LLM_MAP["coordinator"]).invoke(messages)
#     logger.debug(f"Current state messages: {state['messages']}")
#     logger.debug(f"reporter response: {response}")
#
#     goto = "__end__"
#     if "handoff_to_planner" in response.content:
#         goto = "planner"
#
#     return Command(
#         goto=goto,
#     )


def reporter_node(state: State) -> Command[Literal["supervisor"]]:
    """Reporter node that writes a final report."""
    logger.info("Reporter write final report")
    messages = apply_prompt_template("reporter", state)
    response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(messages)
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"reporter response: {response}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format("reporter", response.content),
                    name="reporter",
                )
            ]
        },
        goto="supervisor",
    )


def url_to_markdown_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the url_to_markdown agent that extracts content from webpages."""
    logger.info("url_to_markdown agent starting task")
    start_time = time.time() * 1000
    result = url_to_markdown_agent.invoke(state)
    logger.info(f"url_to_markdown agent completed task, spending time: {time.time()*1000 - start_time}")
    logger.debug(f"url_to_markdown agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "url_to_markdown", result["messages"][-1].content
                    ),
                    name="url_to_markdown",
                )
            ]
        },
        goto="supervisor",
    )


def rag_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the RAG agent that performs knowledge retrieval tasks."""
    logger.info("RAG agent starting task")
    result = rag_agent.invoke(state)
    logger.info("RAG agent completed task")
    logger.debug(f"RAG agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "rag", result["messages"][-1].content
                    ),
                    name="rag",
                )
            ]
        },
        goto="supervisor",
    )
