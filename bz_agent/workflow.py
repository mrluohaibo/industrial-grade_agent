import logging
import re
from typing import Optional

from bz_agent.config import TEAM_MEMBERS
from bz_agent.graph import build_graph
from bz_agent.storage import context_middleware
from utils.logger_config import logger


# Create the graph
graph = build_graph()


def run_agent_workflow(
    user_input: str,
    session_id: Optional[str] = None
):
    """Run the agent workflow with the given user input.

    Args:
        user_input: The user's query or request
        session_id: Optional session ID for conversation context persistence.
                    If not provided, a new session will be created.
                    If provided, the conversation history will be loaded and continued.

    Returns:
        The final state after the workflow completes, including the session_id
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    # 构建初始 state
    state = {
        # Constants
        "TEAM_MEMBERS": TEAM_MEMBERS,
        # Runtime Variables
        "messages": [{"role": "user", "content": user_input}],
        "deep_thinking_mode": True,
        "search_before_planning": False,
        # Session management
        "session_id": session_id,
    }

    # 预处理：加载会话上下文（如果session_id存在则加载历史，否则创建新会话）
    state = context_middleware.pre_process(state)

    logger.info(f"Starting workflow with user input: {user_input}, session_id: {state.get('session_id')}")

    # 执行 workflow
    result = graph.invoke(state)

    # 后处理：保存消息到 MongoDB
    result = context_middleware.post_process(result)

    logger.debug(f"Final workflow state: {result}")
    logger.info(f"Workflow completed successfully, session_id: {result.get('session_id')}")
    return result


def request_url_content_to_markdown(url: str, session_id: Optional[str] = None):
    """
    Request webpage content and convert to markdown format.

    Args:
        url: The webpage URL to fetch
        session_id: Optional session ID for conversation continuity

    Returns:
        The extracted markdown content or None
    """
    user_query = f"将网页 {url} 中的标题+发表时间+正文 提取出来并以markdown格式输出，注意只返回markdown内容"
    result = run_agent_workflow(user_input=user_query, session_id=session_id)
    if len(result["messages"]) == 0:
        return None
    last_content = result["messages"][-1].content
    match = re.search(r'<response>(.*?)</response>', last_content, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return content
    else:
        return None

if __name__ == "__main__":
    # print(graph.get_graph().draw_mermaid())

    # user_query = "将网页 https://cj.sina.com.cn/articles/view/2023821012/78a10ed402001rnpw 中的标题+正文 提取出来并以markdown格式输出，注意只返回markdown内容"
    # result = run_agent_workflow(user_input=user_query)
    # logger.debug(f"Final workflow state: {result}")
    #
    # for message in result["messages"]:
    #     role = message.type
    #     print(f"\n[{role.upper()}]: {message.content}")

    result = request_url_content_to_markdown("https://cj.sina.com.cn/articles/view/2023821012/78a10ed402001rnpw","dsafsdjifdjsl")
    logger.info(f"Final workflow state: {result}")