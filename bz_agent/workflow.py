import logging
import re
import sys

from bz_agent.config import TEAM_MEMBERS
from bz_agent.graph import build_graph
from utils.logger_config import logger


# Create the graph
graph = build_graph()


def run_agent_workflow(user_input: str):
    """Run the agent workflow with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging

    Returns:
        The final state after the workflow completes
    """
    if not user_input:
        raise ValueError("Input could not be empty")



    logger.info(f"Starting workflow with user input: {user_input}")
    '''
    {
            # Constants
            "TEAM_MEMBERS": TEAM_MEMBERS,
            # Runtime Variables
            "messages": [{"role": "user", "content": user_input}],
            "deep_thinking_mode": True,
            "search_before_planning": True,
        }
        
        初始化的State 
        先从start 节点开始
        
    '''
    result = graph.invoke(
        {
            # Constants
            "TEAM_MEMBERS": TEAM_MEMBERS,
            # Runtime Variables
            "messages": [{"role": "user", "content": user_input}],
            "deep_thinking_mode": True,
            "search_before_planning": False,
        }
    )
    logger.debug(f"Final workflow state: {result}")
    logger.info("Workflow completed successfully")
    return result


def request_url_content_to_markdown(url: str):
    user_query = f"将网页 {url} 中的标题+发表时间+正文 提取出来并以markdown格式输出，注意只返回markdown内容"
    result = run_agent_workflow(user_input=user_query)
    if len(result["messages"]) == 0:
        return None
    last_content =  result["messages"][-1].content
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

    result = request_url_content_to_markdown("https://cj.sina.com.cn/articles/view/2023821012/78a10ed402001rnpw")
    logger.info(f"Final workflow state: {result}")