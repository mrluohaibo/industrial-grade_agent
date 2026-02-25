import asyncio

from pydantic import Field

from bz_agent.native_agent.toolcall import ToolCallAgent

from bz_agent.native_agent.tools import Terminate,PageHtmlTool,ToolCollection,ReadFileTool

from utils.logger_config import logger

SYSTEM_PROMPT = """
You are an AI agent designed to use tools to obtain the source code of web pages and extract the main content of the webpage, outputting it in Markdown format.

"""

NEXT_STEP_PROMPT = """
Based on user needs, break down the problem and use different tools step by step to solve it.
# Note
1. Each step select the most appropriate tool proactively (ONLY ONE).
2. After using each tool, clearly explain the execution results and suggest the next steps.
3. When observation with Error, review and fix it.
4. If you want to stop interaction, use `terminate` tool/function call
5. After obtaining the HTML source code, analyze the main content to generate Markdown. Do not perform any additional actions.

"""



class UrlToMarkdownAgent(ToolCallAgent):
    """


    """

    name: str = "url_to_markdown_agent"
    description: str = "An agent that use tools to obtain the source code of web pages and extract the main content of the webpage, outputting it in Markdown format"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 15000
    max_steps: int = 20

    # Add general-purpose tools to the tool collection
    # 当个agent会含有一组未实现特定功能的一组工具
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            Terminate(),
            PageHtmlTool(),
            ReadFileTool()
        )
    )


async def test_agent():
    agent = UrlToMarkdownAgent()
    prompt = "将网页 https://cj.sina.com.cn/articles/view/2023821012/78a10ed402001rnpw 中的标题+正文 提取出来并以markdown格式输出，注意只返回标题加内容的markdown内容,不需要总结性语句"
    result = await agent.run(prompt)
    logger.info(result)

if __name__ == "__main__":
    asyncio.run(test_agent())
