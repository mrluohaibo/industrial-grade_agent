from browser_use.agent.views import AgentHistoryList
from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Type
from langchain.tools import BaseTool
import asyncio

from .decorators import create_logged_tool
from utils.page_snapshot import screenShot_tool
from utils.logger_config import logger

class UrlInput(BaseModel):
    """Input for WriteFileTool."""

    url: str = Field(..., description="the url of the web page")


class PageHtmlTool(BaseTool):
    name: ClassVar[str] = "pageSnapshot"
    args_schema: Type[BaseModel] = UrlInput
    description: ClassVar[str] = (
        "Use this tool to get html source of webpages. Input should be a web page url, such as 'http://yidian.weather.com.cn/mweather15d/101200805.shtml',Returns the path to a locally stored html"
    )

    async def do_url_html(self, url) :
        """Snapshot the web page."""
        save_file_path = screenShot_tool.get_url_html(url)
        logger.info(f"get url: {url} content saved to: {save_file_path}")
        return save_file_path

    def _run(self, url: str) -> str:
        """Run the browser task synchronously."""

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.do_url_html(url))
                return result
            finally:
                loop.close()
        except Exception as e:
            return f"Error executing browser task: {str(e)}"

    async def _arun(self, url: str) -> str:
        """Run the browser task asynchronously."""

        try:
            result = await self.do_url_html(url)
            return result
        except Exception as e:
            return f"Error executing browser task: {str(e)}"


pageHtmlTool = create_logged_tool(PageHtmlTool)
page_html_tool = pageHtmlTool()
