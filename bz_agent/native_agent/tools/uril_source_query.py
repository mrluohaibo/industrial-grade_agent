from utils.page_snapshot import screenShot_tool
from .base import BaseTool
from utils.logger_config import logger

class PageHtmlTool(BaseTool):
    name: str = "page_html_tool"
    description: str = 'This tool primarily retrieves the HTML source code of webpages, saves the HTML source code to the local disk, and returns the absolute path of the saved HTML file.'
    parameters: dict = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The url of the webpage"
            }
        },
        "required": ["url"],
    }

    async def execute(self, url: str) -> str:
        """Snapshot the web page."""
        save_file_path = screenShot_tool.get_url_html(url)
        logger.info(f"get url: {url} content saved to: {save_file_path}")
        return save_file_path


