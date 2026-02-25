import logging
from langchain_community.tools.file_management import WriteFileTool,ReadFileTool
from .decorators import create_logged_tool

logger = logging.getLogger(__name__)

# Initialize file management tool with logging
LoggedWriteFile = create_logged_tool(WriteFileTool)
write_file_tool = LoggedWriteFile()


LoggedReadFile = create_logged_tool(ReadFileTool)
read_file_tool = LoggedReadFile()