from pathlib import Path

from .base import BaseTool


class ReadFileTool(BaseTool):
    name: str = "read_file_tool"
    description: str = 'Read a file from the local disk and return its original content.'
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path of the file"
            }
        },
        "required": ["file_path"],
    }

    async def execute(self, file_path: str) -> str:

        read_path = Path(file_path)
        try:
            with read_path.open("r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return "Error: " + str(e)


