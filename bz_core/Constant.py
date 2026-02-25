from pathlib import Path
from enum import Enum


current_file = Path(__file__).resolve()

root_path = current_file.parent.parent  # 跟目录


class NewsType(Enum):
    NEWS = 1
    NOTICE = 2
