import os
import re
from datetime import datetime
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState

from utils.logger_config import logger

# Prompt 加载源配置
# "file" - 仅从文件加载
# "mongo" - 仅从 MongoDB 加载
# "mongo_fallback" - 优先 MongoDB，不存在则回退文件
PROMPT_SOURCE = os.getenv("PROMPT_SOURCE", "mongo_fallback")

# Prompt 文件目录（用于文件加载或回退）
PROMPTS_DIR = os.path.join(os.path.dirname(__file__))


def get_prompt_from_mongo(prompt_name: str) -> Optional[str]:
    """
    从 MongoDB 获取 prompt 模板

    Args:
        prompt_name: prompt 名称

    Returns:
        prompt 模板内容，如果不存在则返回 None
    """
    try:
        from bz_agent.storage import prompt_store
        return prompt_store.get_prompt(prompt_name)
    except ImportError:
        logger.warning("Prompt store module not available, skipping MongoDB load")
        return None
    except Exception as e:
        logger.error(f"Error loading prompt from MongoDB: {e}")
        return None


def _get_prompt_from_file(prompt_name: str) -> Optional[str]:
    """
    从文件获取 prompt 模板

    Args:
        prompt_name: prompt 名称

    Returns:
        prompt 模板内容，如果文件不存在则返回 None
    """
    try:
        file_path = os.path.join(PROMPTS_DIR, f"{prompt_name}.md")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading prompt file: {e}")
        return None


def _process_template(template: str) -> str:
    """
    处理模板字符串，转换变量语法

    转换规则:
    1. { -> {{ (转义）
    2. } -> }} (转义）
    3. <<VAR>> -> {VAR} (我们的变量语法）

    Args:
        template: 原始模板字符串

    Returns:
        处理后的模板字符串
    """
    # 转义花括号
    template = template.replace("{", "{{").replace("}", "}}")
    # 将 <<VAR>> 替换为 {VAR}
    template = re.sub(r"<<([^>>]+)>>", r"{\1}", template)
    return template


def get_prompt_template(prompt_name: str) -> str:
    """
    获取 prompt 模板，支持多种来源

    优先级:
    1. PROMPT_SOURCE="mongo" -> 仅从 MongoDB 加载
    2. PROMPT_SOURCE="mongo_fallback" -> 先 MongoDB，不存在则文件
    3. PROMPT_SOURCE="file" -> 仅从文件加载（默认）

    Args:
        prompt_name: prompt 名称

    Returns:
        处理后的模板字符串

    Raises:
        ValueError: 如果找不到 prompt 模板
    """
    logger.debug(f"Loading prompt '{prompt_name}' from source: {PROMPT_SOURCE}")

    template = None

    if PROMPT_SOURCE == "mongo":
        template = get_prompt_from_mongo(prompt_name)
        if not template:
            raise ValueError(f"Prompt '{prompt_name}' not found in MongoDB (source: {PROMPT_SOURCE})")

    elif PROMPT_SOURCE == "mongo_fallback":
        template = get_prompt_from_mongo(prompt_name)
        if not template:
            logger.info(f"Prompt '{prompt_name}' not in MongoDB, falling back to file")
            template = _get_prompt_from_file(prompt_name)
    else:  # "file"
        template = _get_prompt_from_file(prompt_name)

    if not template:
        raise ValueError(f"Prompt '{prompt_name}' not found (source: {PROMPT_SOURCE})")

    # 处理模板语法
    return _process_template(template)


def apply_prompt_template(prompt_name: str, state: AgentState) -> list:
    """
    应用 prompt 模板到状态

    Args:
        prompt_name: prompt 名称
        state: Agent 状态

    Returns:
        消息列表，包含系统 prompt 和历史消息
    """
    system_prompt = PromptTemplate(
        input_variables=["CURRENT_TIME"],
        template=get_prompt_template(prompt_name),
    ).format(
        CURRENT_TIME=datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        **state
    )

    return [{"role": "system", "content": system_prompt}] + state["messages"]


def import_prompts_to_mongo(prompts_dir: Optional[str] = None) -> int:
    """
    将目录下所有 .md prompt 文件导入到 MongoDB

    用于初始化 prompt 数据库或迁移

    Args:
        prompts_dir: prompt 文件目录，默认为当前 prompts 目录

    Returns:
        导入的 prompt 数量
    """
    from bz_agent.storage import prompt_store

    if prompts_dir is None:
        prompts_dir = PROMPTS_DIR

    logger.info(f"Importing prompts from directory: {prompts_dir}")

    count = prompt_store.import_all_from_directory(prompts_dir)
    logger.info(f"Successfully imported {count} prompts to MongoDB")

    return count


def reload_prompt_cache() -> None:
    """
    清除可能的缓存并重新加载

    当前实现没有缓存，但这个函数预留用于未来扩展
    """
    logger.debug("Reloading prompt cache (no-op in current implementation)")
