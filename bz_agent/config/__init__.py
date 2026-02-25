from .init_config import (
    # Reasoning LLM
    REASONING_MODEL,
    REASONING_BASE_URL,
    REASONING_API_KEY,
    # Basic LLM
    BASIC_MODEL,
    BASIC_BASE_URL,
    BASIC_API_KEY,
    # Vision-language LLM
    VL_MODEL,
    VL_BASE_URL,
    VL_API_KEY,
    # Other configurations
    LOCAL_BASIC_MODEL_PATH,
    LOCAL_BASIC_BASE_URL,
    LOCAL_BASIC_API_KEY,
    LOCAL_BASIC_MODEL_NAME,
    LOCAL_BASIC_MODEL_MAXTOKEN,
)


# Team configuration
TEAM_MEMBERS = ["url_to_markdown", "coder", "browser", "reporter", "rag"]

__all__ = [
    # Reasoning LLM
    "REASONING_MODEL",
    "REASONING_BASE_URL",
    "REASONING_API_KEY",
    # Basic LLM
    "BASIC_MODEL",
    "BASIC_BASE_URL",
    "BASIC_API_KEY",
    # Vision-language LLM
    "VL_MODEL",
    "VL_BASE_URL",
    "VL_API_KEY",
    # Other configurations
    "LOCAL_BASIC_MODEL_PATH",
    "LOCAL_BASIC_BASE_URL",
    "LOCAL_BASIC_API_KEY",
    "LOCAL_BASIC_MODEL_NAME",
    "LOCAL_BASIC_MODEL_MAXTOKEN",
    "TEAM_MEMBERS",
]
