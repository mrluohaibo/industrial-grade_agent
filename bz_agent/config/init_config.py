import os
from dotenv import load_dotenv

from bz_core.Constant import root_path

# Load environment variables
load_dotenv(dotenv_path = os.path.join(root_path, "config/llm.env"))





# Reasoning LLM configuration (for complex reasoning tasks)
REASONING_MODEL = os.getenv("REASONING_MODEL", "o1-mini")
REASONING_BASE_URL = os.getenv("REASONING_BASE_URL")
REASONING_API_KEY = os.getenv("REASONING_API_KEY")

# Non-reasoning LLM configuration (for straightforward tasks)
BASIC_MODEL = os.getenv("BASIC_MODEL", "gpt-4o")
BASIC_BASE_URL = os.getenv("BASIC_BASE_URL")
BASIC_API_KEY = os.getenv("BASIC_API_KEY")

# Vision-language LLM configuration (for tasks requiring visual understanding)
VL_MODEL = os.getenv("VL_MODEL", "gpt-4o")
VL_BASE_URL = os.getenv("VL_BASE_URL")
VL_API_KEY = os.getenv("VL_API_KEY")


LOCAL_BASIC_API_KEY = os.getenv("LOCAL_BASIC_API_KEY", "ollama")
LOCAL_BASIC_BASE_URL = os.getenv("LOCAL_BASIC_BASE_URL")
LOCAL_BASIC_MODEL_NAME = os.getenv("LOCAL_BASIC_MODEL_NAME")
LOCAL_BASIC_MODEL_MAXTOKEN = os.getenv("LOCAL_BASIC_MODEL_MAXTOKEN")

LOCAL_BASIC_MODEL_PATH = os.getenv("LOCAL_BASIC_MODEL_PATH", "")


# Chrome Instance configuration
CHROME_INSTANCE_PATH = os.getenv("CHROME_INSTANCE_PATH")


