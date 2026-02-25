# 开发计划 (TODO)

## 需求概述

基于 `docs/need_module.md` 的三个核心需求：

1. **会话上下文管理**: State 中添加会话ID (session_id)，根据会话ID获取对应的上下文消息列表，MongoDB 持久化
2. **RAG Agent 节点**: 封装 `bz_agent/rag` 中的 RAG 实现为 RAG agent 节点，集成到多 agent 系统中
3. **Prompt 配置管理**: MongoDB 保存 prompt 提示词，支持网页动态配置和实时生效

---

## 当前项目状态

### 已完成:
- [x] State 中添加 `session_id: str` 字段 (`bz_agent/graph/types.py`)
- [x] RAG 模块基础代码 (`bz_agent/rag/`):
  - `bm25_es_search.py` - ES BM25 搜索器
  - `embedding_data_handler.py` - 向量数据处理
  - `multi_call_rag_api.py` - RAG API (BM25 + 向量融合)
  - `split_data_handler.py` - 数据分割
  - `save_embedding_to_milvus.py` - Milvus 存储

### 待完成:
- [ ] 会话存储模块（MongoDB）
- [ ] 会话上下文中间件
- [ ] RAG Agent 封装和集成
- [ ] Prompt 配置管理系统
- [ ] Workflow 集成 session_id

---

## 开发步骤与细节

### 阶段一: 会话上下文管理基础设施

#### 1.1 创建会话存储模块 (`bz_agent/storage/`)

**新建文件**: `bz_agent/storage/__init__.py`
```python
from .conversation_store import ConversationStore, conversation_store

__all__ = ["ConversationStore", "conversation_store"]
```

**新建文件**: `bz_agent/storage/conversation_store.py`

**数据结构设计**:
```
MongoDB Collection: `conversations`

Document Schema:
{
    "_id": ObjectId,
    "session_id": str,           # 会话唯一标识
    "messages": List[Dict],       # 消息历史列表 (LangChain Message格式)
    "created_at": datetime,       # 会话创建时间
    "updated_at": datetime,       # 最后更新时间
    "metadata": Dict,             # 可选: 额外元数据
    "user_id": Optional[str],     # 可选: 用户标识
    "status": str                 # 会话状态: "active", "closed", "archived"
}
```

**核心方法**:
```python
class ConversationStore:
    """会话上下文 MongoDB 存储管理"""

    COLLECTION_NAME = "conversations"
    MAX_MESSAGES_PER_SESSION = 1000

    def generate_session_id(self) -> str
    def create_session(self, session_id: Optional[str] = None, ...) -> str
    def add_message(self, session_id: str, message: BaseMessage, ...) -> bool
    def add_messages(self, session_id: str, messages: List[BaseMessage], ...) -> int
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[BaseMessage]
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]
    def update_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool
    def close_session(self, session_id: str) -> bool
    def delete_session(self, session_id: str) -> int
    def list_sessions(self, user_id: Optional[str] = None, ...) -> List[Dict[str, Any]]

    # 全局单例
conversation_store = ConversationStore()
```

**说明**:
- 使用 `langchain_core.messages.message_to_dict` 和 `messages_from_dict` 进行序列化
- 支持消息数量限制防止无限增长
- 支持用户标识、会话状态管理

---

#### 1.2 创建会话上下文中间件 (`bz_agent/storage/context_middleware.py`)

**新建文件**: `bz_agent/storage/context_middleware.py`

```python
class ContextMiddleware:
    """会话上下文中间件"""

    def pre_process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Workflow 执行前的预处理:
        - 如果没有 session_id，创建新会话
        - 如果有 session_id，从 MongoDB 加载历史消息并合并
        """
        pass

    def post_process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Workflow 执行后的后处理:
        - 保存所有新消息到 MongoDB
        """
        pass

def with_context_middleware(workflow_func: Callable) -> Callable:
    """装饰器：为 workflow 函数添加会话上下文处理"""
    pass
```

---

#### 1.3 更新 Workflow 入口 (`bz_agent/workflow.py`)

**修改内容**:
```python
def run_agent_workflow(
    user_input: str,
    session_id: Optional[str] = None
):
    """
    新增 session_id 参数，支持:
    1. 不提供 session_id -> 创建新会话
    2. 提供 session_id -> 加载历史消息，继续对话
    """
    # 1. 构建初始 state
    # 2. 使用 context_middleware 预处理（加载历史）
    # 3. 执行 workflow
    # 4. 使用 context_middleware 后处理（保存消息）
    pass
```

---

### 阶段二: RAG Agent 节点封装

#### 2.1 创建 RAG 工具 (`bz_agent/tools/rag_tool.py`)

**新建文件**: `bz_agent/tools/rag_tool.py`

**设计思路**:
- 封装 `MultiCallRagApi` 为 LangChain 工具
- 提供同步和异步两个版本
- 使用单例模式缓存 RAG 实例
- 支持自定义查询参数（ES索引名、向量表名、top_k等）

```python
# 全局 RAG 实例缓存
def get_rag_instance() -> MultiCallRagApi

# 同步工具（接受 JSON 字符串参数）
@tool
def rag_knowledge_retrieval(input_params: str) -> str:
    """从知识库检索相关信息"""
    pass

# 异步工具
@tool
async def rag_knowledge_retrieval_async(
    query: str,
    top_k: int = 10,
    es_index_name: str = "cmrc2018_train",
    vec_table_name: str = "q_content"
) -> str:
    """从知识库检索相关信息（异步版本）"""
    pass
```

**说明**:
- 工具名称: `rag_knowledge_retrieval` / `rag_knowledge_retrieval_async`
- 输入参数: 查询文本 + 可选的检索参数
- 输出: 检索到的知识内容或 LLM 生成的回答

---

#### 2.2 创建 RAG Prompt 模板 (`bz_agent/prompts/rag.md`)

**新建文件**: `bz_agent/prompts/rag.md`

```markdown
---
CURRENT_TIME: <<CURRENT_TIME>>
---

You are a knowledge retrieval specialist responsible for finding relevant information from the knowledge base using RAG (Retrieval-Augmented Generation).

# Role

Your expertise lies in:
- Understanding user queries and identifying key information needs
- Formulating effective search queries for knowledge retrieval
- Synthesizing retrieved information into clear, accurate responses
- Citing sources appropriately when providing information

# Steps

1. **Analyze the Query**: Carefully read the user's question to understand what information they need.
2. **Formulate Search**: Use the `rag_knowledge_retrieval` tool to search the knowledge base.
3. **Synthesize Results**: Review the retrieved information and provide a clear, accurate answer.
4. **Cite Sources**: When providing information, mention that it comes from the knowledge base.

# Notes

- Always use the RAG tool to retrieve information before answering knowledge-related questions.
- If the knowledge base doesn't contain relevant information, clearly state this.
- Focus on providing factual information from the retrieved documents.
- Always use the same language as the initial question.
```

---

#### 2.3 创建 RAG Agent (`bz_agent/agents/agents.py` - 新增)

```python
rag_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["rag"]),
    tools=[rag_knowledge_retrieval_async],
    prompt=lambda state: apply_prompt_template("rag", state),
)
```

---

#### 2.4 更新配置文件

**修改**: `bz_agent/config/agents_map.py`
```python
AGENT_LLM_MAP: dict[str, LLMType] = {
    # ...existing...
    "rag": "basic",  # 新增
}
```

**修改**: `bz_agent/config/__init__.py`
```python
TEAM_MEMBERS = ["url_to_markdown", "coder", "browser", "reporter", "rag"]  # 新增 "rag"
```

---

#### 2.5 添加 RAG 节点 (`bz_agent/graph/nodes.py` - 新增)

```python
def rag_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the RAG agent that performs knowledge retrieval tasks."""
    result = rag_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMAT.format(
                        "rag", result["messages"][-1].content
                    ),
                    name="rag",
                )
            ]
        },
        goto="supervisor",
    )
```

---

#### 2.6 更新 Graph 构建器 (`bz_agent/graph/builder.py`)

```python
from .nodes import (
    # ...existing...
    rag_node,  # 新增
)

def build_graph():
    builder = StateGraph(State)
    builder.add_edge(START, "planner")
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("url_to_markdown", url_to_markdown_node)
    builder.add_node("coder", code_node)
    builder.add_node("browser", browser_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("rag", rag_node)  # 新增
    return builder.compile()
```

---

### 阶段三: Prompt 配置管理系统

#### 3.1 创建 Prompt 存储模块 (`bz_agent/storage/prompt_store.py`)

**新建文件**: `bz_agent/storage/prompt_store.py`

**数据结构设计**:
```
MongoDB Collection: `prompts`

Document Schema:
{
    "_id": ObjectId,
    "prompt_name": "planner",        # prompt 名称（与文件名对应）
    "template": "...",              # prompt 模板内容（使用 <<VAR>> 语法）
    "description": "Planner agent prompt",  # 描述
    "active": true,                 # 是否激活（使用数据库版本时）
    "version": 1,                   # 版本号
    "created_at": datetime,         # 创建时间
    "updated_at": datetime,         # 更新时间
    "created_by": "user",           # 创建者（可选）
    "tags": ["agent", "planner"]    # 标签（可选）
}
```

**核心方法**:
```python
class PromptStore:
    """Prompt 模板 MongoDB 存储管理"""

    COLLECTION_NAME = "prompts"

    def __init__(self):
        self.collection = mongo_client.get_collection(self.COLLECTION_NAME)

    def get_prompt(self, prompt_name: str, version: Optional[int] = None) -> Optional[str]:
        """
        获取 prompt 模板

        Args:
            prompt_name: prompt 名称
            version: 可选版本号，不指定则获取最新激活版本

        Returns:
            prompt 模板内容，如果不存在则返回 None
        """
        pass

    def save_prompt(
        self,
        prompt_name: str,
        template: str,
        description: str = "",
        created_by: str = "system"
    ) -> str:
        """
        保存或更新 prompt 模板

        Args:
            prompt_name: prompt 名称
            template: 模板内容
            description: 描述
            created_by: 创建者

        Returns:
            文档 ID
        """
        pass

    def update_prompt(
        self,
        prompt_name: str,
        template: str,
        description: Optional[str] = None
    ) -> bool:
        """更新现有的 prompt 模板"""
        pass

    def list_prompts(
        self,
        active_only: bool = False,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """列出所有 prompt 模板"""
        pass

    def delete_prompt(self, prompt_name: str) -> int:
        """删除 prompt 模板"""
        pass

    def activate_prompt(self, prompt_name: str) -> bool:
        """激活指定 prompt（设置 active=true，其他同名的设置为 false）"""
        pass

    def import_from_file(self, prompt_name: str) -> Optional[str]:
        """
        从文件导入 prompt 到数据库

        用于初始化或迁移现有 prompt 文件到数据库
        """
        pass

    def import_all_from_directory(self, prompts_dir: str) -> int:
        """批量导入目录下所有 .md prompt 文件"""
        pass

# 全局单例
prompt_store = PromptStore()
```

---

#### 3.2 修改 Prompt 模板加载逻辑 (`bz_agent/prompts/template.py`)

**修改文件**: `bz_agent/prompts/template.py`

**设计思路**:
- 添加一个配置项控制 prompt 来源（文件 vs MongoDB）
- 支持优先从 MongoDB 加载，不存在时回退到文件
- 保持现有 API 兼容性

```python
# 新增配置
PROMPT_SOURCE = os.getenv("PROMPT_SOURCE", "file")  # "file" | "mongo" | "mongo_fallback"

# 新增函数
def get_prompt_from_mongo(prompt_name: str) -> Optional[str]:
    """从 MongoDB 获取 prompt 模板"""
    from bz_agent.storage import prompt_store
    return prompt_store.get_prompt(prompt_name)

# 修改原有函数
def get_prompt_template(prompt_name: str) -> str:
    """
    获取 prompt 模板，支持多种来源

    优先级:
    1. PROMPT_SOURCE="mongo" -> 仅从 MongoDB 加载
    2. PROMPT_SOURCE="mongo_fallback" -> 先 MongoDB，不存在则文件
    3. PROMPT_SOURCE="file" -> 仅从文件加载（默认）
    """
    if PROMPT_SOURCE == "mongo":
        template = get_prompt_from_mongo(prompt_name)
        if not template:
            raise ValueError(f"Prompt '{prompt_name}' not found in MongoDB")
    elif PROMPT_SOURCE == "mongo_fallback":
        template = get_prompt_from_mongo(prompt_name)
        if not template:
            logger.warning(f"Prompt '{prompt_name}' not in MongoDB, falling back to file")
            template = _get_prompt_from_file(prompt_name)
    else:  # "file"
        template = _get_prompt_from_file(prompt_name)

    # 转换模板语法: { -> {{, } -> }}, <<VAR>> -> {VAR}
    template = template.replace("{", "{{").replace("}", "}}")
    template = re.sub(r"<<([^>>]+)>>", r"{\1}", template)
    return template

def _get_prompt_from_file(prompt_name: str) -> str:
    """原始的文件加载逻辑（提取为独立函数）"""
    return open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()

# 新增：批量导入工具函数
def import_prompts_to_mongo(prompts_dir: Optional[str] = None) -> int:
    """
    将目录下所有 .md prompt 文件导入到 MongoDB

    用于初始化 prompt 数据库或迁移

    Args:
        prompts_dir: prompt 文件目录，默认为当前 prompts 目录

    Returns:
        导入的 prompt 数量
    """
    pass
```

---

#### 3.3 添加配置项 (`config/application.yaml`)

**新增内容**:
```yaml
# 现有配置...

# Prompt 配置
prompt:
  source: "mongo_fallback"  # "file" | "mongo" | "mongo_fallback"
  collection_name: "prompts"
  directory: "bz_agent/prompts"  # prompt 文件目录
```

---

#### 3.4 创建 Prompt 管理 API (`bz_agent/api/prompt_api.py`)

**新建文件**: `bz_agent/api/prompt_api.py`

**说明**: 为未来的 Web 界面提供 API 支持

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/prompts", tags=["prompts"])

class PromptCreate(BaseModel):
    prompt_name: str
    template: str
    description: str = ""

class PromptUpdate(BaseModel):
    template: str
    description: Optional[str] = None

@router.get("/{prompt_name}")
async def get_prompt(prompt_name: str, version: Optional[int] = None):
    """获取 prompt 模板"""
    pass

@router.post("/")
async def create_prompt(data: PromptCreate):
    """创建 prompt 模板"""
    pass

@router.put("/{prompt_name}")
async def update_prompt(prompt_name: str, data: PromptUpdate):
    """更新 prompt 模板"""
    pass

@router.delete("/{prompt_name}")
async def delete_prompt(prompt_name: str):
    """删除 prompt 模板"""
    pass

@router.get("/")
async def list_prompts(active_only: bool = False, tag: Optional[str] = None):
    """列出所有 prompt 模板"""
    pass

@router.post("/import")
async def import_prompts():
    """从文件批量导入 prompt 到数据库"""
    pass

@router.post("/{prompt_name}/activate")
async def activate_prompt(prompt_name: str):
    """激活指定 prompt"""
    pass
```

---

#### 3.5 创建初始化脚本 (`scripts/init_prompts.py`)

**新建文件**: `scripts/init_prompts.py`

```python
"""
初始化脚本：将现有 prompt 文件导入到 MongoDB

使用方法:
    python -m scripts.init_prompts
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bz_agent.prompts.template import import_prompts_to_mongo
from utils.logger_config import logger

def main():
    logger.info("Starting prompt import to MongoDB...")

    count = import_prompts_to_mongo()
    logger.info(f"Imported {count} prompts to MongoDB")

    # 验证导入
    from bz_agent.storage import prompt_store
    all_prompts = prompt_store.list_prompts()
    logger.info(f"Total prompts in MongoDB: {len(all_prompts)}")

if __name__ == "__main__":
    main()
```

---

### 阶段四: 配置文件更新

#### 4.1 添加 RAG 配置 (`config/rag_config.yaml`)

**新建文件**: `config/rag_config.yaml`

```yaml
# RAG 配置

elasticsearch:
  host: "http://192.168.99.108:9200"
  basic_auth:
    username: "buz_ac"
    password: "123456"
  default_index: "cmrc2018_train"

milvus:
  url: "http://192.168.99.108:19530"
  default_table: "q_content"

embedding:
  model_path: "H:/large_data/modelscope_model/bge_m3"
  use_fp16: true
  dimension: 1024

retrieval:
  default_top_k: 10
  rrf_m: 60  # RRF 算法的 m 参数

llm:
  # RAG 使用的 LLM 配置（可选，默认使用 reasoning LLM）
  model: "deepseek-reasoner"
```

---

#### 4.2 更新应用配置 (`config/application.yaml`)

**新增内容**:
```yaml
# 现有配置...

# 会话存储配置
session:
  max_messages_per_session: 1000
  default_status: "active"
  collection_name: "conversations"

# Prompt 配置
prompt:
  source: "mongo_fallback"  # "file" | "mongo" | "mongo_fallback"
  collection_name: "prompts"
  directory: "bz_agent/prompts"
```

---

### 阶段五: 测试与验证

#### 5.1 创建会话上下文测试 (`tests/test_session_context.py`)

**新建文件**: `tests/test_session_context.py`

```python
def test_create_session()
def test_add_message()
def test_get_messages_with_limit()
def test_session_persistence()
def test_workflow_with_session()
```

---

#### 5.2 创建 RAG 测试 (`tests/test_rag_agent.py`)

**新建文件**: `tests/test_rag_agent.py`

```python
@pytest.mark.asyncio
async def test_rag_retrieval()

@pytest.mark.asyncio
async def test_rag_no_match()

@pytest.mark.asyncio
async def test_rag_agent_node()
```

---

#### 5.3 创建 Prompt 管理测试 (`tests/test_prompt_store.py`)

**新建文件**: `tests/test_prompt_store.py`

```python
def test_save_prompt()
def test_get_prompt()
def test_update_prompt()
def test_list_prompts()
def test_activate_prompt()
def test_import_from_file()
def test_template_loading_from_mongo()
def test_template_loading_with_fallback()
```

---

## 开发顺序建议

### 第一阶段: 会话上下文管理 (优先级 P0)
1. 创建会话存储模块 (`bz_agent/storage/conversation_store.py`)
2. 创建会话上下文中间件 (`bz_agent/storage/context_middleware.py`)
3. 更新 Workflow 入口 (`bz_agent/workflow.py`)
4. 编写测试验证会话功能

### 第二阶段: RAG Agent 集成 (优先级 P0)
1. 创建 RAG 工具 (`bz_agent/tools/rag_tool.py`)
2. 创建 RAG Prompt 模板 (`bz_agent/prompts/rag.md`)
3. 创建 RAG Agent (`bz_agent/agents/agents.py`)
4. 更新配置文件
5. 添加 RAG 节点到 graph
6. 编写测试验证 RAG 功能

### 第三阶段: Prompt 配置管理 (优先级 P1)
1. 创建 Prompt 存储模块 (`bz_agent/storage/prompt_store.py`)
2. 修改 Prompt 模板加载逻辑 (`bz_agent/prompts/template.py`)
3. 添加配置项 (`config/application.yaml`)
4. 创建 Prompt 管理 API (`bz_agent/api/prompt_api.py`)
5. 创建初始化脚本 (`scripts/init_prompts.py`)
6. 编写测试验证 Prompt 功能

### 第四阶段: Web 界面集成 (优先级 P2，可选)
1. 创建 Web API 服务 (FastAPI)
2. 创建前端界面（Prompt 编辑器）
3. 集成实时预览功能

---

## 文件清单

### 新建文件:
```
# 会话管理
bz_agent/storage/__init__.py
bz_agent/storage/conversation_store.py
bz_agent/storage/context_middleware.py

# RAG 工具
bz_agent/tools/rag_tool.py
bz_agent/prompts/rag.md

# Prompt 管理
bz_agent/storage/prompt_store.py
bz_agent/api/prompt_api.py
scripts/init_prompts.py

# 配置
config/rag_config.yaml

# 测试
tests/test_session_context.py
tests/test_rag_agent.py
tests/test_prompt_store.py
```

### 修改文件:
```
# 会话管理集成
bz_agent/workflow.py

# RAG 集成
bz_agent/agents/agents.py
bz_agent/config/agents_map.py
bz_agent/config/__init__.py
bz_agent/graph/nodes.py
bz_agent/graph/builder.py

# Prompt 管理
bz_agent/prompts/template.py
config/application.yaml
```

---

## 注意事项

1. **MongoDB 索引**: 确保 `conversations` 和 `prompts` 集合在相应字段上有索引
2. **消息格式**: 确保存储的消息格式与 LangChain 兼容
3. **异步处理**: RAG 工具需要异步执行
4. **错误处理**: RAG 组件依赖外部服务，需要错误处理和重试逻辑
5. **配置管理**: RAG 和 Prompt 相关配置应可配置化
6. **日志记录**: 关键操作应有适当的日志记录
7. **回退机制**: Prompt 加载应支持 MongoDB 不可用时回退到文件
8. **版本控制**: Prompt 支持版本管理，便于回滚

---

## 开发检查清单

- [ ] 阶段一: 会话上下文管理完成
  - [ ] ConversationStore 创建完成
  - [ ] ContextMiddleware 创建完成
  - [ ] Workflow 集成 session_id
  - [ ] 会话测试通过

- [ ] 阶段二: RAG Agent 集成完成
  - [ ] RAG 工具创建完成
  - [ ] RAG Agent 创建完成
  - [ ] RAG 节点添加到 graph
  - [ ] RAG 测试通过

- [ ] 阶段三: Prompt 配置管理完成
  - [ ] PromptStore 创建完成
  - [ ] 模板加载逻辑更新
  - [ ] Prompt API 创建完成
  - [ ] 初始化脚本创建完成
  - [ ] Prompt 测试通过

- [ ] 阶段四: 配置和文档完成
  - [ ] 配置文件更新完成
  - [ ] API 文档更新完成
  - [ ] 开发文档更新完成
