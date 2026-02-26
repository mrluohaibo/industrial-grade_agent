# 工业级智能体开发脚手架

基于 LangGraph 和 LangChain 构建的生产级多智能体编排框架，具备专用智能体执行复杂任务、RAG 能力以及 MCP（模型上下文协议）支持。

## 特性

- **多智能体编排**：基于 LangGraph 的工作流，包含专用智能体（规划器、监督器、编码器、浏览器、报告器等）
- **会话管理**：基于 MongoDB 的对话持久化，支持自动上下文加载
- **RAG 集成**：使用 Elasticsearch 的 BM25 搜索、使用 Milvus 的向量搜索以及 BGE-M3 嵌入
- **MCP 支持**：完整的模型上下文协议客户端/服务器实现，用于工具集成
- **提示词管理**：支持版本化的提示词模板，可存储于 MongoDB 并支持导入/导出
- **原生智能体框架**：自定义 BaseAgent、ReActAgent 和 MCPAgent 实现
- **多 LLM 支持**：灵活的 LLM 配置（推理、基础、视觉、本地模型）

## 架构

```
industrial-grade_agent scaffolding/
├── bz_agent/                 # 核心智能体框架
│   ├── agents/              # 智能体实现
│   ├── config/              # 配置和 LLM 映射
│   ├── graph/               # LangGraph 工作流编排
│   ├── native_agent/        # 自定义智能体框架
│   ├── prompts/             # 提示词模板 (*.md 文件)
│   ├── tools/               # LangChain 工具实现
│   ├── mcp/                # MCP 服务器和传输层
│   ├── rag/                # RAG 模块 (BM25、嵌入、Milvus)
│   └── storage/            # MongoDB 存储（对话/提示词）
├── bz_core/                # 股票信息 API
├── utils/                  # 工具类（数据库、日志、配置）
├── config/                 # 应用配置
├── config_example/          # 配置模板
├── scripts/                # 实用脚本
└── tests/                  # 测试套件
```

## 快速开始

### 前置要求

- Python 3.9+
- MongoDB（用于对话/提示词存储）
- Elasticsearch（用于 BM25 搜索）
- Milvus（用于向量存储）
- MySQL（可选，用于股票数据）

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd industrial-grade_agent_scaffolding

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装 Playwright 浏览器（浏览器智能体必需）
playwright install
```

### 配置

1. **复制配置模板**：

```bash
cp config_example/llm.env config/llm.env
cp config_example/application.yaml config/application.yaml
```

2. **配置 LLM 凭证**（`config/llm.env`）：

```env
# 推理 LLM（用于复杂任务）
REASONING_API_KEY=your_api_key
REASONING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
REASONING_MODEL=qwq-plus

# 基础 LLM（用于标准任务）
BASIC_API_KEY=your_api_key
BASIC_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BASIC_MODEL=qwen-max-latest

# 视觉 LLM（用于视觉任务）
VL_API_KEY=your_api_key
VL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VL_MODEL=qwen2.5-vl-72b-instruct

# 本地 LLM（Ollama）
LOCAL_BASIC_API_KEY=ollama
LOCAL_BASIC_BASE_URL=http://127.0.0.1:11434/v1
LOCAL_BASIC_MODEL_NAME=qwen3:4b
```

3. **配置数据库**（`config/application.yaml`）：

```yaml
mongo:
  host: 'your_mongo_host'
  port: 27017
  db_name: 'stock_db'

milvus:
  ip: 'your_milvus_host'
  port: 19530
  bge_m3_model_path: '/path/to/bge_m3_model'

es:
  host: 'http://your_es_host:9200'
  u_name: 'your_username'
  u_pwd: 'your_password'

prompt:
  source: 'mongo_fallback'  # file | mongo | mongo_fallback
  directory: 'bz_agent/prompts'
```

### 初始化提示词

将提示词模板导入到 MongoDB：

```bash
# 预览导入（试运行）
python -m scripts.import_prompts --dry-run

# 导入所有提示词
python -m scripts.import_prompts

# 强制重新导入（创建新版本）
python -m scripts.import_prompts --force
```

## 使用

### 多智能体工作流

运行主多智能体工作流：

```python
from bz_agent.workflow import run_agent_workflow

# 简单查询
result = run_agent_workflow("分析最新的股票趋势")

# 支持会话连续性
result = run_agent_workflow(
    user_input="我上一个问题是什么？",
    session_id="existing-session-id"
)

# 访问结果
final_message = result["messages"][-1].content
session_id = result["session_id"]  # 用于下一次请求
```

### 网页内容提取

提取网页内容为 Markdown 格式：

```python
from bz_agent.workflow import request_url_content_to_markdown

content = request_url_content_to_markdown("https://example.com/article")
print(content)
```

### MCP 智能体

运行 MCP 智能体进行基于工具的操作：

```bash
# 交互模式
python -m bz_agent.run_mcp -i

# 单次提示
python -m bz_agent.run_mcp -p "你的提示内容"

# SSE 连接
python -m bz_agent.run_mcp -c sse --server-url http://127.0.0.1:8000/sse
```

### 可用智能体

| 智能体 | 描述 |
|--------|-------------|
| `planner` | 使用深度推理生成执行计划 |
| `supervisor` | 将任务路由到合适的智能体 |
| `coder` | 执行 Python/Bash 代码和计算 |
| `browser` | 与网页交互进行复杂操作 |
| `url_to_markdown` | 从网页提取内容 |
| `rag` | 从向量数据库检索知识 |
| `reporter` | 生成专业报告 |

## 工作流执行流程

```
用户输入
    ↓
[规划器] → 生成执行计划 (JSON)
    ↓
[监督器] → 将任务路由到工作智能体
    ↓
[工作智能体] → 执行任务：
    • 编码器（代码执行）
    • 浏览器（网页交互）
    • URL 转 Markdown（内容提取）
    • RAG（知识检索）
    ↓
[报告器] → 生成最终报告
    ↓
用户输出
```

## 提示词管理

通过 CLI 管理提示词模板：

```bash
# 预览导入
python -m scripts.import_prompts --dry-run

# 导入指定提示词
python -m scripts.import_prompts --name planner supervisor

# 显示差异
python -m scripts.import_prompts --diff planner

# 从 MongoDB 导出提示词
python -m scripts.import_prompts --export

# 导入前备份
python -m scripts.import_prompts --backup
```

### 提示词源配置

在 `config/application.yaml` 中配置提示词加载策略：

| 源 | 描述 |
|---------|-------------|
| `file` | 仅从 markdown 文件加载 |
| `mongo` | 仅从 MongoDB 加载 |
| `mongo_fallback` | 优先 MongoDB，回退到文件（推荐） |

## RAG 配置

### 设置 Milvus（向量搜索）

1. 下载 BGE-M3 模型：
```bash
# ModelScope 或 HuggingFace
pip install FlagEmbedding
# 或从 https://modelscope.cn/models/AI-ModelScope/bge-m3 下载
```

2. 在 `config/application.yaml` 中配置：
```yaml
milvus:
  ip: 'your_milvus_host'
  port: 19530
  bge_m3_model_path: '/path/to/bge_m3_model'
```

### 设置 Elasticsearch（BM25 搜索）

1. 安装中文 IK 分词器：
```bash
# 下载并安装 IK 分析器
# https://github.com/medcl/elasticsearch-analysis-ik
```

2. 在 `config/application.yaml` 中配置：
```yaml
es:
  host: 'http://your_es_host:9200'
  u_name: 'your_username'
  u_pwd: 'your_password'
```

## 开发

### 代码质量

```bash
# 运行代码检查
ruff check

# 自动修复问题
ruff check --fix

# 格式化代码
ruff format

# 运行 pre-commit 钩子
pre-commit run --all-files
```

### 添加新智能体

1. 在 `bz_agent/agents/agents.py` 中创建智能体
2. 在 `bz_agent/config/__init__.py` 中添加到 `TEAM_MEMBERS`
3. 在 `bz_agent/config/agents_map.py` 中配置 LLM 映射
4. 在 `bz_agent/graph/nodes.py` 中创建节点函数
5. 在 `bz_agent/graph/builder.py` 的 `build_graph()` 中添加

### 添加工具

**LangChain 工具：**
```python
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    """工具描述。"""
    # 实现
    return result
```

**原生工具：**
```python
from bz_agent.native_agent.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "工具描述"

    def execute(self, input_data: str) -> str:
        # 实现
        return result
```

### 添加 MCP 工具

连接到 MCP 服务器获取额外工具：

```python
from bz_agent.tools.mcp import MCPClients

async def use_mcp_tools():
    mcp_clients = MCPClients()

    # 连接到服务器
    await mcp_clients.connect_stdio("my_server", "python -m my_mcp_server")

    # 调用工具
    result = await mcp_clients.call_tool("mcp_my_server_tool_name", {"arg": "value"})

    return result
```

## 配置参考

### LLM 类型

| 类型 | 用途 | 示例模型 |
|------|----------|----------------|
| `reasoning` | 复杂推理任务 | qwq-plus, deepseek-r1 |
| `basic` | 标准任务 | qwen-max-latest, gpt-4 |
| `vision` | 视觉理解 | qwen2.5-vl-72b-instruct |
| `local_basic` | 本地执行 | qwen3:4b (Ollama) |

### 数据库连接

| 服务 | 用途 | 必需 |
|---------|---------|----------|
| MongoDB | 对话/提示词存储 | 是 |
| Milvus | 向量嵌入 | 仅 RAG |
| Elasticsearch | BM25 搜索 | 仅 RAG |
| Redis | 缓存 | 可选 |
| TD Engine | 时序数据 | 仅股票数据 |
| MySQL | 关系型数据 | 仅股票数据 |

## 故障排除

### 常见问题

**MongoDB 连接失败：**
- 检查 `config/application.yaml` 中的 MongoDB 配置
- 确保 MongoDB 正在运行：`mongod --version`

**Playwright 浏览器未安装：**
```bash
playwright install
```

**提示词未找到：**
- 检查 `config/application.yaml` 中的 `PROMPT_SOURCE`
- 运行提示词导入：`python -m scripts.import_prompts`

**RAG 搜索返回空：**
- 验证 Milvus 和 Elasticsearch 正在运行
- 检查 BGE-M3 模型路径
- 确保数据已被索引

## 项目结构详情

### 核心组件

- **`bz_agent/graph/`**: LangGraph StateGraph 实现，包含规划器、监督器和工作节点
- **`bz_agent/native_agent/`**: 自定义智能体框架，包含 BaseAgent、ReActAgent、MCPAgent
- **`bz_agent/storage/`**: 基于 MongoDB 的 ConversationStore 和 PromptStore
- **`bz_agent/rag/`**: BM25 (Elasticsearch)、嵌入 (BGE-M3)、向量 (Milvus)

### 配置

- **`config/llm.env`**: LLM API 密钥和模型名称
- **`config/application.yaml`**: 数据库、RAG、提示词和会话设置
- **`bz_agent/config/agents_map.py`**: 智能体到 LLM 类型的映射

### 入口点

- **`bz_agent/workflow.py`**: 主多智能体工作流
- **`bz_agent/run_mcp.py`**: MCP 智能体运行器
- **`scripts/init_prompts.py`**: 提示词初始化
- **`scripts/import_prompts.py`**: 增强版提示词管理

## 许可证

[您的许可证]

## 贡献

欢迎贡献！请阅读我们的贡献指南并提交拉取请求。

## 支持

如有问题和疑问，请在仓库上提交 issue 或联系开发团队。
