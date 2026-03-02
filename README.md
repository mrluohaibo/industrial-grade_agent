# Industrial-Grade Agent Scaffolding

A production-ready multi-agent orchestration framework built with LangGraph and LangChain, featuring specialized agents for complex task execution, RAG capabilities, and MCP (Model Context Protocol) support.

## Features

- **Multi-Agent Orchestration**: LangGraph-based workflow with specialized agents (planner, supervisor, coder, browser, reporter, etc.)
- **Session Management**: MongoDB-backed conversation persistence with automatic context loading
- **RAG Integration**: BM25 search with Elasticsearch, vector search with Milvus, and BGE-M3 embeddings
- **MCP Support**: Full Model Context Protocol client/server implementation for tool integration
- **Prompt Management**: Versioned prompt templates with MongoDB storage and import/export capabilities
- **Native Agent Framework**: Custom BaseAgent, ReActAgent, and MCPAgent implementations
- **Multiple LLM Support**: Flexible LLM configuration (reasoning, basic, vision, local models)

## Architecture

```
industrial-grade_agent_scaffolding/
├── bz_agent/                 # Core agent framework
│   ├── agents/              # Agent implementations
│   ├── config/              # Configuration and LLM mapping
│   ├── graph/               # LangGraph workflow orchestration
│   ├── native_agent/        # Custom agent framework
│   ├── prompts/             # Prompt templates (*.md files)
│   ├── tools/               # LangChain tool implementations
│   ├── mcp/                # MCP server and transport
│   ├── rag/                # RAG modules (BM25, embedding, Milvus)
│   └── storage/            # MongoDB storage for conversations/prompts
├── bz_core/                # Stock info API
├── utils/                  # Utilities (database, logger, config)
├── config/                 # Application configuration
├── config_example/          # Configuration templates
├── scripts/                # Utility scripts
└── tests/                  # Test suite
```

## Quick Start

### Prerequisites

- Python 3.9+
- MongoDB (for conversation/prompt storage)
- Elasticsearch (for BM25 search)
- Milvus (for vector storage)
- MySQL (optional, for stock data)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd industrial-grade_agent_scaffolding

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (required for browser agent)
playwright install
```

### Configuration

1. **Copy configuration templates**:

```bash
cp config_example/llm.env config/llm.env
cp config_example/application.yaml config/application.yaml
```

2. **Configure LLM credentials** (`config/llm.env`):

```env
# Reasoning LLM (for complex tasks)
REASONING_API_KEY=your_api_key
REASONING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
REASONING_MODEL=qwq-plus

# Basic LLM (for standard tasks)
BASIC_API_KEY=your_api_key
BASIC_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BASIC_MODEL=qwen-max-latest

# Vision LLM (for visual tasks)
VL_API_KEY=your_api_key
VL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VL_MODEL=qwen2.5-vl-72b-instruct

# Local LLM (Ollama)
LOCAL_BASIC_API_KEY=ollama
LOCAL_BASIC_BASE_URL=http://127.0.0.1:11434/v1
LOCAL_BASIC_MODEL_NAME=qwen3:4b
```

3. **Configure databases** (`config/application.yaml`):

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

### Initialize Prompts

Import prompt templates to MongoDB:

```bash
# Preview import (dry-run)
python -m scripts.import_prompts --dry-run

# Import all prompts
python -m scripts.import_prompts

# Force re-import (creates new versions)
python -m scripts.import_prompts --force
```

## Usage

### Multi-Agent Workflow

Run the main multi-agent workflow:

```python
from bz_agent.workflow import run_agent_workflow

# Simple query
result = run_agent_workflow("Analyze the latest stock trends")

# With session continuity
result = run_agent_workflow(
    user_input="What was my previous question?",
    session_id="existing-session-id"
)

# Access results
final_message = result["messages"][-1].content
session_id = result["session_id"]  # For next request
```

### Web Content Extraction

Extract web content as markdown:

```python
from bz_agent.workflow import request_url_content_to_markdown

content = request_url_content_to_markdown("https://example.com/article")
print(content)
```

### MCP Agent

Run the MCP agent for tool-based operations:

```bash
# Interactive mode
python -m bz_agent.run_mcp -i

# Single prompt
python -m bz_agent.run_mcp -p "Your prompt here"

# SSE connection
python -m bz_agent.run_mcp -c sse --server-url http://127.0.0.1:8000/sse
```

### Available Agents

| Agent | Description |
|--------|-------------|
| `planner` | Generates execution plans using deep reasoning |
| `supervisor` | Routes tasks to appropriate worker agents |
| `coder` | Executes Python/Bash code and calculations |
| `browser` | Interacts with web pages for complex operations |
| `url_to_markdown` | Extracts content from web pages |
| `rag` | Retrieves knowledge from vector database |
| `reporter` | Generates professional reports |

## Workflow Execution Flow

```
User Input
    ↓
[Planner] → Generates execution plan (JSON)
    ↓
[Supervisor] → Routes tasks to workers
    ↓
[Worker Agents] → Execute tasks:
    • Coder (code execution)
    • Browser (web interaction)
    • URL to Markdown (content extraction)
    • RAG (knowledge retrieval)
    ↓
[Reporter] → Final report generation
    ↓
User Output
```

## Prompt Management

Manage prompt templates via CLI:

```bash
# Preview import
python -m scripts.import_prompts --dry-run

# Import specific prompts
python -m scripts.import_prompts --name planner supervisor

# Show differences
python -m scripts.import_prompts --diff planner

# Export prompts from MongoDB
python -m scripts.import_prompts --export

# Backup before import
python -m scripts.import_prompts --backup
```

### Prompt Source Configuration

Configure prompt loading strategy in `config/application.yaml`:

| Source | Description |
|---------|-------------|
| `file` | Load only from markdown files |
| `mongo` | Load only from MongoDB |
| `mongo_fallback` | Try MongoDB first, fallback to files (recommended) |

## RAG Configuration

### Setup Milvus (Vector Search)

1. Download BGE-M3 model:
```bash
# ModelScope or HuggingFace
pip install FlagEmbedding
# Or download from https://modelscope.cn/models/AI-ModelScope/bge-m3
```

2. Configure in `config/application.yaml`:
```yaml
milvus:
  ip: 'your_milvus_host'
  port: 19530
  bge_m3_model_path: '/path/to/bge_m3_model'
```

### Setup Elasticsearch (BM25 Search)

1. Install IK tokenizer for Chinese:
```bash
# Download and install IK analyzer
# https://github.com/medcl/elasticsearch-analysis-ik
```

2. Configure in `config/application.yaml`:
```yaml
es:
  host: 'http://your_es_host:9200'
  u_name: 'your_username'
  u_pwd: 'your_password'
```

## Development

### Code Quality

```bash
# Run linter
ruff check

# Auto-fix issues
ruff check --fix

# Format code
ruff format

# Run pre-commit hooks
pre-commit run --all-files
```

### Adding New Agents

1. Create agent in `bz_agent/agents/agents.py`
2. Add to `TEAM_MEMBERS` in `bz_agent/config/__init__.py`
3. Configure LLM mapping in `bz_agent/config/agents_map.py`
4. Create node function in `bz_agent/graph/nodes.py`
5. Add to `build_graph()` in `bz_agent/graph/builder.py`

### Adding Tools

**LangChain Tools:**
```python
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    """Tool description."""
    # Implementation
    return result
```

**Native Tools:**
```python
from bz_agent.native_agent.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Tool description"

    def execute(self, input_data: str) -> str:
        # Implementation
        return result
```

### Adding MCP Tools

Connect to MCP servers for additional tools:

```python
from bz_agent.tools.mcp import MCPClients

async def use_mcp_tools():
    mcp_clients = MCPClients()

    # Connect to server
    await mcp_clients.connect_stdio("my_server", "python -m my_mcp_server")

    # Call tool
    result = await mcp_clients.call_tool("mcp_my_server_tool_name", {"arg": "value"})

    return result
```

## Configuration Reference

### LLM Types

| Type | Purpose | Example Models |
|------|----------|----------------|
| `reasoning` | Complex reasoning tasks | qwq-plus, deepseek-r1 |
| `basic` | Standard tasks | qwen-max-latest, gpt-4 |
| `vision` | Visual understanding | qwen2.5-vl-72b-instruct |
| `local_basic` | Local execution | qwen3:4b (Ollama) |

### Database Connections

| Service | Purpose | Required |
|---------|---------|----------|
| MongoDB | Conversation/Prompt storage | Yes |
| Milvus | Vector embeddings | RAG only |
| Elasticsearch | BM25 search | RAG only |
| Redis | Caching | Optional |
| TD Engine | Time-series data | Stock data only |
| MySQL | Relational data | Stock data only |

## Troubleshooting

### Common Issues

**MongoDB connection failed:**
- Check `config/application.yaml` MongoDB configuration
- Ensure MongoDB is running: `mongod --version`

**Playwright browsers not installed:**
```bash
playwright install
```

**Prompt not found:**
- Check `PROMPT_SOURCE` in `config/application.yaml`
- Run prompt import: `python -m scripts.import_prompts`

**RAG search returns empty:**
- Verify Milvus and Elasticsearch are running
- Check BGE-M3 model path
- Ensure data has been indexed

## Project Structure Details

### Core Components

- **`bz_agent/graph/`**: LangGraph StateGraph implementation with planner, supervisor, and worker nodes
- **`bz_agent/native_agent/`**: Custom agent framework with BaseAgent, ReActAgent, MCPAgent
- **`bz_agent/storage/`**: MongoDB-backed ConversationStore and PromptStore
- **`bz_agent/rag/`**: BM25 (Elasticsearch), embedding (BGE-M3), vector (Milvus)

### Configuration

- **`config/llm.env`**: LLM API keys and model names
- **`config/application.yaml`**: Database, RAG, prompt, and session settings
- **`bz_agent/config/agents_map.py`**: Agent to LLM type mapping

### Entry Points

- **`bz_agent/workflow.py`**: Main multi-agent workflow
- **`bz_agent/run_mcp.py`**: MCP agent runner
- **`scripts/init_prompts.py`**: Prompt initialization
- **`scripts/import_prompts.py`**: Enhanced prompt management

## License

[Your License Here]

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## Support

For issues and questions, please open an issue on the repository or contact the development team.
