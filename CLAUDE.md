# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is an industrial-grade multi-agent system built with LangGraph and LangChain. The project implements a sophisticated agent orchestration framework with specialized agents for different tasks.

### Core Components

**Multi-Agent System (`bz_agent/`):**
- `bz_agent/graph/` - LangGraph-based workflow orchestration using StateGraph
- `bz_agent/agents/` - Individual agent implementations (coder, browser, url_to_markdown, reporter)
- `bz_agent/native_agent/` - Native agent framework with BaseAgent, ReActAgent, MCPAgent classes
- `bz_agent/prompts/` - Prompt templates in Markdown with custom template syntax (`<<VAR>>`)
- `bz_agent/tools/` - Tool implementations (python_repl, browser, bash, file_management, mcp, rag_tool)
- `bz_agent/storage/` - MongoDB-based storage for conversations and prompts
- `bz_agent/rag/` - RAG module with BM25 search, embedding, Milvus, document processing, and reranking
- `bz_agent/mcp/` - Model Context Protocol server and HTTP transport implementations
- `bz_agent/api/` - FastAPI-based RAG document processing API

**Agent Types and LLM Configuration:**
- Agent-LLM mapping in `bz_agent/config/agents_map.py`
- LLM types: `reasoning`, `basic`, `vision`, `local_basic`, `rag`
- Environment configuration from `config/llm.env`

### Workflow Graph

The main workflow (defined in `bz_agent/graph/builder.py`) follows this flow:
1. `planner` - Generates execution plan (uses reasoning LLM in deep_thinking_mode)
2. `supervisor` - Routes tasks to worker agents
3. Worker agents: `coder`, `browser`, `url_to_markdown`, `rag`, `reporter`
4. Each worker returns results wrapped in `<response>` tags

State is managed via `bz_agent/graph/types.py` which extends `MessagesState` with `TEAM_MEMBERS`, `next`, `full_plan`, `deep_thinking_mode`, `search_before_planning`, `session_id`.

**Session Context Management:**
- Workflows use `context_middleware` for pre/post-processing
- Pre-process: Loads conversation history from MongoDB if `session_id` provided
- Post-process: Saves messages to MongoDB
- New sessions are auto-generated if no `session_id` provided

### RAG Document Processing System

The RAG module (`bz_agent/rag/`) provides enterprise-grade document processing:

**DocumentProcessor (`document_processor.py`):**
- Orchestrates the entire document processing pipeline
- Supports PDF, DOCX, Markdown, and TXT files
- Integrates with Milvus (vector storage) and Elasticsearch (full-text search)

**DocumentSplitter (`document_splitter.py`):**
Multiple splitting strategies for optimal chunk quality:
- `recursive` - Recursive character splitting with Chinese-aware delimiters
- `markdown_header` - Split by Markdown headers
- `semantic` - Semantic-based splitting using embedding similarity
- `hybrid` - Combines multiple strategies for best results

**Semantic Refinement (`semantic_refiner.py`):**
- Generates refined summaries for each chunk
- Extracts keywords and entities using LLM
- Improves search result relevance

**BM25 Reranker (`bge_reranker.py`):**
- Uses BGE-Reranker-Large model for result reranking
- Reorders search results based on query-chunk relevance
- Significantly improves search quality

**File Parser (`file_parser.py`):**
- Parses PDF, DOCX, Markdown, and TXT files
- Converts to plain text for processing

**RAG Configuration (from `config/application.yaml`):**
- `milvus`: IP, port, BGE-M3 model path, BGE-Reranker path
- `es`: Elasticsearch host, username, password for BM25 search
- `document`: Storage path, max file size, allowed extensions
- `semantic_refinement`: Enable/disable, LLM model, max keywords/entities

### FastAPI RAG API (`bz_agent/api/`)

**Main App (`api/main.py`):**
- FastAPI application with CORS middleware
- Lifespan management for DocumentProcessor initialization
- Global exception handling and request logging
- Health check endpoint at `/health`

**Document Routes (`document_routes.py`):**
- `POST /api/v1/documents/upload` - Upload and process documents
- `DELETE /api/v1/documents/{document_id}` - Delete document
- `GET /api/v1/documents/{document_id}` - Get document info
- `GET /api/v1/documents/{document_id}/chunks` - Get document chunks
- `GET /api/v1/documents/` - List documents (paginated)
- `POST /api/v1/documents/batch` - Batch upload multiple documents

**RAG Routes (`rag_routes.py`):**
- `GET/POST /api/v1/rag/search` - Semantic search with optional reranking
- `GET /api/v1/rag/health` - RAG service health check (Milvus, ES, embedding)

### Native Agent Framework

`bz_agent/native_agent/` provides a custom agent implementation:
- `BaseAgent` - Abstract base with state management, memory, and execution loop
- `ReActAgent` - Think-act pattern implementation
- `MCPAgent` - Model Context Protocol client for connecting to MCP servers
- `schema.py` - Defines Message, Memory, AgentState, ToolCall, Role types

### MCP (Model Context Protocol)

**MCP Client (`bz_agent/native_agent/mcp.py`):**
- `MCPAgent` connects to MCP servers via SSE or stdio transport
- Automatically discovers and refreshes tools from connected servers
- Supports multiple concurrent MCP server connections
- Tools are prefixed with `mcp_{server_id}_{tool_name}`

**MCP Tools (`bz_agent/tools/mcp.py`):**
- `MCPClients` - Collection class for managing multiple MCP server connections
- `MCPClientTool` - Proxy tool that calls remote MCP server tools
- Supports SSE and stdio transports

**MCP Server (`bz_agent/mcp/server.py`):**
- FastMCP-based server using FastMCP framework
- Registers tools with automatic parameter validation and docstring generation
- Standard tools: bash, browser, editor, terminate

**Running MCP Agent:**
```bash
# Run MCP client with stdio connection
python -m bz_agent.run_mcp --connection stdio

# Run MCP client with SSE connection
python -m bz_agent.run_mcp --connection sse --server-url http://127.0.0.1:8000/sse

# Interactive mode
python -m bz_agent.run_mcp -i

# Single prompt
python -m bz_agent.run_mcp -p "your prompt here"
```

### Storage Layer (`bz_agent/storage/`)

MongoDB-backed storage for conversation and prompt management:

**ConversationStore (`conversation_store.py`):**
- Stores conversation history with session_id
- Automatic message trimming (max 1000 messages per session)
- Methods: `create_session()`, `add_message()`, `get_messages()`, `close_session()`, `delete_session()`
- Indexes on: session_id (unique), user_id, updated_at

**PromptStore (`prompt_store.py`):**
- Versioned prompt templates in MongoDB
- Methods: `save_prompt()`, `get_prompt()`, `update_prompt()`, `activate_prompt()`
- Import from files: `import_from_file()`, `import_all_from_directory()`
- Indexes on: prompt_name, active, updated_at

**ContextMiddleware (`context_middleware.py`):**
- Integrates storage with workflow execution
- `pre_process(state)` - Loads conversation history if session_id exists
- `post_process(state)` - Saves messages to MongoDB after execution

### Database Layer

`utils/db_tool_init.py` initializes global database clients:
- MySQL via `TransactionalMySQLClient` (with autocommit_default=False for transactional safety)
- MongoDB via `MongoManager`
- Redis via `RedisClient`
- TD Engine via `TDEngineClient`

Configuration loaded from YAML in `config/application.yaml` via `utils/config_init.py`.

## Common Commands

**Code Quality:**
```bash
# Run ruff linter
ruff check

# Auto-fix ruff issues
ruff check --fix

# Format code with ruff
ruff format

# Run pre-commit hooks
pre-commit run --all-files

# Install pre-commit hooks
pre-commit install
```

**Running the Workflow:**
```bash
# Run main agent workflow
python main.py

# Run agent workflow programmatically (from Python)
from bz_agent.workflow import run_agent_workflow
result = run_agent_workflow("your query here", session_id="optional-session-id")

# Run url to markdown conversion
python -m bz_agent.workflow

# Extract webpage content to markdown
from bz_agent.workflow import request_url_content_to_markdown
content = request_url_content_to_markdown("https://example.com")
```

**Running FastAPI RAG API:**
```bash
# Start API server (using uvicorn directly)
python -m api.main

# Or with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# API docs available at http://localhost:8000/docs
```

**MCP Agent:**
```bash
# Run MCP client with stdio connection (default)
python -m bz_agent.run_mcp

# Run MCP client with SSE connection
python -m bz_agent.run_mcp --connection sse --server-url http://127.0.0.1:8000/sse

# Interactive mode
python -m bz_agent.run_mcp -i

# Single prompt
python -m bz_agent.run_mcp -p "your prompt here"
```

**Running Tests:**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/api/test_document_routes.py

# Run with verbose output
pytest -v

# Run specific test
pytest tests/api/test_document_routes.py::TestDocumentRoutes::test_upload_markdown

# Run with coverage
pytest --cov=bz_agent --cov-report=html
```

**Prompt Management:**
```bash
# Import all prompt files to MongoDB
python -m scripts.init_prompts

# Import prompts programmatically
from bz_agent.prompts.template import import_prompts_to_mongo
count = import_prompts_to_mongo()

# Reload prompt cache (no-op in current implementation)
from bz_agent.prompts.template import reload_prompt_cache
reload_prompt_cache()
```

**Playwright Setup:**
```bash
# Install Playwright browsers (required for browser agent tools)
playwright install
```

**Dependencies:**
```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

## Development Notes

**Prompt Templates:**
- Located in `bz_agent/prompts/*.md`
- Use `<<VAR>>` syntax for variables (converted to `{VAR}` by `template.py`)
- Current time is automatically injected via `CURRENT_TIME` variable
- **Prompt Source Configuration** (via `PROMPT_SOURCE` env var or `config/application.yaml`):
  - `"file"` - Load only from markdown files (default)
  - `"mongo"` - Load only from MongoDB
  - `"mongo_fallback"` - Try MongoDB first, fallback to files (recommended for production)

**Tool Creation:**
- LangChain tools: Subclass `BaseTool` in `bz_agent/tools/`, use `@tool` decorator
- Native tools: Subclass `BaseTool` from `bz_agent/native_agent/tools/base.py`
- Apply `@log_io` decorator from `tools/decorators.py` for logging

**Adding New Agents:**
1. Create agent in `bz_agent/agents/agents.py` using `create_react_agent`
2. Add to `TEAM_MEMBERS` in `bz_agent/config/__init__.py`
3. Configure LLM mapping in `bz_agent/config/agents_map.py`
4. Create node function in `bz_agent/graph/nodes.py`
5. Add node and edges to `build_graph()` in `bz_agent/graph/builder.py`

**Adding MCP Tools:**
1. Connect to MCP server using `MCPAgent` or `MCPClients`
2. Tools are auto-discovered and prefixed with `mcp_{server_id}_{tool_name}`
3. Use `await mcp_clients.call_tool(tool_name, kwargs)` to execute

**Database Transactions:**
- MySQL client uses `autocommit_default=False`
- Use explicit transaction handling with `commit()`/`rollback()`

**Configuration:**
- Application config: `config/application.yaml` (databases, proxy, prompt, session, milvus, es, document, api, semantic_refinement)
- LLM config: `config/llm.env` (API keys, model names, base URLs)
- Access via `utils.config_init.application_conf.get_properties("path.to.key")`

**Logging:**
- Custom logger from `utils/logger_config.py`
- Logs written to `logs/` directory with daily rotation
- Console output uses colorlog with color-coded levels

**Testing:**
- Tests are located in `tests/` directory
- Use `pytest` for running tests
- API tests use `fastapi.testclient.TestClient`
- Test fixtures defined in individual test files
