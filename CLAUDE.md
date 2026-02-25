# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is an industrial-grade multi-agent system built with LangGraph and LangChain. The project implements a sophisticated agent orchestration framework with specialized agents for different tasks.

### Core Components

**Multi-Agent System (`bz_agent/`):**
- `bz_agent/graph/` - LangGraph-based workflow orchestration using StateGraph
- `bz_agent/agents/` - Individual agent implementations (coder, browser, url_to_markdown, reporter)
- `bz_agent/native_agent/` - Native agent framework with BaseAgent, ReActAgent classes
- `bz_agent/prompts/` - Prompt templates in Markdown with custom template syntax (`<<VAR>>`)
- `bz_agent/tools/` - Tool implementations (python_repl, browser, bash, file_management)

**Agent Types and LLM Configuration:**
- Agent-LLM mapping in `bz_agent/config/agents_map.py`
- LLM types: `reasoning`, `basic`, `vision`, `local_basic`
- Environment configuration from `config/llm.env`

### Workflow Graph

The main workflow (defined in `bz_agent/graph/builder.py`) follows this flow:
1. `planner` - Generates execution plan (uses reasoning LLM in deep_thinking_mode)
2. `supervisor` - Routes tasks to worker agents
3. Worker agents: `coder`, `browser`, `url_to_markdown`, `reporter`
4. Each worker returns results wrapped in `<response>` tags

State is managed via `bz_agent/graph/types.py` which extends `MessagesState` with `TEAM_MEMBERS`, `next`, `full_plan`, `deep_thinking_mode`, `search_before_planning`.

### Native Agent Framework

`bz_agent/native_agent/` provides a custom agent implementation:
- `BaseAgent` - Abstract base with state management, memory, and execution loop
- `ReActAgent` - Think-act pattern implementation
- `schema.py` - Defines Message, Memory, AgentState, ToolCall, Role types

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

# Run url to markdown conversion
python -m bz_agent.workflow
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

**Tool Creation:**
- Tools are LangChain `BaseTool` subclasses in `bz_agent/tools/`
- Use `@tool` decorator from `langchain_core.tools`
- Apply `@log_io` decorator from `tools/decorators.py` for logging

**Adding New Agents:**
1. Create agent in `bz_agent/agents/agents.py` using `create_react_agent`
2. Add to `TEAM_MEMBERS` in `bz_agent/config/__init__.py`
3. Configure LLM mapping in `bz_agent/config/agents_map.py`
4. Create node function in `bz_agent/graph/nodes.py`
5. Add node and edges to `build_graph()` in `bz_agent/graph/builder.py`

**Database Transactions:**
- MySQL client uses `autocommit_default=False`
- Use explicit transaction handling with `commit()`/`rollback()`

**Configuration:**
- Application config: `config/application.yaml` (databases, proxy settings)
- LLM config: `config/llm.env` (API keys, model names, base URLs)
- Access via `utils.config_init.application_conf.get_properties("path.to.key")`

**Logging:**
- Custom logger from `utils/logger_config.py`
- Logs written to `logs/` directory with daily rotation
- Console output uses colorlog with color-coded levels
