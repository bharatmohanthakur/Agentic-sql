# Agentic Text-to-SQL 2.0

A best-in-class agentic text-to-SQL framework combining modern AI agent patterns with enterprise-grade features.

## Features

### Core Architecture
- **ReAct Pattern**: Reason + Act loop for intelligent query generation
- **Reflection**: Self-evaluation and correction of outputs
- **Multi-Agent Collaboration**: Specialized agents for different tasks
- **Planning**: Complex query decomposition

### Memory System
- **Hybrid Storage**: Graph + Vector + SQL persistence (inspired by Cognee & Memori)
- **ECL Pipeline**: Extract, Cognify, Load pattern for knowledge management
- **Multi-level Hierarchy**: Entity, process, and session-level memories
- **Pattern Learning**: Stores successful queries for future reference

### Enterprise Features
- **User-Aware Pipeline**: Identity propagation at every layer
- **Row-Level Security**: Automatic SQL filtering per user permissions
- **Audit Logging**: Track all queries and actions
- **Rate Limiting**: Per-user rate controls
- **Multi-Provider LLM**: OpenAI, Anthropic, Ollama, Azure, etc.

### API Layer
- **FastAPI Server**: Production-ready REST API
- **SSE Streaming**: Real-time progress updates
- **Flexible Auth**: JWT, cookies, API keys, OAuth
- **Built-in Docs**: OpenAPI/Swagger documentation

## Installation

```bash
# Basic installation
pip install agentic-sql

# With all LLM providers
pip install agentic-sql[all-llms]

# With API server
pip install agentic-sql[api]

# Full installation
pip install agentic-sql[all]
```

## Quick Start

```python
import asyncio
from agentic_sql import (
    SQLAgent, SQLAgentConfig,
    MemoryManager, MemoryConfig,
    OpenAIClient, LLMConfig,
    ToolRegistry, UserContext,
)

async def main():
    # Create memory manager
    memory = MemoryManager(MemoryConfig())

    # Create LLM client
    llm = OpenAIClient(LLMConfig(
        model="gpt-4",
        api_key="your-api-key",
    ))

    # Create tool registry and SQL agent
    registry = ToolRegistry()
    agent = SQLAgent(
        config=SQLAgentConfig(),
        tool_registry=registry,
        memory=memory,
        llm_client=llm,
        db_executor=your_db.execute,
    )

    # Execute query
    user = UserContext(user_id="user123", roles=["analyst"])
    result = await agent.execute(
        "What were total sales last month?",
        user_context=user,
    )

    print(result.data)

asyncio.run(main())
```

## Running the API Server

```python
from agentic_sql import create_app, APIConfig, JWTUserResolver

# Setup
config = APIConfig(port=8000)
resolver = JWTUserResolver(secret_key="your-secret")

app = create_app(
    config=config,
    sql_agent=your_agent,
    user_resolver=resolver.resolve,
)

# Run with: uvicorn main:app
```

## Multi-Agent Workflows

```python
from agentic_sql import AgentOrchestrator, PipelineBuilder

# Create orchestrator
orchestrator = AgentOrchestrator()
orchestrator.register_agent("sql", sql_agent)
orchestrator.register_agent("analyst", analyst_agent)
orchestrator.register_agent("validator", validator_agent)

# Build pipeline
result = await (
    PipelineBuilder(orchestrator)
    .create("analysis_pipeline")
    .add("validator")  # Validate first
    .add("sql")        # Then execute
    .add("analyst")    # Then analyze
    .run(user_context=user, initial_input="Show sales trends")
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                       │
│                    SSE Streaming + Auth                          │
├─────────────────────────────────────────────────────────────────┤
│                     Agent Orchestrator                           │
│              Multi-Agent Workflows + DAG Execution               │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  SQL Agent   │   Analyst    │  Validator   │  Custom Agents    │
│  (ReAct)     │   Agent      │   Agent      │                   │
├──────────────┴──────────────┴──────────────┴───────────────────┤
│                      Tool Registry                               │
│         Schema │ Execute │ Validate │ Visualize                 │
├─────────────────────────────────────────────────────────────────┤
│                    Memory Manager                                │
│         Graph Store │ Vector Store │ SQL Store                  │
├─────────────────────────────────────────────────────────────────┤
│                      LLM Router                                  │
│       OpenAI │ Anthropic │ Ollama │ Azure │ Bedrock             │
└─────────────────────────────────────────────────────────────────┘
```

## Design Patterns

### 1. ReAct (Reason + Act)
```python
async def execute(self, input_data):
    while not done:
        thought = await self.think(context, input_data)  # Reason
        action = await self.act(context, thought)         # Act
        observation = observe(action.result)              # Observe
        if should_reflect:
            reflection = await self.reflect(context)      # Reflect
```

### 2. Tool Registry
```python
@tool(
    name="execute_sql",
    description="Run SQL query",
    permission_level=PermissionLevel.AUTHENTICATED,
)
async def execute_sql(user_context: UserContext, sql: str):
    # Automatic permission checking and audit logging
    return await db.execute(sql)
```

### 3. Hybrid Memory
```python
# Add -> Cognify -> Memify -> Store
memory = await manager.ingest(
    content="SELECT * FROM users WHERE active = true",
    memory_type=MemoryType.QUERY_PATTERN,
    entity_id=user.id,
)

# Search with graph+vector hybrid
results = await manager.search(
    query="user queries",
    include_graph_context=True,
)
```

## Configuration

### Environment Variables
```bash
# LLM
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=gpt-4

# API
PORT=8000
JWT_SECRET=your-secret
DEBUG=false

# Database
DATABASE_URL=postgresql://...
```

## License

MIT

## Credits

Inspired by:
- [Vanna AI](https://github.com/vanna-ai/vanna) - Text-to-SQL framework
- [Cognee](https://github.com/topoteretes/cognee) - AI memory architecture
- [Memori](https://github.com/MemoriLabs/Memori) - SQL native memory layer
- Agentic AI Design Patterns (2026 Edition)
