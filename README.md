# Agentic SQL 2.0

> **Pure LLM Intelligence for Text-to-SQL**

A truly intelligent Text-to-SQL framework that works on **ANY database** with **ZERO hardcoded rules**. The system learns, adapts, and improves automatically through pure LLM intelligence.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Philosophy](#philosophy)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Tutorials & Examples](#tutorials--examples)
- [Quick Start](#quick-start)
- [Database Support](#database-support)
- [Memory System](#memory-system)
- [Multi-Agent Workflows](#multi-agent-workflows)
- [API Server](#api-server)
- [Security](#security)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Test Results](#test-results)
- [Contributing](#contributing)

---

## Philosophy

> **"No rules, no patterns, no hardcoding - just pure LLM understanding"**

Traditional Text-to-SQL systems rely on:
- Hardcoded regex patterns for SQL dialects
- Pre-defined error handling rules
- Manual training with example queries
- Fixed prompt templates

**Agentic SQL is different.** The LLM:
- **Discovers** database schema and dialect automatically
- **Generates** its own test questions to learn
- **Learns** from every success and failure
- **Adapts** to any database without code changes

---

## Key Features

### 1. Auto-Discovery

When connecting to a new database, the system automatically:

```python
await agent.connect(db_executor=database.execute)
```

**What happens behind the scenes:**
- Probes database to detect SQL dialect (MSSQL, PostgreSQL, MySQL, SQLite, Oracle)
- Discovers all tables, columns, and data types
- Identifies relationships between tables
- Learns naming conventions and patterns
- Stores insights for future queries

```
Output:
  "Connected to MS SQL Server: MyDatabase"
  "LLM detected dialect: mssql"
  "LLM discovered 19 schema insights"
```

### 2. Auto-Learning

The system trains itself with a single command:

```python
results = await agent.auto_learn(intensity="medium")
```

**The Auto-Learn Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│                      AUTO-LEARN FLOW                         │
├─────────────────────────────────────────────────────────────┤
│  Step 1: LLM explores database and understands domain       │
│  Step 2: Calculate questions based on schema complexity     │
│  Step 3: LLM generates diverse test questions per table     │
│  Step 4: LLM runs questions and learns from results         │
│  Step 5: LLM identifies weak areas                          │
│  Step 6: LLM generates targeted questions to improve        │
│  Step 7: Knowledge is persisted for future sessions         │
└─────────────────────────────────────────────────────────────┘
```

**Dynamic Question Calculation:**

Training questions are calculated based on your database schema, not hardcoded:

```
Formula: (tables × 2) + (columns ÷ 10) + (relationships ÷ 3) × intensity_multiplier
```

| Database Size | Tables | Columns | Light | Medium | Heavy | Exhaustive |
|---------------|--------|---------|-------|--------|-------|------------|
| Tiny          | 2      | 4       | 3     | 3      | 5     | 7          |
| Small         | 5      | 17      | 4     | 8      | 14    | 21         |
| Medium        | 15     | 68      | 13    | 27     | 45    | 67         |
| Large         | 40     | 334     | 39    | 78     | 100   | 100        |
| Very Large    | 80     | 880     | 82    | 100    | 100   | 100        |

**Intensity Levels:**
- `light` - 0.3× multiplier, quick validation
- `medium` - 0.6× multiplier, balanced coverage
- `heavy` - 1.0× multiplier, comprehensive learning
- `exhaustive` - 1.5× multiplier, deep training

**Bounds:** Minimum 3 questions, Maximum 100 questions (to prevent excessive API calls)

### 3. Intelligent Query Processing

Every query goes through the **THINK-RESEARCH-DESIGN-EXECUTE-LEARN** loop:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  THINK   │───▶│ RESEARCH │───▶│  DESIGN  │───▶│ EXECUTE  │───▶│  LEARN   │
│          │    │          │    │          │    │          │    │          │
│ Classify │    │ Find     │    │ Generate │    │ Run SQL  │    │ Store    │
│ problem  │    │ similar  │    │ dynamic  │    │ Fix if   │    │ solution │
│ type     │    │ solutions│    │ prompt   │    │ needed   │    │ & errors │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 4. Self-Healing

When errors occur, the LLM:
1. Analyzes the error message
2. Searches database for correct table/column names
3. Learns corrections for future queries
4. Retries with fixed SQL (up to 4 attempts)

```python
# Automatic learning from errors:
# "LEARNED: 'Categories' should be 'Category'"
# "LEARNED: For 'category queries' use table 'Legislations'"
```

### 5. Dynamic Prompts

No fixed templates. Every prompt is generated dynamically based on:
- Problem type classification
- Similar successful solutions from history
- Previous failure patterns to avoid
- Learned dialect specifics
- Schema insights and relationships

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                   │
│                    FastAPI + SSE Streaming + Auth                        │
├─────────────────────────────────────────────────────────────────────────┤
│                          AGENT ORCHESTRATOR                              │
│              Multi-Agent Workflows + DAG Execution + Retries             │
├───────────────┬───────────────┬───────────────┬─────────────────────────┤
│   SQL Agent   │    Analyst    │   Validator   │     Custom Agents       │
│   (ReAct)     │    Agent      │    Agent      │                         │
├───────────────┴───────────────┴───────────────┴─────────────────────────┤
│                            META AGENT                                    │
│           THINK → RESEARCH → DESIGN → EXECUTE → LEARN                   │
├─────────────────────────────────────────────────────────────────────────┤
│                          TOOL REGISTRY                                   │
│         Database Tools │ Visualization │ Validation │ Custom            │
├─────────────────────────────────────────────────────────────────────────┤
│                         MEMORY MANAGER                                   │
│              Graph Store + Vector Store + SQL Store                      │
│                     (ECL: Extract, Cognify, Load)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                           LLM ROUTER                                     │
│      OpenAI │ Azure OpenAI │ Anthropic │ Google │ Ollama │ Bedrock      │
├─────────────────────────────────────────────────────────────────────────┤
│                       DATABASE ADAPTERS                                  │
│      PostgreSQL │ MySQL │ MS SQL Server │ SQLite │ Snowflake │ BigQuery │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/bharatmohanthakur/Agentic-sql.git
cd Agentic-sql

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Optional Dependencies

```bash
# All LLM providers
pip install -e ".[all-llms]"

# API server with FastAPI
pip install -e ".[api]"

# PostgreSQL support
pip install -e ".[postgres]"

# MySQL support
pip install -e ".[mysql]"

# Vector database for memory
pip install -e ".[vector]"

# Graph database for memory
pip install -e ".[graph]"

# Visualization support
pip install -e ".[viz]"

# Full installation
pip install -e ".[all]"
```

---

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥2.16.0 | OpenAI/Azure OpenAI LLM client |
| `pydantic` | ≥2.0.0 | Data validation and settings |
| `pyodbc` | ≥5.3.0 | MS SQL Server connectivity |
| `python-dotenv` | ≥1.2.1 | Environment variable management |
| `typing-extensions` | ≥4.0.0 | Extended type hints |

### Optional Dependencies

Install specific extras based on your needs:

```bash
# LLM Providers
pip install -e ".[openai]"      # OpenAI (included in core)
pip install -e ".[anthropic]"   # Anthropic Claude
pip install -e ".[all-llms]"    # All LLM providers (OpenAI, Anthropic, Google)

# Database Adapters
pip install -e ".[postgres]"    # PostgreSQL (asyncpg, psycopg2)
pip install -e ".[mysql]"       # MySQL (aiomysql)
pip install -e ".[sqlite]"      # SQLite (aiosqlite)
pip install -e ".[snowflake]"   # Snowflake
pip install -e ".[bigquery]"    # Google BigQuery

# Memory & Storage
pip install -e ".[vector]"      # Vector store (ChromaDB, pgvector)
pip install -e ".[qdrant]"      # Qdrant vector store
pip install -e ".[opensearch]"  # OpenSearch hybrid store
pip install -e ".[graph]"       # Graph store (Neo4j)
pip install -e ".[memory]"      # Basic memory (ChromaDB + Neo4j)
pip install -e ".[memory-all]"  # All memory backends

# API & Auth
pip install -e ".[api]"         # FastAPI server (FastAPI, uvicorn, SSE)
pip install -e ".[auth]"        # JWT authentication

# Visualization
pip install -e ".[viz]"         # Charts (Plotly, Pandas)

# Development
pip install -e ".[dev]"         # Testing & linting tools
```

### Full Dependency Matrix

| Extra | Packages |
|-------|----------|
| `openai` | openai≥1.0.0 |
| `anthropic` | anthropic≥0.18.0 |
| `all-llms` | openai, anthropic, google-generativeai |
| `api` | fastapi≥0.100.0, uvicorn≥0.22.0, sse-starlette≥1.6.0 |
| `auth` | PyJWT≥2.8.0 |
| `postgres` | asyncpg≥0.28.0, psycopg2-binary≥2.9.0 |
| `mysql` | aiomysql≥0.2.0 |
| `sqlite` | aiosqlite≥0.19.0 |
| `snowflake` | snowflake-connector-python≥3.0.0 |
| `bigquery` | google-cloud-bigquery≥3.0.0 |
| `vector` | chromadb≥0.4.0, pgvector≥0.2.0 |
| `qdrant` | qdrant-client≥1.7.0 |
| `opensearch` | opensearch-py≥2.4.0 |
| `graph` | neo4j≥5.0.0 |
| `memory-all` | chromadb, neo4j, qdrant-client, opensearch-py, aiosqlite |
| `viz` | plotly≥5.0.0, pandas≥2.0.0 |
| `dev` | pytest, pytest-asyncio, black, ruff, mypy |

### System Requirements

- **Python**: 3.10, 3.11, or 3.12
- **MS SQL Server**: ODBC Driver 17 or 18 (for MSSQL connections)
- **Memory**: 4GB+ RAM recommended for vector embeddings

---

## Tutorials & Examples

Learn by doing with our step-by-step tutorials in the [`examples/`](examples/) directory:

| Tutorial | Description |
|----------|-------------|
| [01_quickstart.py](examples/01_quickstart.py) | Your first query in 5 minutes |
| [02_databases.py](examples/02_databases.py) | Connect to MSSQL, PostgreSQL, MySQL, SQLite |
| [03_auto_learning.py](examples/03_auto_learning.py) | Train the agent automatically |
| [04_memory_system.py](examples/04_memory_system.py) | Persistent knowledge storage |
| [05_multi_agent.py](examples/05_multi_agent.py) | Complex workflows with multiple agents |
| [06_api_server.py](examples/06_api_server.py) | Production REST API with FastAPI |
| [07_complete_example.py](examples/07_complete_example.py) | Full interactive analytics demo |

```bash
# Run the complete interactive demo
cd examples
python 07_complete_example.py
```

---

## Quick Start

### Basic Usage

```python
import asyncio
from src.intelligence.meta_agent import MetaAgent
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
from src.database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

async def main():
    # 1. Setup LLM client
    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key="your-api-key",
        azure_endpoint="https://your-endpoint.openai.azure.com",
        azure_deployment="gpt-4o",
        api_version="2024-02-01",
    ))

    # 2. Create the intelligent agent
    agent = MetaAgent(llm_client=llm)

    # 3. Connect to database
    db = MSSQLAdapter(ConnectionConfig(
        name="my_database",
        db_type=DatabaseType.MSSQL,
        host="server.database.windows.net,1433",
        database="MyDatabase",
        username="user",
        password="password",
    ))
    await db.connect()

    # 4. Connect agent (auto-discovers schema & dialect)
    await agent.connect(db_executor=db.execute)

    # 5. Optional: Auto-train on the database
    await agent.auto_learn(intensity="light")

    # 6. Query naturally
    result = await agent.query("How many orders were placed last month?")

    if result["success"]:
        print(f"SQL: {result['sql']}")
        print(f"Rows: {result['row_count']}")
        print(f"Data: {result['data']}")

asyncio.run(main())
```

### Using Different LLM Providers

```python
# OpenAI
from src.llm.openai_client import OpenAIClient, OpenAIConfig
llm = OpenAIClient(OpenAIConfig(
    api_key="sk-...",
    model="gpt-4",
))

# Anthropic Claude
from src.llm.anthropic_client import AnthropicClient, AnthropicConfig
llm = AnthropicClient(AnthropicConfig(
    api_key="sk-ant-...",
    model="claude-3-opus-20240229",
))

# Azure OpenAI
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
llm = AzureOpenAIClient(AzureOpenAIConfig(
    api_key="your-key",
    azure_endpoint="https://your-endpoint.openai.azure.com",
    azure_deployment="gpt-4o",
))
```

---

## Database Support

The system auto-detects and adapts to any SQL database:

| Database | Adapter | Auto-Detected Features |
|----------|---------|------------------------|
| MS SQL Server | `MSSQLAdapter` | TOP, GETDATE, square brackets, T-SQL |
| PostgreSQL | `PostgreSQLAdapter` | LIMIT, NOW(), double quotes, PL/pgSQL |
| MySQL | `MySQLAdapter` | LIMIT, backticks, NOW(), AUTO_INCREMENT |
| SQLite | `SQLiteAdapter` | LIMIT, datetime functions, no types |
| Oracle | Coming Soon | ROWNUM, SYSDATE, PL/SQL |
| Snowflake | Coming Soon | LIMIT, warehouse syntax |
| BigQuery | Coming Soon | LIMIT, backticks, standard SQL |

### Database Initialization Examples

#### MS SQL Server

```python
from src.database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

db = MSSQLAdapter(ConnectionConfig(
    name="my_mssql_db",
    db_type=DatabaseType.MSSQL,
    host="server.database.windows.net,1433",
    database="MyDatabase",
    username="user",
    password="password",
))
await db.connect()
```

#### PostgreSQL

```python
from src.database.multi_db import PostgreSQLAdapter, ConnectionConfig, DatabaseType

db = PostgreSQLAdapter(ConnectionConfig(
    name="my_postgres_db",
    db_type=DatabaseType.POSTGRESQL,
    host="localhost",
    port=5432,
    database="mydb",
    username="postgres",
    password="password",
))
await db.connect()
```

#### MySQL

```python
from src.database.multi_db import MySQLAdapter, ConnectionConfig, DatabaseType

db = MySQLAdapter(ConnectionConfig(
    name="my_mysql_db",
    db_type=DatabaseType.MYSQL,
    host="localhost",
    port=3306,
    database="mydb",
    username="root",
    password="password",
))
await db.connect()
```

#### SQLite

```python
from src.database.multi_db import SQLiteAdapter, ConnectionConfig, DatabaseType

db = SQLiteAdapter(ConnectionConfig(
    name="my_sqlite_db",
    db_type=DatabaseType.SQLITE,
    database="/path/to/database.db",
))
await db.connect()
```

#### Using Connection String

```python
db = PostgreSQLAdapter(ConnectionConfig(
    name="my_db",
    db_type=DatabaseType.POSTGRESQL,
    connection_string="postgresql://user:pass@localhost:5432/mydb",
))
await db.connect()
```

---

## Memory System

The system includes a hybrid memory architecture combining **Graph + Vector + SQL** storage for intelligent context management.

### Storage Backend Options

| Store | Type | Best For | Install |
|-------|------|----------|---------|
| SQLite | File-based | Simple deployments | `pip install -e ".[sqlite]"` |
| ChromaDB | Vector | Semantic search, local | `pip install -e ".[vector]"` |
| Qdrant | Vector | High-performance | `pip install -e ".[qdrant]"` |
| OpenSearch | Hybrid | Vector + full-text | `pip install -e ".[opensearch]"` |
| Neo4j | Graph | Entity relationships | `pip install -e ".[graph]"` |

### Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MEMORY MANAGER                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │   GRAPH     │  │   VECTOR    │  │    SQL      │        │
│   │   STORE     │  │   STORE     │  │   STORE     │        │
│   │             │  │             │  │             │        │
│   │ Entities &  │  │ Semantic    │  │ Structured  │        │
│   │ Relations   │  │ Embeddings  │  │ Persistence │        │
│   │  (Neo4j)    │  │(Chroma/     │  │  (SQLite)   │        │
│   │             │  │ Qdrant/     │  │             │        │
│   │             │  │ OpenSearch) │  │             │        │
│   └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│              ECL PIPELINE (Extract, Cognify, Load)           │
└─────────────────────────────────────────────────────────────┘
```

### Memory Types

```python
from src.memory.manager import MemoryManager, MemoryConfig, MemoryType

# Initialize memory manager
memory = MemoryManager(MemoryConfig(
    vector_store_path="./vector_store",
    graph_store_path="./graph_store",
    sql_store_path="./memory.db",
))

# Available memory types:
MemoryType.CONVERSATION    # Chat history
MemoryType.ENTITY_FACT     # Facts about entities
MemoryType.SCHEMA          # Database schema knowledge
MemoryType.QUERY_PATTERN   # Successful SQL patterns
MemoryType.ERROR_PATTERN   # Error patterns to avoid
MemoryType.USER_PREFERENCE # User-specific preferences
MemoryType.SEMANTIC        # General semantic knowledge
```

### ECL Pipeline

The **Extract-Cognify-Load** pattern processes all memories:

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│  EXTRACT   │────▶│  COGNIFY   │────▶│    LOAD    │
│            │     │            │     │            │
│ Parse raw  │     │ Generate   │     │ Store in   │
│ content    │     │ embeddings │     │ hybrid     │
│ and meta   │     │ & links    │     │ storage    │
└────────────┘     └────────────┘     └────────────┘
```

### Memory Usage

```python
# Add a memory
await memory.add(
    content="SELECT * FROM users WHERE active = true",
    memory_type=MemoryType.QUERY_PATTERN,
    metadata={"table": "users", "success": True},
)

# Search memories with hybrid retrieval
results = await memory.search(
    query="user queries",
    memory_types=[MemoryType.QUERY_PATTERN],
    limit=5,
)

# Get related memories via graph traversal
related = await memory.get_related(
    memory_id=some_memory_id,
    depth=2,
)
```

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     MEMORY LEVELS                            │
├─────────────────────────────────────────────────────────────┤
│  GLOBAL    │ Shared knowledge across all users              │
│  ENTITY    │ Facts about database entities                  │
│  USER      │ User-specific preferences and patterns         │
│  SESSION   │ Current conversation context                   │
└─────────────────────────────────────────────────────────────┘
```

### Using Different Memory Stores

```python
from memory.stores import (
    SQLiteMemoryStore,
    ChromaMemoryStore,
    QdrantMemoryStore,
    OpenSearchMemoryStore,
    Neo4jMemoryStore,
)

# SQLite - Simple, no server required
from memory.stores.sqlite_store import SQLiteConfig
sqlite_store = SQLiteMemoryStore(SQLiteConfig(db_path="./memories.db"))

# ChromaDB - Local vector search
from memory.stores.chroma_store import ChromaConfig
chroma_store = ChromaMemoryStore(ChromaConfig(path="./chroma_data"))

# Qdrant - High-performance vector search
from memory.stores.qdrant_store import QdrantConfig
qdrant_store = QdrantMemoryStore(QdrantConfig(host="localhost", port=6333))

# OpenSearch - Hybrid vector + text search
from memory.stores.opensearch_store import OpenSearchConfig
opensearch_store = OpenSearchMemoryStore(OpenSearchConfig(
    hosts=["https://localhost:9200"],
    username="admin",
    password="admin",
))

# Neo4j - Graph relationships
from memory.stores.neo4j_store import Neo4jConfig
neo4j_store = Neo4jMemoryStore(Neo4jConfig(uri="bolt://localhost:7687"))

# Connect and use
await store.connect()
await store.store(memory)
results = await store.retrieve("query", limit=10)
```

---

## Multi-Agent Workflows

The system supports complex multi-agent workflows with DAG-based orchestration.

### Available Agents

| Agent | Purpose |
|-------|---------|
| `SQLAgent` | Converts natural language to SQL using ReAct pattern |
| `AnalystAgent` | Analyzes query results and generates insights |
| `ValidatorAgent` | Validates SQL for security and correctness |

### Creating Workflows

```python
from src.core.orchestrator import AgentOrchestrator, PipelineBuilder
from src.agents.sql_agent import SQLAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.validator_agent import ValidatorAgent

# Create orchestrator
orchestrator = AgentOrchestrator()

# Register agents
orchestrator.register_agent("sql", sql_agent)
orchestrator.register_agent("analyst", analyst_agent)
orchestrator.register_agent("validator", validator_agent)

# Build and run pipeline
result = await (
    PipelineBuilder(orchestrator)
    .create("analysis_pipeline")
    .add("validator")    # Step 1: Validate query
    .add("sql")          # Step 2: Generate and execute SQL
    .add("analyst")      # Step 3: Analyze results
    .run(user_context=user, initial_input="Show sales trends")
)
```

### Workflow Features

- **DAG-based dependencies** - Tasks run in correct order
- **Parallel execution** - Independent tasks run simultaneously
- **Error handling** - Automatic retries with backoff
- **Progress tracking** - Real-time status updates
- **Result aggregation** - Combined output from all agents

---

## API Server

Production-ready REST API with FastAPI and Server-Sent Events for streaming.

### Starting the Server

```python
from src.api.server import create_app, APIConfig
from src.api.auth import JWTUserResolver

# Configure
config = APIConfig(
    host="0.0.0.0",
    port=8000,
    cors_origins=["*"],
    rate_limit_per_minute=100,
)

# Setup authentication
resolver = JWTUserResolver(
    secret_key="your-secret-key",
    algorithm="HS256",
)

# Create app
app = create_app(
    config=config,
    sql_agent=your_agent,
    user_resolver=resolver.resolve,
)

# Run with: uvicorn main:app
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Execute natural language query |
| POST | `/query/stream` | Execute with SSE streaming |
| GET | `/health` | Health check |
| GET | `/schema` | Get database schema |

### Streaming Response

```python
# Client-side SSE handling
const eventSource = new EventSource('/query/stream?q=Show+sales');

eventSource.addEventListener('thinking', (e) => {
    console.log('Thinking:', JSON.parse(e.data));
});

eventSource.addEventListener('sql', (e) => {
    console.log('SQL:', JSON.parse(e.data));
});

eventSource.addEventListener('data', (e) => {
    console.log('Results:', JSON.parse(e.data));
});

eventSource.addEventListener('done', (e) => {
    eventSource.close();
});
```

---

## Security

### Authentication Options

```python
from src.api.auth import JWTUserResolver, APIKeyResolver, OAuthResolver

# JWT Authentication
jwt_resolver = JWTUserResolver(
    secret_key="your-secret",
    algorithm="HS256",
)

# API Key Authentication
api_key_resolver = APIKeyResolver(
    api_keys={"key1": "user1", "key2": "user2"},
)

# OAuth Authentication
oauth_resolver = OAuthResolver(
    provider="auth0",
    domain="your-domain.auth0.com",
)
```

### Row-Level Security

```python
from src.core.base import UserContext

# User context with permissions
user = UserContext(
    user_id="user123",
    roles=["analyst"],
    permissions={
        "allowed_schemas": ["sales", "marketing"],
        "allowed_tables": ["orders", "customers"],
        "sql_filters": {
            "orders": "region = 'US'",  # Auto-applied to queries
        },
    },
)

# Execute with security context
result = await agent.query(
    "Show all orders",
    user_context=user,
)
# SQL automatically includes: WHERE region = 'US'
```

### Security Features

- **SQL Injection Protection** - Parameterized queries
- **Destructive Query Blocking** - DROP, DELETE, TRUNCATE blocked by default
- **Schema Filtering** - Users only see allowed schemas
- **Audit Logging** - All queries logged with user context
- **Rate Limiting** - Per-user request limits

---

## Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com

# Database (optional, can be configured in code)
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb

# API Server
API_HOST=0.0.0.0
API_PORT=8000
JWT_SECRET=your-secret-key

# Memory Storage
VECTOR_STORE_PATH=./vector_store
GRAPH_STORE_PATH=./graph_store

# Logging
LOG_LEVEL=INFO
```

### Knowledge Persistence

All learnings are automatically saved to `~/.vanna/meta_agent.json`:

```json
{
  "successful_solutions": [
    {
      "question": "How many users?",
      "sql": "SELECT COUNT(*) FROM users",
      "problem_type": "counting"
    }
  ],
  "failed_attempts": [...],
  "dialect_learnings": ["Uses TOP instead of LIMIT"],
  "schema_insights": ["Users table has email column"],
  "name_corrections": {
    "categories": "Category",
    "users": "User"
  },
  "table_relationships": {
    "orders": "joins with customers on customer_id"
  }
}
```

---

## Test Results

### Comprehensive Test Suite

| Category | Queries | Success Rate |
|----------|---------|--------------|
| Simple Queries | 3 | 100% |
| Filtering | 3 | 100% |
| Date Queries | 4 | 100% |
| Aggregations | 3 | 100% |
| Comparisons | 2 | 100% |
| Rankings | 2 | 100% |
| Complex Joins | 3 | 100% |
| Business Questions | 3 | 100% |
| **Total** | **23** | **100%** |

### Stress Test Results

| Test Type | Result | Notes |
|-----------|--------|-------|
| Window Functions | ✅ Pass | ROW_NUMBER, DENSE_RANK, running totals |
| Recursive CTEs | ✅ Pass | Hierarchical queries |
| PIVOT Tables | ✅ Pass | Dynamic pivoting |
| Self-Joins | ✅ Pass | 12,449 pairs found |
| Standard Deviation | ✅ Pass | Statistical functions |
| Arabic Language | ✅ Pass | عرض التشريعات works |
| SQL Injection | ✅ Safe | Handled securely |
| Typos | ✅ Pass | "Shwo teh legislatoins" → works |
| Vague Questions | ✅ Pass | "What's wrong with our data?" |
| Nonsense Input | ✅ Pass | Handled gracefully |

---

## API Reference

### MetaAgent

```python
class MetaAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        storage_path: Optional[Path] = None,
    ):
        """Initialize the meta-learning agent."""

    async def connect(
        self,
        db_executor: Callable,
    ) -> Dict:
        """
        Connect to database and auto-discover schema/dialect.

        Returns:
            Dict with connection info and discovered insights
        """

    async def query(
        self,
        question: str,
    ) -> Dict:
        """
        Process natural language question.

        Returns:
            {
                "success": bool,
                "sql": str,
                "data": List[Dict],
                "row_count": int,
                "iterations": int,
                "problem_type": str,
                "execution_time_ms": float,
                "error": Optional[str],
            }
        """

    async def auto_learn(
        self,
        intensity: str = "medium",
    ) -> Dict:
        """
        Self-train on the connected database.

        Args:
            intensity: "light", "medium", "heavy", or "exhaustive"
                      Questions calculated dynamically based on schema:
                      (tables × 2 + columns/10 + relationships/3) × multiplier

        Returns:
            {
                "domain": str,
                "questions_generated": int,
                "questions_tested": int,
                "successes": int,
                "failures": int,
                "success_rate": float,
                "schema_stats": {"tables": int, "total_columns": int},
                "target_questions": int,
                "intensity": str,
            }
        """

    def get_stats(self) -> Dict:
        """Get current knowledge statistics."""
```

---

## Contributing

Contributions are welcome! The key principle:

> **No hardcoding** - If you find yourself writing a rule or pattern,
> ask "Can the LLM figure this out?" The answer is usually yes.

### Development Setup

```bash
# Clone and install dev dependencies
git clone https://github.com/bharatmohanthakur/Agentic-sql.git
cd Agentic-sql
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
ruff check src tests

# Type checking
mypy src
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Credits

Built with pure LLM intelligence, inspired by:
- The vision of truly intelligent AI systems
- Modern agentic AI design patterns
- The open-source community

**Powered by:**
- OpenAI GPT-4 / Azure OpenAI
- Anthropic Claude
- And other leading LLM providers
