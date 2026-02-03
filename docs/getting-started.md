---
layout: default
title: Getting Started - Agentic SQL
---

# Getting Started

Get up and running with Agentic SQL in minutes.

---

## Installation

### Prerequisites

- Python 3.10, 3.11, or 3.12
- An LLM API key (OpenAI, Azure OpenAI, or Anthropic)
- A database to connect to

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

Install extras based on your needs:

```bash
# LLM Providers
pip install -e ".[openai]"      # OpenAI
pip install -e ".[anthropic]"   # Anthropic Claude
pip install -e ".[all-llms]"    # All LLM providers

# Database Adapters
pip install -e ".[postgres]"    # PostgreSQL
pip install -e ".[mysql]"       # MySQL
pip install -e ".[sqlite]"      # SQLite

# Memory & Storage
pip install -e ".[vector]"      # Vector store (ChromaDB)
pip install -e ".[graph]"       # Graph store (Neo4j)

# API Server
pip install -e ".[api]"         # FastAPI server

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

### Optional Dependencies

| Extra | Packages |
|-------|----------|
| `all-llms` | openai, anthropic, google-generativeai |
| `api` | fastapi, uvicorn, sse-starlette |
| `postgres` | asyncpg, psycopg2-binary |
| `mysql` | aiomysql |
| `sqlite` | aiosqlite |
| `vector` | chromadb, pgvector |
| `graph` | neo4j |
| `viz` | plotly, pandas |

---

## Environment Setup

Create a `.env` file in your project root:

```bash
# LLM Configuration (choose one)
OPENAI_API_KEY=sk-...
# OR
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o
# OR
ANTHROPIC_API_KEY=sk-ant-...

# Database (optional - can configure in code)
DB_HOST=localhost
DB_NAME=mydb
DB_USER=user
DB_PASSWORD=password
```

---

## Quick Start

### Step 1: Setup LLM Client

```python
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

llm = AzureOpenAIClient(AzureOpenAIConfig(
    api_key="your-api-key",
    azure_endpoint="https://your-endpoint.openai.azure.com",
    azure_deployment="gpt-4o",
    api_version="2024-02-01",
))
```

**Other LLM options:**

```python
# OpenAI
from src.llm.openai_client import OpenAIClient, OpenAIConfig
llm = OpenAIClient(OpenAIConfig(api_key="sk-...", model="gpt-4"))

# Anthropic Claude
from src.llm.anthropic_client import AnthropicClient, AnthropicConfig
llm = AnthropicClient(AnthropicConfig(api_key="sk-ant-...", model="claude-3-opus"))
```

### Step 2: Create the Agent

```python
from src.intelligence.meta_agent import MetaAgent

agent = MetaAgent(llm_client=llm)
```

### Step 3: Connect to Database

```python
from src.database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

db = MSSQLAdapter(ConnectionConfig(
    name="my_database",
    db_type=DatabaseType.MSSQL,
    host="server.database.windows.net,1433",
    database="MyDatabase",
    username="user",
    password="password",
))
await db.connect()

# Connect agent (auto-discovers schema & dialect)
stats = await agent.connect(db_executor=db.execute)
print(f"Discovered {stats['tables']} tables")
print(f"Dialect: {stats['dialect']}")
```

### Step 4: Auto-Train (Optional)

```python
results = await agent.auto_learn(intensity="light")
print(f"Success rate: {results['success_rate']*100:.0f}%")
```

### Step 5: Query!

```python
result = await agent.query("How many orders were placed last month?")

if result["success"]:
    print(f"SQL: {result['sql']}")
    print(f"Rows: {result['row_count']}")
    print(f"Data: {result['data']}")
else:
    print(f"Error: {result['error']}")
```

---

## Complete Example

```python
import asyncio
import os
from dotenv import load_dotenv

from src.intelligence.meta_agent import MetaAgent
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
from src.database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

load_dotenv()

async def main():
    # 1. Setup LLM
    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4o",
    ))

    # 2. Create agent
    agent = MetaAgent(llm_client=llm)

    # 3. Connect to database
    db = MSSQLAdapter(ConnectionConfig(
        name="my_db",
        db_type=DatabaseType.MSSQL,
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    ))
    await db.connect()

    # 4. Connect agent
    stats = await agent.connect(db_executor=db.execute)
    print(f"Connected: {stats['tables']} tables, dialect: {stats['dialect']}")

    # 5. Auto-train
    learn_results = await agent.auto_learn(intensity="light")
    print(f"Trained: {learn_results['success_rate']*100:.0f}% success")

    # 6. Interactive queries
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['quit', 'exit']:
            break

        result = await agent.query(question)
        if result["success"]:
            print(f"SQL: {result['sql']}")
            print(f"Results: {result['data'][:5]}")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Auto-Learning

The `auto_learn()` method trains the agent on your database automatically.

### Dynamic Question Calculation

Questions are calculated based on your database complexity:

```
Formula: (tables × 2) + (columns ÷ 10) + (relationships ÷ 3) × multiplier
```

| Database Size | Tables | Columns | Light | Medium | Heavy |
|---------------|--------|---------|-------|--------|-------|
| Tiny          | 2      | 4       | 3     | 3      | 5     |
| Small         | 5      | 17      | 4     | 8      | 14    |
| Medium        | 15     | 68      | 13    | 27     | 45    |
| Large         | 40     | 334     | 39    | 78     | 100   |

### Intensity Levels

| Intensity | Multiplier | Use Case |
|-----------|------------|----------|
| `light` | 0.3× | Quick validation |
| `medium` | 0.6× | Balanced coverage |
| `heavy` | 1.0× | Comprehensive learning |
| `exhaustive` | 1.5× | Deep training |

### Example

```python
results = await agent.auto_learn(intensity="medium")

print(f"Domain: {results['domain']}")
print(f"Questions tested: {results['questions_tested']}")
print(f"Success rate: {results['success_rate']*100:.0f}%")
print(f"Schema: {results['schema_stats']}")
```

---

## What's Next?

- [API Reference](api-reference) - Complete API documentation
- [Database Support](databases) - Connect to different databases
- [Memory System](memory-system) - How the agent learns
- [Examples](examples) - More code examples

---

<p align="center">
  <a href="api-reference">API Reference →</a>
</p>
