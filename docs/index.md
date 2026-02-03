---
layout: default
title: Agentic SQL - Pure LLM Intelligence for Text-to-SQL
---

# Agentic SQL 2.0

> **Pure LLM Intelligence for Text-to-SQL**

A truly intelligent Text-to-SQL framework that works on **ANY database** with **ZERO hardcoded rules**. The system learns, adapts, and improves automatically through pure LLM intelligence.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/github/stars/bharatmohanthakur/Agentic-sql?style=social)](https://github.com/bharatmohanthakur/Agentic-sql)

---

## Why Agentic SQL?

Traditional Text-to-SQL systems rely on:
- Hardcoded regex patterns for SQL dialects
- Pre-defined error handling rules
- Manual training with example queries
- Fixed prompt templates

**Agentic SQL is different.** The LLM:

| Traditional | Agentic SQL |
|-------------|-------------|
| Manual dialect configuration | Auto-discovers SQL dialect |
| Hardcoded error patterns | LLM analyzes and fixes errors |
| Fixed training questions | Generates its own test questions |
| Static prompts | Dynamic prompts per problem |

---

## Key Features

### 1. Auto-Discovery

```python
await agent.connect(db_executor=database.execute)
# Output: "LLM detected dialect: mssql, discovered 19 schema insights"
```

### 2. Auto-Learning

```python
results = await agent.auto_learn(intensity="medium")
# Automatically trains based on your database complexity
```

### 3. Self-Healing

```python
# Automatic correction:
# "LEARNED: 'Categories' should be 'Category'"
# "LEARNED: For 'category queries' use table 'Legislations'"
```

### 4. Multi-Database Support

Works with **MS SQL Server**, **PostgreSQL**, **MySQL**, **SQLite**, and more.

---

## Quick Example

```python
import asyncio
from src.intelligence.meta_agent import MetaAgent
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

async def main():
    # Setup LLM
    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key="your-api-key",
        azure_endpoint="https://your-endpoint.openai.azure.com",
        azure_deployment="gpt-4o",
    ))

    # Create intelligent agent
    agent = MetaAgent(llm_client=llm)

    # Connect to any database
    await agent.connect(db_executor=database.execute)

    # Auto-train (questions based on your schema)
    await agent.auto_learn(intensity="light")

    # Query naturally
    result = await agent.query("How many orders last month?")
    print(result["data"])

asyncio.run(main())
```

---

## Documentation

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started) | Installation, setup, and first query |
| [API Reference](api-reference) | Complete API documentation |
| [Database Support](databases) | Connecting to different databases |
| [Memory System](memory-system) | How the agent remembers and learns |
| [Multi-Agent](multi-agent) | Complex workflows with multiple agents |
| [Examples](examples) | Real-world code examples |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/bharatmohanthakur/Agentic-sql.git
cd Agentic-sql

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

[View full installation guide →](getting-started#installation)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                   │
│                    FastAPI + SSE Streaming + Auth                        │
├─────────────────────────────────────────────────────────────────────────┤
│                            META AGENT                                    │
│           THINK → RESEARCH → DESIGN → EXECUTE → LEARN                   │
├─────────────────────────────────────────────────────────────────────────┤
│                         MEMORY MANAGER                                   │
│              Graph Store + Vector Store + SQL Store                      │
├─────────────────────────────────────────────────────────────────────────┤
│                       DATABASE ADAPTERS                                  │
│      PostgreSQL │ MySQL │ MS SQL Server │ SQLite │ Snowflake            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Test Results

| Category | Success Rate |
|----------|--------------|
| Simple Queries | 100% |
| Date Queries | 100% |
| Aggregations | 100% |
| Complex Joins | 100% |
| Window Functions | 100% |
| **Overall** | **100%** |

---

## License

MIT License - [View on GitHub](https://github.com/bharatmohanthakur/Agentic-sql/blob/main/LICENSE)

---

<p align="center">
  <a href="getting-started">Get Started →</a>
</p>
