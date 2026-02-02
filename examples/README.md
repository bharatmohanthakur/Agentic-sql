# Agentic SQL - Tutorials & Examples

Step-by-step tutorials to master Agentic SQL.

## Prerequisites

1. Python 3.10+
2. Install the package:
   ```bash
   cd ..
   pip install -e .
   ```
3. Set up your LLM API key:
   ```bash
   export AZURE_OPENAI_API_KEY=your-key
   export AZURE_OPENAI_ENDPOINT=your-endpoint
   # Or for OpenAI:
   export OPENAI_API_KEY=sk-...
   ```

## Tutorial Overview

| # | Tutorial | Description | Time |
|---|----------|-------------|------|
| 01 | [Quickstart](01_quickstart.py) | Your first query in 5 minutes | 5 min |
| 02 | [Databases](02_databases.py) | Connect to MSSQL, PostgreSQL, MySQL, SQLite | 10 min |
| 03 | [Auto-Learning](03_auto_learning.py) | Train the agent automatically | 10 min |
| 04 | [Memory System](04_memory_system.py) | Persistent knowledge storage | 15 min |
| 05 | [Multi-Agent](05_multi_agent.py) | Complex workflows with multiple agents | 15 min |
| 06 | [API Server](06_api_server.py) | Production REST API with FastAPI | 15 min |
| 07 | [Complete Example](07_complete_example.py) | Full analytics assistant demo | 20 min |

## Quick Start

```bash
# Run the quickstart tutorial
python 01_quickstart.py

# Or jump straight to the complete demo
python 07_complete_example.py
```

## Tutorial Details

### 01 - Quickstart

Learn the basics:
- Setting up an LLM client
- Creating a MetaAgent
- Connecting to a database
- Asking your first questions

```bash
python 01_quickstart.py
```

### 02 - Databases

Connect to different database types:
- MS SQL Server
- PostgreSQL
- MySQL
- SQLite
- Connection strings

```bash
python 02_databases.py
```

### 03 - Auto-Learning

Train the agent automatically:
- Domain exploration
- Question generation
- Self-testing
- Weak area improvement

```bash
python 03_auto_learning.py
```

### 04 - Memory System

Understand persistent memory:
- Memory types
- ECL pipeline
- Semantic search
- Memory hierarchy

```bash
python 04_memory_system.py
```

### 05 - Multi-Agent Workflows

Build complex pipelines:
- Agent types
- Sequential execution
- Parallel execution
- Error handling

```bash
python 05_multi_agent.py
```

### 06 - API Server

Deploy a production API:
- FastAPI setup
- SSE streaming
- Authentication
- Rate limiting

```bash
python 06_api_server.py
```

### 07 - Complete Example

Full interactive demo:
- Sample database
- Auto-learning
- Interactive session
- Real business queries

```bash
python 07_complete_example.py
```

## Common Issues

### "No module named 'llm'"
Make sure you're running from the examples directory:
```bash
cd examples
python 01_quickstart.py
```

### "AZURE_OPENAI_API_KEY not set"
Set your API key:
```bash
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
```

### Database connection failed
Check your database credentials and ensure the server is running.

## Need Help?

- Check the [main README](../README.md) for detailed documentation
- Open an issue on GitHub
- See the source code in `src/`
