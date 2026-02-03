# SQL Agent

Converts natural language to SQL using ReAct pattern.

---

## Overview

The SQL Agent implements:

- **ReAct** - Reasoning + Acting loop
- **Reflection** - Self-correction
- **Tool Use** - Database execution

---

## Configuration

```python
from src.agents.sql_agent import SQLAgent, SQLAgentConfig

agent = SQLAgent(
    config=SQLAgentConfig(
        name="sql_agent",
        max_sql_retries=3,
        enable_query_validation=True,
        block_destructive_queries=True,
    ),
    llm_client=llm,
    db_executor=db.execute,
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_sql_retries` | `int` | `3` | Max retry attempts |
| `enable_query_validation` | `bool` | `True` | Validate SQL |
| `block_destructive_queries` | `bool` | `True` | Block DROP/DELETE |
