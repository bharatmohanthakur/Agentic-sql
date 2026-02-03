# Quick Start

Your first query in 5 minutes.

---

## Step 1: Setup LLM Client

=== "Azure OpenAI"

    ```python
    from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key="your-api-key",
        azure_endpoint="https://your-endpoint.openai.azure.com",
        azure_deployment="gpt-4o",
        api_version="2024-02-01",
    ))
    ```

=== "OpenAI"

    ```python
    from src.llm.openai_client import OpenAIClient, OpenAIConfig

    llm = OpenAIClient(OpenAIConfig(
        api_key="sk-...",
        model="gpt-4",
    ))
    ```

=== "Anthropic Claude"

    ```python
    from src.llm.anthropic_client import AnthropicClient, AnthropicConfig

    llm = AnthropicClient(AnthropicConfig(
        api_key="sk-ant-...",
        model="claude-3-opus-20240229",
    ))
    ```

---

## Step 2: Create the Agent

```python
from src.intelligence.meta_agent import MetaAgent

agent = MetaAgent(llm_client=llm)
```

That's it! No configuration needed.

---

## Step 3: Connect to Database

=== "MS SQL Server"

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
    ```

=== "PostgreSQL"

    ```python
    from src.database.multi_db import PostgreSQLAdapter, ConnectionConfig, DatabaseType

    db = PostgreSQLAdapter(ConnectionConfig(
        name="my_database",
        db_type=DatabaseType.POSTGRESQL,
        host="localhost",
        port=5432,
        database="mydb",
        username="postgres",
        password="password",
    ))
    await db.connect()
    ```

=== "SQLite"

    ```python
    from src.database.multi_db import SQLiteAdapter, ConnectionConfig, DatabaseType

    db = SQLiteAdapter(ConnectionConfig(
        name="my_database",
        db_type=DatabaseType.SQLITE,
        database="/path/to/database.db",  # or ":memory:"
    ))
    await db.connect()
    ```

---

## Step 4: Connect Agent

```python
stats = await agent.connect(db_executor=db.execute)

print(f"Dialect: {stats['dialect']}")      # e.g., "mssql"
print(f"Tables: {stats['tables']}")        # e.g., 25
print(f"Insights: {stats['schema_insights']}")  # e.g., 19
```

!!! success "What happens automatically"
    - Probes database to detect SQL dialect
    - Discovers all tables and columns
    - Identifies relationships
    - Learns naming conventions

---

## Step 5: Query!

```python
result = await agent.query("How many orders were placed last month?")

if result["success"]:
    print(f"SQL: {result['sql']}")
    print(f"Rows: {result['row_count']}")
    print(f"Data: {result['data']}")
else:
    print(f"Error: {result['error']}")
```

### Response Structure

```python
{
    "success": True,
    "sql": "SELECT COUNT(*) FROM orders WHERE ...",
    "data": [{"count": 1234}],
    "row_count": 1,
    "iterations": 1,  # Self-correction attempts
    "problem_type": "counting",
    "execution_time_ms": 145.5,
}
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

    # 4. Connect agent (auto-discovers everything)
    stats = await agent.connect(db_executor=db.execute)
    print(f"Connected: {stats['tables']} tables")

    # 5. Query
    result = await agent.query("Show top 10 customers by revenue")

    if result["success"]:
        print(f"\nSQL:\n{result['sql']}\n")
        print(f"Results ({result['row_count']} rows):")
        for row in result["data"]:
            print(f"  {row}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Next Steps

- [Auto-Learning](../guide/auto-learning.md) - Train the agent on your database
- [Query Processing](../guide/query-processing.md) - How queries work
- [API Reference](../api/meta-agent.md) - Full API documentation
