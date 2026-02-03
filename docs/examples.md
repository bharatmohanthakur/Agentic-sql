---
layout: default
title: Examples - Agentic SQL
---

# Examples

Real-world code examples and tutorials.

---

## Available Tutorials

| Tutorial | Description | Complexity |
|----------|-------------|------------|
| [01_quickstart.py](https://github.com/bharatmohanthakur/Agentic-sql/blob/main/examples/01_quickstart.py) | Your first query in 5 minutes | Beginner |
| [02_databases.py](https://github.com/bharatmohanthakur/Agentic-sql/blob/main/examples/02_databases.py) | Connect to different databases | Beginner |
| [03_auto_learning.py](https://github.com/bharatmohanthakur/Agentic-sql/blob/main/examples/03_auto_learning.py) | Train the agent automatically | Intermediate |
| [04_memory_system.py](https://github.com/bharatmohanthakur/Agentic-sql/blob/main/examples/04_memory_system.py) | Persistent knowledge storage | Intermediate |
| [05_multi_agent.py](https://github.com/bharatmohanthakur/Agentic-sql/blob/main/examples/05_multi_agent.py) | Complex multi-agent workflows | Advanced |
| [06_api_server.py](https://github.com/bharatmohanthakur/Agentic-sql/blob/main/examples/06_api_server.py) | Production REST API | Advanced |
| [07_complete_example.py](https://github.com/bharatmohanthakur/Agentic-sql/blob/main/examples/07_complete_example.py) | Full interactive demo | All Levels |

---

## Basic Query

```python
import asyncio
from src.intelligence.meta_agent import MetaAgent
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
from src.database.multi_db import SQLiteAdapter, ConnectionConfig, DatabaseType

async def main():
    # Setup LLM
    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key="your-key",
        azure_endpoint="https://your-endpoint.openai.azure.com",
        azure_deployment="gpt-4o",
    ))

    # Create in-memory database for testing
    db = SQLiteAdapter(ConnectionConfig(
        name="test",
        db_type=DatabaseType.SQLITE,
        database=":memory:",
    ))
    await db.connect()

    # Create sample data
    await db.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute("INSERT INTO users (name, email) VALUES ('Alice', 'alice@email.com')")
    await db.execute("INSERT INTO users (name, email) VALUES ('Bob', 'bob@email.com')")

    # Create agent and connect
    agent = MetaAgent(llm_client=llm)
    await agent.connect(db_executor=db.execute)

    # Query
    result = await agent.query("How many users are there?")
    print(f"SQL: {result['sql']}")
    print(f"Answer: {result['data']}")

asyncio.run(main())
```

---

## Auto-Learning Example

```python
async def auto_learn_example():
    # Connect to your database
    agent = MetaAgent(llm_client=llm)
    await agent.connect(db_executor=db.execute)

    # Auto-train (questions based on schema complexity)
    results = await agent.auto_learn(intensity="medium")

    print(f"Domain: {results['domain']}")
    print(f"Tables: {results['schema_stats']['tables']}")
    print(f"Questions tested: {results['questions_tested']}")
    print(f"Success rate: {results['success_rate']*100:.0f}%")

    # Now query with learned knowledge
    result = await agent.query("Show top customers by revenue")
    print(f"SQL: {result['sql']}")
```

---

## Different Databases

### MS SQL Server

```python
from src.database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

db = MSSQLAdapter(ConnectionConfig(
    name="mssql",
    db_type=DatabaseType.MSSQL,
    host="server.database.windows.net,1433",
    database="MyDB",
    username="user",
    password="pass",
))
await db.connect()

agent = MetaAgent(llm_client=llm)
await agent.connect(db_executor=db.execute)

# Agent auto-detects MSSQL and uses TOP instead of LIMIT
result = await agent.query("Show first 10 orders")
# SQL: SELECT TOP 10 * FROM orders
```

### PostgreSQL

```python
from src.database.multi_db import PostgreSQLAdapter, ConnectionConfig, DatabaseType

db = PostgreSQLAdapter(ConnectionConfig(
    name="postgres",
    db_type=DatabaseType.POSTGRESQL,
    host="localhost",
    port=5432,
    database="mydb",
    username="postgres",
    password="pass",
))
await db.connect()

agent = MetaAgent(llm_client=llm)
await agent.connect(db_executor=db.execute)

# Agent auto-detects PostgreSQL and uses LIMIT
result = await agent.query("Show first 10 orders")
# SQL: SELECT * FROM orders LIMIT 10
```

---

## API Server

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

# Auth
resolver = JWTUserResolver(secret_key="your-secret")

# Create app
app = create_app(
    config=config,
    sql_agent=agent,
    user_resolver=resolver.resolve,
)

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"question": "How many users signed up last month?"}'
```

### API Response

```json
{
  "success": true,
  "sql": "SELECT COUNT(*) FROM users WHERE created_at >= ...",
  "data": [{"count": 1234}],
  "row_count": 1,
  "execution_time_ms": 145.5
}
```

---

## Streaming with SSE

### Server

```python
@app.get("/query/stream")
async def query_stream(q: str):
    async def event_generator():
        yield {"event": "thinking", "data": {"step": "Analyzing..."}}

        result = await agent.query(q)

        yield {"event": "sql", "data": {"sql": result["sql"]}}
        yield {"event": "data", "data": {"rows": result["data"]}}
        yield {"event": "done", "data": {"success": True}}

    return EventSourceResponse(event_generator())
```

### Client (JavaScript)

```javascript
const eventSource = new EventSource('/query/stream?q=' + encodeURIComponent(question));

eventSource.addEventListener('thinking', (e) => {
    showSpinner(JSON.parse(e.data));
});

eventSource.addEventListener('sql', (e) => {
    displaySQL(JSON.parse(e.data).sql);
});

eventSource.addEventListener('data', (e) => {
    renderTable(JSON.parse(e.data).rows);
});

eventSource.addEventListener('done', (e) => {
    eventSource.close();
});
```

---

## Memory System

```python
from src.memory.manager import MemoryManager, MemoryConfig, MemoryType

# Initialize memory
memory = MemoryManager(MemoryConfig())

# Store successful pattern
await memory.add(
    content="SELECT COUNT(*) FROM users WHERE active = true",
    memory_type=MemoryType.QUERY_PATTERN,
    metadata={"question": "How many active users?", "success": True},
)

# Search memories
results = await memory.search(
    query="count users",
    memory_types=[MemoryType.QUERY_PATTERN],
    limit=5,
)

for r in results:
    print(f"Found: {r.content}")
```

---

## Interactive CLI

```python
async def interactive_cli():
    agent = MetaAgent(llm_client=llm)
    await agent.connect(db_executor=db.execute)
    await agent.auto_learn(intensity="light")

    print("Agentic SQL - Ask questions about your data")
    print("Type 'quit' to exit, 'stats' to see agent statistics")
    print("-" * 50)

    while True:
        question = input("\n> ").strip()

        if question.lower() == 'quit':
            break

        if question.lower() == 'stats':
            stats = agent.get_stats()
            print(f"Dialect: {stats['dialect']}")
            print(f"Tables: {stats['tables']}")
            print(f"Solutions learned: {stats['solutions_stored']}")
            continue

        result = await agent.query(question)

        if result["success"]:
            print(f"\nSQL: {result['sql']}")
            print(f"\nResults ({result['row_count']} rows):")
            for row in result["data"][:10]:
                print(f"  {row}")
        else:
            print(f"\nError: {result['error']}")

asyncio.run(interactive_cli())
```

---

## Error Handling

```python
async def safe_query(agent, question):
    try:
        result = await agent.query(question)

        if result["success"]:
            return {
                "status": "success",
                "sql": result["sql"],
                "data": result["data"],
                "iterations": result["iterations"],
            }
        else:
            return {
                "status": "failed",
                "error": result["error"],
                "sql_attempted": result["sql"],
            }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - DATABASE_URL=${DATABASE_URL}
      - JWT_SECRET=${JWT_SECRET}
    restart: unless-stopped
```

---

## Running the Examples

```bash
# Clone the repository
git clone https://github.com/bharatmohanthakur/Agentic-sql.git
cd Agentic-sql

# Install dependencies
pip install -e .

# Set up environment
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com

# Run quickstart
cd examples
python 01_quickstart.py

# Run complete demo
python 07_complete_example.py
```

---

<p align="center">
  <a href="https://github.com/bharatmohanthakur/Agentic-sql">View on GitHub →</a>
</p>
