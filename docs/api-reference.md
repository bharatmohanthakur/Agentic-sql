---
layout: default
title: API Reference - Agentic SQL
---

# API Reference

Complete API documentation for Agentic SQL.

---

## MetaAgent

The core intelligent agent that handles text-to-SQL conversion.

### Constructor

```python
from src.intelligence.meta_agent import MetaAgent

agent = MetaAgent(
    llm_client: LLMClient,
    storage_path: Optional[Path] = None,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm_client` | `LLMClient` | LLM client instance (OpenAI, Azure, Anthropic) |
| `storage_path` | `Optional[Path]` | Path to save learned knowledge. Default: `~/.vanna/meta_agent.json` |

---

### connect()

Connect to a database and auto-discover schema and dialect.

```python
stats = await agent.connect(
    db_executor: Callable,
    driver: Optional[str] = None,
) -> Dict
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `db_executor` | `Callable` | Async function to execute SQL queries |
| `driver` | `Optional[str]` | Hint for database driver (optional) |

**Returns:**

```python
{
    "dialect": str,      # Detected SQL dialect (mssql, postgresql, etc.)
    "tables": int,       # Number of tables discovered
    "schema_insights": int,  # Number of insights generated
}
```

**Example:**

```python
stats = await agent.connect(db_executor=db.execute)
print(f"Dialect: {stats['dialect']}")
print(f"Tables: {stats['tables']}")
```

---

### query()

Process a natural language question and return SQL results.

```python
result = await agent.query(
    question: str,
) -> Dict
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `question` | `str` | Natural language question |

**Returns:**

```python
{
    "success": bool,           # Whether query succeeded
    "sql": str,                # Generated SQL query
    "data": List[Dict],        # Query results
    "row_count": int,          # Number of rows returned
    "iterations": int,         # Self-correction attempts used
    "problem_type": str,       # LLM-classified problem type
    "execution_time_ms": float, # Total execution time
    "error": Optional[str],    # Error message if failed
}
```

**Example:**

```python
result = await agent.query("How many orders last month?")

if result["success"]:
    print(f"SQL: {result['sql']}")
    print(f"Data: {result['data']}")
    print(f"Iterations: {result['iterations']}")
else:
    print(f"Error: {result['error']}")
```

---

### auto_learn()

Self-train on the connected database.

```python
results = await agent.auto_learn(
    intensity: str = "medium",
) -> Dict
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `intensity` | `str` | Training intensity: `"light"`, `"medium"`, `"heavy"`, or `"exhaustive"` |

**Intensity Levels:**

| Level | Multiplier | Description |
|-------|------------|-------------|
| `light` | 0.3× | Quick validation |
| `medium` | 0.6× | Balanced coverage |
| `heavy` | 1.0× | Comprehensive |
| `exhaustive` | 1.5× | Deep training |

**Question Formula:**

```
(tables × 2) + (columns ÷ 10) + (relationships ÷ 3) × multiplier
```

**Returns:**

```python
{
    "domain": str,              # Detected domain (ecommerce, healthcare, etc.)
    "questions_generated": int, # Questions created by LLM
    "questions_tested": int,    # Questions actually tested
    "successes": int,           # Successful queries
    "failures": int,            # Failed queries
    "success_rate": float,      # Success ratio (0.0 - 1.0)
    "schema_stats": {
        "tables": int,
        "total_columns": int,
    },
    "target_questions": int,    # Calculated question count
    "intensity": str,           # Requested intensity
}
```

**Example:**

```python
results = await agent.auto_learn(intensity="medium")

print(f"Domain: {results['domain']}")
print(f"Success rate: {results['success_rate']*100:.0f}%")
print(f"Questions tested: {results['questions_tested']}")
```

---

### get_stats()

Get current knowledge statistics.

```python
stats = agent.get_stats() -> Dict
```

**Returns:**

```python
{
    "dialect": str,              # Current SQL dialect
    "tables": int,               # Tables in schema
    "problem_types_learned": int, # Unique problem types
    "actions_learned": int,      # Fix strategies stored
    "solutions_stored": int,     # Successful solutions
    "failures_analyzed": int,    # Analyzed failures
    "dialect_learnings": int,    # Dialect-specific learnings
    "schema_insights": int,      # Schema insights
}
```

---

## LLM Clients

### AzureOpenAIClient

```python
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

llm = AzureOpenAIClient(AzureOpenAIConfig(
    api_key: str,
    azure_endpoint: str,
    azure_deployment: str,
    api_version: str = "2024-02-01",
))
```

### OpenAIClient

```python
from src.llm.openai_client import OpenAIClient, OpenAIConfig

llm = OpenAIClient(OpenAIConfig(
    api_key: str,
    model: str = "gpt-4",
    temperature: float = 0.3,
))
```

### AnthropicClient

```python
from src.llm.anthropic_client import AnthropicClient, AnthropicConfig

llm = AnthropicClient(AnthropicConfig(
    api_key: str,
    model: str = "claude-3-opus-20240229",
))
```

---

## Database Adapters

### ConnectionConfig

```python
from src.database.multi_db import ConnectionConfig, DatabaseType

config = ConnectionConfig(
    name: str,                    # Connection name
    db_type: DatabaseType,        # Database type enum
    host: Optional[str] = None,   # Server hostname
    port: Optional[int] = None,   # Server port
    database: str,                # Database name
    username: Optional[str] = None,
    password: Optional[str] = None,
    connection_string: Optional[str] = None,  # Alternative to host/port
)
```

### DatabaseType Enum

```python
from src.database.multi_db import DatabaseType

DatabaseType.MSSQL       # MS SQL Server
DatabaseType.POSTGRESQL  # PostgreSQL
DatabaseType.MYSQL       # MySQL
DatabaseType.SQLITE      # SQLite
DatabaseType.SNOWFLAKE   # Snowflake
DatabaseType.BIGQUERY    # Google BigQuery
```

### MSSQLAdapter

```python
from src.database.multi_db import MSSQLAdapter

db = MSSQLAdapter(config)
await db.connect()
results = await db.execute("SELECT * FROM users")
```

### PostgreSQLAdapter

```python
from src.database.multi_db import PostgreSQLAdapter

db = PostgreSQLAdapter(config)
await db.connect()
results = await db.execute("SELECT * FROM users")
```

### MySQLAdapter

```python
from src.database.multi_db import MySQLAdapter

db = MySQLAdapter(config)
await db.connect()
results = await db.execute("SELECT * FROM users")
```

### SQLiteAdapter

```python
from src.database.multi_db import SQLiteAdapter

db = SQLiteAdapter(ConnectionConfig(
    name="mydb",
    db_type=DatabaseType.SQLITE,
    database="/path/to/database.db",  # or ":memory:"
))
await db.connect()
```

---

## Memory Manager

### MemoryManager

```python
from src.memory.manager import MemoryManager, MemoryConfig

memory = MemoryManager(MemoryConfig(
    enable_graph: bool = True,
    enable_vector: bool = True,
    enable_sql: bool = True,
    vector_dimensions: int = 1536,
    similarity_threshold: float = 0.7,
    max_memories_per_query: int = 10,
))
```

### MemoryType Enum

```python
from src.memory.manager import MemoryType

MemoryType.CONVERSATION     # Chat history
MemoryType.ENTITY_FACT      # Facts about entities
MemoryType.SCHEMA           # Database schema
MemoryType.QUERY_PATTERN    # Successful SQL patterns
MemoryType.ERROR_PATTERN    # Error patterns to avoid
MemoryType.USER_PREFERENCE  # User preferences
MemoryType.SEMANTIC         # General knowledge
```

### Memory Operations

```python
# Add memory
await memory.add(
    content="SELECT * FROM users",
    memory_type=MemoryType.QUERY_PATTERN,
    metadata={"table": "users"},
)

# Search memories
results = await memory.search(
    query="user queries",
    memory_types=[MemoryType.QUERY_PATTERN],
    limit=5,
)

# Full ECL pipeline
memory = await memory.ingest(
    content="...",
    memory_type=MemoryType.SCHEMA,
)
```

---

## API Server

### APIConfig

```python
from src.api.server import APIConfig

config = APIConfig(
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False,
    cors_origins: List[str] = ["*"],
    rate_limit_per_minute: int = 100,
    max_concurrent_requests: int = 50,
    request_timeout_seconds: float = 300.0,
)
```

### create_app()

```python
from src.api.server import create_app

app = create_app(
    config: APIConfig,
    sql_agent: MetaAgent,
    user_resolver: Callable,
)

# Run with: uvicorn main:app
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Execute natural language query |
| `POST` | `/query/stream` | Execute with SSE streaming |
| `GET` | `/health` | Health check |
| `GET` | `/schema` | Get database schema |

---

## Authentication

### JWTUserResolver

```python
from src.api.auth import JWTUserResolver

resolver = JWTUserResolver(
    secret_key: str,
    algorithm: str = "HS256",
)
```

### APIKeyResolver

```python
from src.api.auth import APIKeyResolver

resolver = APIKeyResolver(
    api_keys: Dict[str, str],  # {"key": "user_id"}
)
```

### UserContext

```python
from src.core.base import UserContext

user = UserContext(
    user_id: str,
    roles: List[str] = [],
    permissions: Dict[str, Any] = {},
)
```

---

<p align="center">
  <a href="databases">Database Support →</a>
</p>
