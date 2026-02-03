# Configuration

Environment setup and configuration options.

---

## Environment Variables

Create a `.env` file in your project root:

```bash title=".env"
# LLM Configuration (choose one provider)

# Azure OpenAI
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01

# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Database
DB_HOST=localhost
DB_PORT=1433
DB_NAME=MyDatabase
DB_USER=user
DB_PASSWORD=password

# Or connection string
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb

# API Server
API_HOST=0.0.0.0
API_PORT=8000
JWT_SECRET=your-secret-key

# Memory Storage
VECTOR_STORE_PATH=./data/vector
GRAPH_STORE_PATH=./data/graph

# Logging
LOG_LEVEL=INFO
```

---

## Loading Environment Variables

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Use in code
api_key = os.getenv("AZURE_OPENAI_API_KEY")
```

---

## Knowledge Persistence

All learned knowledge is automatically saved to:

```
~/.vanna/meta_agent.json
```

### Knowledge Structure

```json title="~/.vanna/meta_agent.json"
{
  "successful_solutions": [
    {
      "question": "How many users?",
      "sql": "SELECT COUNT(*) FROM users",
      "problem_type": "counting",
      "was_corrected": false
    }
  ],
  "failed_attempts": [...],
  "dialect_learnings": [
    "DIALECT: mssql - Uses TOP instead of LIMIT"
  ],
  "schema_insights": [
    "- Users table has email, name columns"
  ],
  "name_corrections": {
    "categories": "Category"
  },
  "table_relationships": {
    "orders": "joins with customers"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Custom Storage Path

```python
from pathlib import Path

agent = MetaAgent(
    llm_client=llm,
    storage_path=Path("./my_knowledge.json"),
)
```

---

## LLM Configuration

### Azure OpenAI

```python
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

llm = AzureOpenAIClient(AzureOpenAIConfig(
    api_key="your-key",
    azure_endpoint="https://your-endpoint.openai.azure.com",
    azure_deployment="gpt-4o",
    api_version="2024-02-01",
    temperature=0.3,  # Lower = more deterministic
    max_tokens=2000,
))
```

### OpenAI

```python
from src.llm.openai_client import OpenAIClient, OpenAIConfig

llm = OpenAIClient(OpenAIConfig(
    api_key="sk-...",
    model="gpt-4",
    temperature=0.3,
    max_tokens=2000,
))
```

### Anthropic

```python
from src.llm.anthropic_client import AnthropicClient, AnthropicConfig

llm = AnthropicClient(AnthropicConfig(
    api_key="sk-ant-...",
    model="claude-3-opus-20240229",
    temperature=0.3,
    max_tokens=2000,
))
```

---

## Database Configuration

### Connection Config

```python
from src.database.multi_db import ConnectionConfig, DatabaseType

config = ConnectionConfig(
    name="my_database",        # Identifier
    db_type=DatabaseType.MSSQL,
    host="localhost",
    port=1433,
    database="MyDB",
    username="user",
    password="password",

    # Or use connection string
    connection_string="...",

    # Optional settings
    pool_size=10,
    pool_max_overflow=20,
    connect_timeout=30,
)
```

### Database Types

```python
from src.database.multi_db import DatabaseType

DatabaseType.MSSQL       # MS SQL Server
DatabaseType.POSTGRESQL  # PostgreSQL
DatabaseType.MYSQL       # MySQL
DatabaseType.SQLITE      # SQLite
DatabaseType.SNOWFLAKE   # Snowflake
DatabaseType.BIGQUERY    # Google BigQuery
```

---

## Memory Configuration

```python
from src.memory.manager import MemoryManager, MemoryConfig

memory = MemoryManager(MemoryConfig(
    enable_graph=True,         # Graph relationships
    enable_vector=True,        # Semantic embeddings
    enable_sql=True,           # Structured persistence

    vector_dimensions=1536,    # OpenAI embedding size
    similarity_threshold=0.7,
    max_memories_per_query=10,
    memory_ttl_days=90,

    enable_compression=True,
    enable_deduplication=True,
))
```

---

## API Server Configuration

```python
from src.api.server import APIConfig

config = APIConfig(
    host="0.0.0.0",
    port=8000,
    debug=False,

    # CORS
    cors_origins=["*"],  # Or specific origins

    # Rate limiting
    rate_limit_per_minute=100,
    max_concurrent_requests=50,

    # Timeouts
    request_timeout_seconds=300.0,

    # Documentation
    enable_docs=True,  # Swagger at /docs
)
```

---

## Logging

```python
import logging

# Set log level
logging.basicConfig(level=logging.INFO)

# Or for specific modules
logging.getLogger("src.intelligence").setLevel(logging.DEBUG)
```

### Log Levels

| Level | Use Case |
|-------|----------|
| `DEBUG` | Detailed execution traces |
| `INFO` | Normal operation |
| `WARNING` | Potential issues |
| `ERROR` | Errors that need attention |
