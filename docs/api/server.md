# API Server

Production-ready REST API with FastAPI.

---

## Quick Start

```python
from src.api.server import create_app, APIConfig

app = create_app(
    config=APIConfig(host="0.0.0.0", port=8000),
    sql_agent=agent,
)

# Run: uvicorn main:app
```

---

## APIConfig

```python
from src.api.server import APIConfig

config = APIConfig(
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False,
    cors_origins: List[str] = ["*"],
    enable_docs: bool = True,
    rate_limit_per_minute: int = 100,
    max_concurrent_requests: int = 50,
    request_timeout_seconds: float = 300.0,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"0.0.0.0"` | Server host |
| `port` | `int` | `8000` | Server port |
| `debug` | `bool` | `False` | Debug mode |
| `cors_origins` | `List[str]` | `["*"]` | CORS allowed origins |
| `enable_docs` | `bool` | `True` | Enable Swagger UI |
| `rate_limit_per_minute` | `int` | `100` | Rate limit per user |
| `max_concurrent_requests` | `int` | `50` | Concurrent request limit |
| `request_timeout_seconds` | `float` | `300.0` | Request timeout |

---

## Endpoints

### POST /query

Execute natural language query.

=== "Request"

    ```bash
    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer YOUR_TOKEN" \
      -d '{
        "question": "How many users signed up last month?",
        "include_sql": true,
        "include_explanation": true,
        "max_rows": 100
      }'
    ```

=== "Response"

    ```json
    {
      "success": true,
      "conversation_id": "conv_123",
      "question": "How many users signed up last month?",
      "sql": "SELECT COUNT(*) FROM users WHERE ...",
      "data": [{"count": 1234}],
      "columns": ["count"],
      "row_count": 1,
      "explanation": "This query counts users...",
      "execution_time_ms": 145.5
    }
    ```

---

### POST /query/stream

Execute with Server-Sent Events streaming.

=== "Request"

    ```bash
    curl -N "http://localhost:8000/query/stream?q=Show+sales+trends" \
      -H "Authorization: Bearer YOUR_TOKEN"
    ```

=== "Response (SSE)"

    ```
    event: thinking
    data: {"step": "Analyzing question..."}

    event: sql
    data: {"sql": "SELECT month, SUM(amount) FROM sales GROUP BY month"}

    event: data
    data: {"rows": [{"month": "Jan", "total": 10000}]}

    event: explanation
    data: {"text": "Sales show an upward trend..."}

    event: done
    data: {"success": true, "execution_time_ms": 234}
    ```

---

### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "version": "2.0.0"
}
```

---

### GET /schema

Get database schema.

```bash
curl http://localhost:8000/schema \
  -H "Authorization: Bearer YOUR_TOKEN"
```

```json
{
  "tables": [
    {
      "name": "users",
      "columns": ["id", "name", "email"]
    }
  ]
}
```

---

## Authentication

### JWTUserResolver

```python
from src.api.auth import JWTUserResolver

resolver = JWTUserResolver(
    secret_key: str,
    algorithm: str = "HS256",
)

app = create_app(
    config=config,
    sql_agent=agent,
    user_resolver=resolver.resolve,
)
```

JWT payload:

```json
{
  "sub": "user123",
  "roles": ["analyst", "viewer"],
  "exp": 1234567890
}
```

### APIKeyResolver

```python
from src.api.auth import APIKeyResolver

resolver = APIKeyResolver(
    api_keys={
        "key_abc123": "user1",
        "key_def456": "user2",
    },
)
```

Usage: `Authorization: Bearer key_abc123`

---

## SSE Client (JavaScript)

```javascript
const eventSource = new EventSource(
  '/query/stream?q=' + encodeURIComponent(question)
);

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

eventSource.addEventListener('error', (e) => {
  showError(e.data);
  eventSource.close();
});
```

---

## Complete Example

```python
import os
import asyncio
from src.api.server import create_app, APIConfig
from src.api.auth import JWTUserResolver
from src.intelligence.meta_agent import MetaAgent
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
from src.database.multi_db import PostgreSQLAdapter, ConnectionConfig, DatabaseType

async def create_agent():
    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4o",
    ))

    db = PostgreSQLAdapter(ConnectionConfig(
        name="production",
        db_type=DatabaseType.POSTGRESQL,
        connection_string=os.getenv("DATABASE_URL"),
    ))
    await db.connect()

    agent = MetaAgent(llm_client=llm)
    await agent.connect(db_executor=db.execute)
    await agent.auto_learn(intensity="light")

    return agent

agent = asyncio.run(create_agent())

config = APIConfig(
    host="0.0.0.0",
    port=int(os.getenv("PORT", 8000)),
    cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
)

resolver = JWTUserResolver(secret_key=os.getenv("JWT_SECRET"))

app = create_app(
    config=config,
    sql_agent=agent,
    user_resolver=resolver.resolve,
)

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## Docker Deployment

=== "Dockerfile"

    ```dockerfile
    FROM python:3.11-slim

    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    COPY . .
    EXPOSE 8000

    CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
    ```

=== "docker-compose.yml"

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
