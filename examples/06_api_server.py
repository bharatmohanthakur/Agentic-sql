#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 6: API SERVER - Production-Ready REST API
=============================================================================

This tutorial shows how to create a production REST API with:
- FastAPI server
- Server-Sent Events (SSE) for streaming
- JWT authentication
- Rate limiting
- CORS configuration

Run: python examples/06_api_server.py
Then: curl http://localhost:8000/health
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def main():
    print("=" * 60)
    print("  TUTORIAL 6: API SERVER")
    print("=" * 60)

    # =========================================================================
    # BASIC SERVER SETUP
    # =========================================================================
    print("\n[Step 1] Basic Server Setup")

    print("""
    ```python
    from api.server import create_app, APIConfig
    from api.auth import JWTUserResolver

    # Configure API
    config = APIConfig(
        host="0.0.0.0",
        port=8000,
        debug=False,
        cors_origins=["*"],               # Or specific origins
        enable_docs=True,                 # Swagger UI at /docs
        rate_limit_per_minute=100,        # Rate limiting
        max_concurrent_requests=50,       # Concurrency limit
        request_timeout_seconds=300.0,    # 5 min timeout
    )

    # Setup authentication
    resolver = JWTUserResolver(
        secret_key=os.getenv("JWT_SECRET", "your-secret-key"),
        algorithm="HS256",
    )

    # Create FastAPI app
    app = create_app(
        config=config,
        sql_agent=your_agent,
        user_resolver=resolver.resolve,
    )

    # Run with uvicorn
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host=config.host, port=config.port)
    ```
    """)

    # =========================================================================
    # ENDPOINTS
    # =========================================================================
    print("[Step 2] Available Endpoints")

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                      API ENDPOINTS                           │
    ├──────────┬─────────────────────┬────────────────────────────┤
    │ Method   │ Endpoint            │ Description                │
    ├──────────┼─────────────────────┼────────────────────────────┤
    │ GET      │ /health             │ Health check               │
    │ GET      │ /schema             │ Get database schema        │
    │ POST     │ /query              │ Execute query              │
    │ POST     │ /query/stream       │ Execute with SSE stream    │
    │ GET      │ /docs               │ Swagger documentation      │
    │ GET      │ /redoc              │ ReDoc documentation        │
    └──────────┴─────────────────────┴────────────────────────────┘
    """)

    # =========================================================================
    # QUERY ENDPOINT
    # =========================================================================
    print("[Step 3] Query Endpoint")

    print("""
    Request:
    ```bash
    curl -X POST http://localhost:8000/query \\
      -H "Content-Type: application/json" \\
      -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
      -d '{
        "question": "How many users signed up last month?",
        "include_sql": true,
        "include_explanation": true,
        "max_rows": 100
      }'
    ```

    Response:
    ```json
    {
      "success": true,
      "conversation_id": "conv_123",
      "question": "How many users signed up last month?",
      "sql": "SELECT COUNT(*) FROM users WHERE created_at >= ...",
      "data": [{"count": 1234}],
      "columns": ["count"],
      "row_count": 1,
      "explanation": "This query counts users created in the last month...",
      "execution_time_ms": 145.5
    }
    ```
    """)

    # =========================================================================
    # STREAMING ENDPOINT
    # =========================================================================
    print("[Step 4] Streaming with Server-Sent Events")

    print("""
    The /query/stream endpoint uses SSE for real-time updates:

    Request:
    ```bash
    curl -N "http://localhost:8000/query/stream?q=Show+sales+trends" \\
      -H "Authorization: Bearer YOUR_JWT_TOKEN"
    ```

    Response (SSE stream):
    ```
    event: thinking
    data: {"step": "Analyzing question..."}

    event: sql
    data: {"sql": "SELECT month, SUM(amount) FROM sales GROUP BY month"}

    event: data
    data: {"rows": [{"month": "Jan", "total": 10000}, ...]}

    event: explanation
    data: {"text": "Sales show an upward trend..."}

    event: done
    data: {"success": true, "execution_time_ms": 234}
    ```

    JavaScript client:
    ```javascript
    const eventSource = new EventSource(
      '/query/stream?q=' + encodeURIComponent(question)
    );

    eventSource.addEventListener('thinking', (e) => {
      showThinking(JSON.parse(e.data));
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
    """)

    # =========================================================================
    # AUTHENTICATION
    # =========================================================================
    print("[Step 5] Authentication Options")

    print("""
    Option A: JWT Authentication
    ```python
    from api.auth import JWTUserResolver

    resolver = JWTUserResolver(
        secret_key="your-secret-key",
        algorithm="HS256",
    )

    # JWT payload should include:
    # {
    #   "sub": "user123",
    #   "roles": ["analyst", "viewer"],
    #   "exp": 1234567890
    # }
    ```

    Option B: API Key Authentication
    ```python
    from api.auth import APIKeyResolver

    resolver = APIKeyResolver(
        api_keys={
            "key_abc123": "user1",
            "key_def456": "user2",
        },
    )

    # Usage: Authorization: Bearer key_abc123
    ```

    Option C: Custom Authentication
    ```python
    from api.auth import UserResolver
    from core.base import UserContext

    class MyCustomResolver(UserResolver):
        async def resolve(self, request) -> UserContext:
            # Your custom logic here
            token = request.headers.get("X-Custom-Token")
            user_info = await validate_token(token)

            return UserContext(
                user_id=user_info["id"],
                roles=user_info["roles"],
            )

        async def get_permissions(self, user_id: str) -> dict:
            return await fetch_permissions(user_id)
    ```
    """)

    # =========================================================================
    # CORS CONFIGURATION
    # =========================================================================
    print("[Step 6] CORS Configuration")

    print("""
    For web applications, configure CORS:

    ```python
    config = APIConfig(
        # Allow all origins (development)
        cors_origins=["*"],

        # Or specific origins (production)
        cors_origins=[
            "https://myapp.com",
            "https://admin.myapp.com",
        ],
    )
    ```

    The server automatically adds CORS headers for:
    - Access-Control-Allow-Origin
    - Access-Control-Allow-Methods
    - Access-Control-Allow-Headers
    - Access-Control-Allow-Credentials
    """)

    # =========================================================================
    # RATE LIMITING
    # =========================================================================
    print("[Step 7] Rate Limiting")

    print("""
    Built-in rate limiting per user:

    ```python
    config = APIConfig(
        rate_limit_per_minute=100,    # 100 requests/min per user
        max_concurrent_requests=50,   # 50 concurrent requests total
    )
    ```

    When rate limited, response:
    ```json
    {
      "error": "Rate limit exceeded",
      "retry_after": 30
    }
    ```
    """)

    # =========================================================================
    # FULL EXAMPLE
    # =========================================================================
    print("\n[Full Example] Complete Server Setup")

    print("""
    ```python
    # server.py
    import os
    import asyncio
    from api.server import create_app, APIConfig
    from api.auth import JWTUserResolver
    from intelligence.meta_agent import MetaAgent
    from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
    from database.multi_db import PostgreSQLAdapter, ConnectionConfig, DatabaseType

    async def create_agent():
        # Setup LLM
        llm = AzureOpenAIClient(AzureOpenAIConfig(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment="gpt-4o",
        ))

        # Setup database
        db = PostgreSQLAdapter(ConnectionConfig(
            name="production",
            db_type=DatabaseType.POSTGRESQL,
            connection_string=os.getenv("DATABASE_URL"),
        ))
        await db.connect()

        # Create and connect agent
        agent = MetaAgent(llm_client=llm)
        await agent.connect(db_executor=db.execute)

        # Optional: auto-train
        await agent.auto_learn(intensity="light")

        return agent

    # Create app
    agent = asyncio.run(create_agent())

    config = APIConfig(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        rate_limit_per_minute=100,
    )

    resolver = JWTUserResolver(
        secret_key=os.getenv("JWT_SECRET"),
    )

    app = create_app(
        config=config,
        sql_agent=agent,
        user_resolver=resolver.resolve,
    )

    # Run with: uvicorn server:app --host 0.0.0.0 --port 8000
    ```
    """)

    # =========================================================================
    # DOCKER DEPLOYMENT
    # =========================================================================
    print("[Bonus] Docker Deployment")

    print("""
    Dockerfile:
    ```dockerfile
    FROM python:3.11-slim

    WORKDIR /app

    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    COPY . .

    EXPOSE 8000

    CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
    ```

    docker-compose.yml:
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
    """)

    print("\n" + "=" * 60)
    print("  TUTORIAL COMPLETE!")
    print("=" * 60)
    print("\nTo run a real server, check the example code above!")


if __name__ == "__main__":
    main()
