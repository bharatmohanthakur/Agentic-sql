"""
FastAPI Server with SSE Streaming
Enterprise-ready API layer for the agentic text-to-SQL system
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class APIConfig(BaseModel):
    """API Server Configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    enable_docs: bool = True
    rate_limit_per_minute: int = 100
    max_concurrent_requests: int = 50
    request_timeout_seconds: float = 300.0


class QueryRequest(BaseModel):
    """Request model for natural language queries"""
    question: str
    conversation_id: Optional[str] = None
    include_sql: bool = True
    include_visualization: bool = True
    include_explanation: bool = True
    max_rows: int = 1000


class StreamEvent(BaseModel):
    """Server-Sent Event model"""
    event: str  # thinking, sql, data, chart, explanation, error, done
    data: Any
    id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_sse(self) -> str:
        """Convert to SSE format"""
        lines = []
        if self.id:
            lines.append(f"id: {self.id}")
        lines.append(f"event: {self.event}")
        lines.append(f"data: {json.dumps(self.data)}")
        lines.append("")
        return "\n".join(lines) + "\n"


class QueryResponse(BaseModel):
    """Response model for queries"""
    success: bool
    conversation_id: str
    question: str
    sql: Optional[str] = None
    data: Optional[List[Dict]] = None
    columns: Optional[List[str]] = None
    row_count: int = 0
    explanation: Optional[str] = None
    visualization: Optional[Dict] = None
    execution_time_ms: float = 0.0
    error: Optional[str] = None


def create_app(
    config: APIConfig,
    sql_agent: Any,
    user_resolver: Callable,
) -> Any:
    """
    Create FastAPI application

    Args:
        config: API configuration
        sql_agent: SQLAgent instance
        user_resolver: Function to resolve user from request

    Returns:
        FastAPI application instance
    """
    try:
        from fastapi import FastAPI, Request, HTTPException, Depends
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse, JSONResponse
        from sse_starlette.sse import EventSourceResponse
    except ImportError:
        raise ImportError(
            "FastAPI dependencies required: "
            "pip install fastapi uvicorn sse-starlette"
        )

    # Lifespan context manager
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting Agentic SQL API server")
        yield
        logger.info("Shutting down server")

    app = FastAPI(
        title="Agentic Text-to-SQL API",
        description="Enterprise-grade natural language to SQL conversion",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if config.enable_docs else None,
        redoc_url="/redoc" if config.enable_docs else None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request counting for rate limiting
    request_counts: Dict[str, List[datetime]] = {}

    async def check_rate_limit(request: Request):
        """Rate limiting middleware"""
        client_ip = request.client.host if request.client else "unknown"
        now = datetime.utcnow()

        if client_ip not in request_counts:
            request_counts[client_ip] = []

        # Clean old entries
        request_counts[client_ip] = [
            ts for ts in request_counts[client_ip]
            if (now - ts).total_seconds() < 60
        ]

        if len(request_counts[client_ip]) >= config.rate_limit_per_minute:
            raise HTTPException(429, "Rate limit exceeded")

        request_counts[client_ip].append(now)

    async def get_user(request: Request):
        """Dependency to resolve user from request"""
        return await user_resolver(request)

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

    @app.post("/query", response_model=QueryResponse)
    async def query(
        request: QueryRequest,
        user=Depends(get_user),
        _=Depends(check_rate_limit),
    ):
        """
        Execute a natural language query

        Returns the complete result as JSON
        """
        from ..core.base import UserContext

        conversation_id = request.conversation_id or str(uuid4())

        try:
            result = await sql_agent.execute(
                request.question,
                user_context=user,
            )

            if not result.success:
                return QueryResponse(
                    success=False,
                    conversation_id=conversation_id,
                    question=request.question,
                    error=result.error,
                )

            data = result.data or {}

            return QueryResponse(
                success=True,
                conversation_id=conversation_id,
                question=request.question,
                sql=data.get("sql") if request.include_sql else None,
                data=data.get("data"),
                columns=data.get("columns"),
                row_count=data.get("row_count", 0),
                explanation=data.get("explanation") if request.include_explanation else None,
                execution_time_ms=data.get("execution_time_ms", 0),
            )

        except Exception as e:
            logger.exception("Query failed")
            return QueryResponse(
                success=False,
                conversation_id=conversation_id,
                question=request.question,
                error=str(e),
            )

    @app.post("/query/stream")
    async def query_stream(
        request: QueryRequest,
        user=Depends(get_user),
        _=Depends(check_rate_limit),
    ):
        """
        Execute a query with streaming progress updates

        Uses Server-Sent Events (SSE) to stream:
        - thinking: Agent reasoning steps
        - sql: Generated SQL query
        - data: Query results
        - chart: Visualization data
        - explanation: Natural language explanation
        - error: Any errors
        - done: Completion signal
        """
        conversation_id = request.conversation_id or str(uuid4())

        async def event_generator() -> AsyncIterator[str]:
            try:
                # Emit thinking event
                yield StreamEvent(
                    event="thinking",
                    data={"message": "Analyzing your question..."},
                    id=str(uuid4()),
                ).to_sse()

                # Execute agent
                result = await sql_agent.execute(
                    request.question,
                    user_context=user,
                )

                if not result.success:
                    yield StreamEvent(
                        event="error",
                        data={"message": result.error},
                        id=str(uuid4()),
                    ).to_sse()
                else:
                    data = result.data or {}

                    # Emit SQL
                    if request.include_sql and data.get("sql"):
                        yield StreamEvent(
                            event="sql",
                            data={"sql": data["sql"]},
                            id=str(uuid4()),
                        ).to_sse()

                    # Emit data
                    if data.get("data"):
                        yield StreamEvent(
                            event="data",
                            data={
                                "data": data["data"][:request.max_rows],
                                "columns": data.get("columns", []),
                                "row_count": data.get("row_count", 0),
                                "truncated": data.get("row_count", 0) > request.max_rows,
                            },
                            id=str(uuid4()),
                        ).to_sse()

                    # Emit explanation
                    if request.include_explanation and data.get("explanation"):
                        yield StreamEvent(
                            event="explanation",
                            data={"text": data["explanation"]},
                            id=str(uuid4()),
                        ).to_sse()

                # Emit done
                yield StreamEvent(
                    event="done",
                    data={"conversation_id": conversation_id},
                    id=str(uuid4()),
                ).to_sse()

            except Exception as e:
                logger.exception("Stream query failed")
                yield StreamEvent(
                    event="error",
                    data={"message": str(e)},
                    id=str(uuid4()),
                ).to_sse()

        return EventSourceResponse(event_generator())

    @app.get("/schema")
    async def get_schema(
        user=Depends(get_user),
        _=Depends(check_rate_limit),
    ):
        """Get database schema information"""
        result = await sql_agent.tool_registry.execute(
            "get_schema",
            user,
        )

        if not result.success:
            raise HTTPException(500, result.error)

        return result.data

    @app.get("/conversations/{conversation_id}")
    async def get_conversation(
        conversation_id: str,
        user=Depends(get_user),
    ):
        """Get conversation history"""
        # Implement conversation retrieval from memory
        return {"conversation_id": conversation_id, "messages": []}

    return app


def run_server(
    app: Any,
    config: APIConfig,
):
    """Run the server using uvicorn"""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn required: pip install uvicorn")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="debug" if config.debug else "info",
    )
