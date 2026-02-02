"""
Agentic Text-to-SQL Framework

A best-in-class implementation combining:
- Vanna 2.0's user-aware text-to-SQL pipeline
- Cognee's graph+vector hybrid memory (ECL pattern)
- Memori's SQL-native memory persistence
- Modern agentic design patterns (ReAct, Reflection, Multi-Agent, Planning)

Usage:
    from agentic_sql import SQLAgent, create_app, MemoryManager

    # Create memory manager
    memory = MemoryManager(MemoryConfig())

    # Create SQL agent
    agent = SQLAgent(
        config=SQLAgentConfig(),
        tool_registry=registry,
        memory=memory,
        llm_client=llm,
        db_executor=db.execute,
    )

    # Run query
    result = await agent.execute("What were total sales last month?")
"""
from .core.base import (
    AgentConfig,
    AgentContext,
    AgentResult,
    AgentState,
    BaseAgent,
    UserContext,
)
from .core.registry import Tool, ToolRegistry, ToolResult, ToolSchema
from .core.orchestrator import AgentOrchestrator, PipelineBuilder, Workflow

from .agents.sql_agent import SQLAgent, SQLAgentConfig
from .memory.manager import MemoryManager, MemoryConfig, MemoryType

from .llm.base import LLMClient, LLMConfig, LLMResponse
from .llm.openai_client import OpenAIClient
from .llm.anthropic_client import AnthropicClient

from .api.server import create_app, APIConfig
from .api.auth import UserResolver, JWTUserResolver

__version__ = "2.0.0"

__all__ = [
    # Core
    "AgentConfig",
    "AgentContext",
    "AgentResult",
    "AgentState",
    "BaseAgent",
    "UserContext",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
    "AgentOrchestrator",
    "PipelineBuilder",
    "Workflow",
    # Agents
    "SQLAgent",
    "SQLAgentConfig",
    # Memory
    "MemoryManager",
    "MemoryConfig",
    "MemoryType",
    # LLM
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "OpenAIClient",
    "AnthropicClient",
    # API
    "create_app",
    "APIConfig",
    "UserResolver",
    "JWTUserResolver",
]
