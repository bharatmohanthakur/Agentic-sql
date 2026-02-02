"""
Tool Registry System - Enterprise-grade tool management
Implements structured outputs (Grammar pattern) and permission-aware execution
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union, get_type_hints

from pydantic import BaseModel, Field, create_model, validator

from .base import UserContext

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Categories for tool organization"""
    DATABASE = "database"
    MEMORY = "memory"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    RETRIEVAL = "retrieval"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    EXTERNAL = "external"


class PermissionLevel(str, Enum):
    """Tool permission levels"""
    PUBLIC = "public"  # Anyone can use
    AUTHENTICATED = "authenticated"  # Logged in users
    ELEVATED = "elevated"  # Users with specific role
    ADMIN = "admin"  # Admin only


class ToolSchema(BaseModel):
    """Schema definition for tool arguments - enables Grammar pattern"""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    returns: Optional[Dict[str, Any]] = None

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool use format"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }


@dataclass
class ToolResult:
    """Standardized tool execution result"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """
    Base Tool class with enterprise features:
    - Typed arguments via Pydantic
    - Permission checking
    - User context injection
    - Audit logging
    - Rate limiting support
    """

    name: str = "base_tool"
    description: str = "Base tool"
    category: ToolCategory = ToolCategory.EXTERNAL
    permission_level: PermissionLevel = PermissionLevel.PUBLIC
    required_roles: List[str] = []
    rate_limit: Optional[int] = None  # calls per minute

    def __init__(self):
        self._call_count: Dict[str, int] = {}  # user_id -> count
        self._call_timestamps: Dict[str, List[datetime]] = {}

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Define the tool's input/output schema"""
        pass

    @abstractmethod
    async def execute(
        self,
        user_context: UserContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute the tool with user context"""
        pass

    def check_permission(self, user_context: UserContext) -> bool:
        """Check if user has permission to use this tool"""
        if self.permission_level == PermissionLevel.PUBLIC:
            return True

        if self.permission_level == PermissionLevel.AUTHENTICATED:
            return user_context is not None

        if self.permission_level == PermissionLevel.ELEVATED:
            return any(role in user_context.roles for role in self.required_roles)

        if self.permission_level == PermissionLevel.ADMIN:
            return "admin" in user_context.roles

        return False

    def check_rate_limit(self, user_context: UserContext) -> bool:
        """Check if user is within rate limits"""
        if self.rate_limit is None:
            return True

        user_id = user_context.user_id
        now = datetime.utcnow()

        # Clean old timestamps
        if user_id in self._call_timestamps:
            self._call_timestamps[user_id] = [
                ts for ts in self._call_timestamps[user_id]
                if (now - ts).total_seconds() < 60
            ]

        # Check limit
        current_count = len(self._call_timestamps.get(user_id, []))
        return current_count < self.rate_limit

    async def __call__(
        self,
        user_context: UserContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute with permission and rate limit checks"""
        start_time = datetime.utcnow()

        # Permission check
        if not self.check_permission(user_context):
            return ToolResult(
                success=False,
                error=f"Permission denied for tool: {self.name}",
            )

        # Rate limit check
        if not self.check_rate_limit(user_context):
            return ToolResult(
                success=False,
                error=f"Rate limit exceeded for tool: {self.name}",
            )

        # Track call
        if user_context.user_id not in self._call_timestamps:
            self._call_timestamps[user_context.user_id] = []
        self._call_timestamps[user_context.user_id].append(start_time)

        try:
            result = await self.execute(user_context, **kwargs)
            end_time = datetime.utcnow()
            result.execution_time_ms = (end_time - start_time).total_seconds() * 1000

            # Audit log
            logger.info(
                f"Tool executed: {self.name} | User: {user_context.user_id} | "
                f"Success: {result.success} | Time: {result.execution_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.exception(f"Tool execution failed: {self.name}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )


class ToolRegistry:
    """
    Central registry for all tools
    Provides discovery, validation, and execution management
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
        self._hooks: Dict[str, List[Callable]] = {
            "pre_execute": [],
            "post_execute": [],
        }

    def register(self, tool: Tool) -> None:
        """Register a tool"""
        self._tools[tool.name] = tool
        self._categories[tool.category].append(tool.name)
        logger.info(f"Registered tool: {tool.name} [{tool.category.value}]")

    def unregister(self, tool_name: str) -> None:
        """Unregister a tool"""
        if tool_name in self._tools:
            tool = self._tools[tool_name]
            self._categories[tool.category].remove(tool_name)
            del self._tools[tool_name]

    def get(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(tool_name)

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        user_context: Optional[UserContext] = None,
    ) -> List[Tool]:
        """List available tools, optionally filtered by category and permissions"""
        if category:
            tool_names = self._categories.get(category, [])
        else:
            tool_names = list(self._tools.keys())

        tools = [self._tools[name] for name in tool_names]

        # Filter by permissions if user context provided
        if user_context:
            tools = [t for t in tools if t.check_permission(user_context)]

        return tools

    def get_schemas(
        self,
        user_context: Optional[UserContext] = None,
        format: str = "openai",
    ) -> List[Dict[str, Any]]:
        """Get schemas for all available tools in specified format"""
        tools = self.list_tools(user_context=user_context)
        schemas = []

        for tool in tools:
            schema = tool.get_schema()
            if format == "openai":
                schemas.append(schema.to_openai_format())
            elif format == "anthropic":
                schemas.append(schema.to_anthropic_format())
            else:
                schemas.append(schema.dict())

        return schemas

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register execution hook"""
        if event in self._hooks:
            self._hooks[event].append(callback)

    async def execute(
        self,
        tool_name: str,
        user_context: UserContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a tool by name with hooks"""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

        # Pre-execution hooks
        for hook in self._hooks["pre_execute"]:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(tool, user_context, kwargs)
                else:
                    hook(tool, user_context, kwargs)
            except Exception as e:
                logger.warning(f"Pre-execute hook failed: {e}")

        # Execute
        result = await tool(user_context, **kwargs)

        # Post-execution hooks
        for hook in self._hooks["post_execute"]:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(tool, user_context, result)
                else:
                    hook(tool, user_context, result)
            except Exception as e:
                logger.warning(f"Post-execute hook failed: {e}")

        return result


def tool(
    name: str,
    description: str,
    category: ToolCategory = ToolCategory.EXTERNAL,
    permission_level: PermissionLevel = PermissionLevel.PUBLIC,
    required_roles: Optional[List[str]] = None,
    rate_limit: Optional[int] = None,
):
    """
    Decorator to create a tool from a function
    Automatically extracts schema from type hints
    """
    def decorator(func: Callable) -> Tool:
        # Extract schema from function signature
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}

        parameters = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'user_context'):
                continue

            param_type = hints.get(param_name, Any)
            param_schema = _type_to_json_schema(param_type)
            parameters[param_name] = param_schema

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        schema = ToolSchema(
            name=name,
            description=description,
            parameters=parameters,
            required=required,
        )

        class FunctionTool(Tool):
            def __init__(self):
                super().__init__()
                self.name = name
                self.description = description
                self.category = category
                self.permission_level = permission_level
                self.required_roles = required_roles or []
                self.rate_limit = rate_limit
                self._func = func
                self._schema = schema

            def get_schema(self) -> ToolSchema:
                return self._schema

            async def execute(
                self,
                user_context: UserContext,
                **kwargs: Any,
            ) -> ToolResult:
                try:
                    if asyncio.iscoroutinefunction(self._func):
                        result = await self._func(user_context=user_context, **kwargs)
                    else:
                        result = self._func(user_context=user_context, **kwargs)

                    return ToolResult(success=True, data=result)
                except Exception as e:
                    return ToolResult(success=False, error=str(e))

        return FunctionTool()

    return decorator


def _type_to_json_schema(python_type: Any) -> Dict[str, Any]:
    """Convert Python type hint to JSON schema"""
    if python_type == str:
        return {"type": "string"}
    elif python_type == int:
        return {"type": "integer"}
    elif python_type == float:
        return {"type": "number"}
    elif python_type == bool:
        return {"type": "boolean"}
    elif python_type == list or getattr(python_type, '__origin__', None) == list:
        return {"type": "array"}
    elif python_type == dict or getattr(python_type, '__origin__', None) == dict:
        return {"type": "object"}
    else:
        return {"type": "string"}
