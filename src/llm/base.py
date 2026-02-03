"""
LLM Base Interface - Multi-provider support with middleware
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"


class MessageRole(str, Enum):
    """Message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """Chat message"""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


class ToolDefinition(BaseModel):
    """Tool definition for function calling"""
    name: str
    description: str
    parameters: Dict[str, Any]


class LLMConfig(BaseModel):
    """Configuration for LLM client"""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout_seconds: float = 60.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0

    # Middleware settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_logging: bool = True
    enable_cost_tracking: bool = True


class TokenUsage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Response from LLM"""
    content: str = ""
    tool_calls: Optional[List[Dict]] = None
    finish_reason: str = "stop"
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    latency_ms: float = 0.0
    cached: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMMiddleware(ABC):
    """Base middleware for LLM requests"""

    @abstractmethod
    async def before_request(
        self,
        messages: List[Message],
        config: LLMConfig,
    ) -> tuple[List[Message], LLMConfig]:
        pass

    @abstractmethod
    async def after_response(
        self,
        response: LLMResponse,
        messages: List[Message],
        config: LLMConfig,
    ) -> LLMResponse:
        pass


class CacheMiddleware(LLMMiddleware):
    """Caching middleware for LLM responses"""

    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, tuple[LLMResponse, float]] = {}
        self.ttl = ttl_seconds

    def _cache_key(self, messages: List[Message], config: LLMConfig) -> str:
        """Generate cache key from messages and config"""
        content = json.dumps(
            {
                "messages": [m.dict() for m in messages],
                "model": config.model,
                "temperature": config.temperature,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    async def before_request(
        self,
        messages: List[Message],
        config: LLMConfig,
    ) -> tuple[List[Message], LLMConfig]:
        return messages, config

    async def after_response(
        self,
        response: LLMResponse,
        messages: List[Message],
        config: LLMConfig,
    ) -> LLMResponse:
        if config.enable_caching:
            key = self._cache_key(messages, config)
            self.cache[key] = (response, time.time())
        return response

    def get_cached(
        self,
        messages: List[Message],
        config: LLMConfig,
    ) -> Optional[LLMResponse]:
        """Get cached response if available"""
        if not config.enable_caching:
            return None

        key = self._cache_key(messages, config)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                response.cached = True
                return response
            else:
                del self.cache[key]
        return None


class CostTrackingMiddleware(LLMMiddleware):
    """Track LLM costs"""

    # Approximate costs per 1K tokens (as of 2024)
    COSTS = {
        # OpenAI models
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        # Anthropic direct API
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku": {"input": 0.0008, "output": 0.004},
        # AWS Bedrock Claude models (same pricing structure)
        "anthropic.claude-3-opus-20240229-v1:0": {"input": 0.015, "output": 0.075},
        "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-5-haiku-20241022-v1:0": {"input": 0.0008, "output": 0.004},
    }

    def __init__(self):
        self.total_cost = 0.0
        self.request_count = 0
        self.costs_by_model: Dict[str, float] = {}

    async def before_request(
        self,
        messages: List[Message],
        config: LLMConfig,
    ) -> tuple[List[Message], LLMConfig]:
        return messages, config

    async def after_response(
        self,
        response: LLMResponse,
        messages: List[Message],
        config: LLMConfig,
    ) -> LLMResponse:
        if not config.enable_cost_tracking:
            return response

        model = config.model
        costs = self.COSTS.get(model, {"input": 0.01, "output": 0.03})

        input_cost = (response.usage.prompt_tokens / 1000) * costs["input"]
        output_cost = (response.usage.completion_tokens / 1000) * costs["output"]
        total = input_cost + output_cost

        self.total_cost += total
        self.request_count += 1
        self.costs_by_model[model] = self.costs_by_model.get(model, 0) + total

        response.metadata["cost_usd"] = total

        return response


class LoggingMiddleware(LLMMiddleware):
    """Logging middleware"""

    async def before_request(
        self,
        messages: List[Message],
        config: LLMConfig,
    ) -> tuple[List[Message], LLMConfig]:
        logger.debug(
            f"LLM Request | Model: {config.model} | "
            f"Messages: {len(messages)} | Temp: {config.temperature}"
        )
        return messages, config

    async def after_response(
        self,
        response: LLMResponse,
        messages: List[Message],
        config: LLMConfig,
    ) -> LLMResponse:
        logger.info(
            f"LLM Response | Model: {response.model} | "
            f"Tokens: {response.usage.total_tokens} | "
            f"Latency: {response.latency_ms:.2f}ms | "
            f"Cached: {response.cached}"
        )
        return response


class LLMClient(ABC):
    """
    Abstract LLM Client with middleware support

    Features:
    - Multi-provider support
    - Request/response middleware
    - Caching
    - Cost tracking
    - Retry logic
    - Streaming support
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.middleware: List[LLMMiddleware] = []

        # Add default middleware
        if config.enable_caching:
            self._cache_middleware = CacheMiddleware(config.cache_ttl_seconds)
            self.middleware.append(self._cache_middleware)
        else:
            self._cache_middleware = None

        if config.enable_cost_tracking:
            self.cost_tracker = CostTrackingMiddleware()
            self.middleware.append(self.cost_tracker)
        else:
            self.cost_tracker = None

        if config.enable_logging:
            self.middleware.append(LoggingMiddleware())

    def add_middleware(self, middleware: LLMMiddleware) -> None:
        """Add custom middleware"""
        self.middleware.append(middleware)

    @abstractmethod
    async def _generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Provider-specific generation logic"""
        pass

    @abstractmethod
    async def _stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Provider-specific streaming logic"""
        pass

    async def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDefinition]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response from the LLM

        Args:
            prompt: Simple text prompt (converted to user message)
            messages: Full message history
            system: System message
            tools: Available tools for function calling
            temperature: Override temperature
            max_tokens: Override max tokens
        """
        # Build messages
        if messages is None:
            messages = []

        if system:
            messages.insert(0, Message(role=MessageRole.SYSTEM, content=system))

        if prompt:
            messages.append(Message(role=MessageRole.USER, content=prompt))

        # Apply config overrides
        config = self.config.copy()
        if temperature is not None:
            config.temperature = temperature
        if max_tokens is not None:
            config.max_tokens = max_tokens

        # Check cache
        if self._cache_middleware:
            cached = self._cache_middleware.get_cached(messages, config)
            if cached:
                return cached.content

        # Run before middleware
        for mw in self.middleware:
            messages, config = await mw.before_request(messages, config)

        # Generate with retry
        response = await self._generate_with_retry(messages, tools, config, **kwargs)

        # Run after middleware
        for mw in self.middleware:
            response = await mw.after_response(response, messages, config)

        return response.content

    async def _generate_with_retry(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]],
        config: LLMConfig,
        **kwargs,
    ) -> LLMResponse:
        """Generate with retry logic"""
        last_error = None

        for attempt in range(config.retry_attempts):
            try:
                start_time = time.time()
                response = await asyncio.wait_for(
                    self._generate(messages, tools, **kwargs),
                    timeout=config.timeout_seconds,
                )
                response.latency_ms = (time.time() - start_time) * 1000
                return response

            except asyncio.TimeoutError:
                last_error = "Request timeout"
                logger.warning(f"LLM timeout (attempt {attempt + 1})")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM error (attempt {attempt + 1}): {e}")

            if attempt < config.retry_attempts - 1:
                await asyncio.sleep(config.retry_delay_seconds * (attempt + 1))

        raise Exception(f"LLM generation failed after {config.retry_attempts} attempts: {last_error}")

    async def stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response tokens"""
        if messages is None:
            messages = []

        if system:
            messages.insert(0, Message(role=MessageRole.SYSTEM, content=system))

        if prompt:
            messages.append(Message(role=MessageRole.USER, content=prompt))

        async for token in self._stream(messages, **kwargs):
            yield token

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Full chat completion with tool support"""
        config = self.config

        # Check cache
        if self._cache_middleware:
            cached = self._cache_middleware.get_cached(messages, config)
            if cached:
                return cached

        response = await self._generate_with_retry(messages, tools, config, **kwargs)

        # Run after middleware
        for mw in self.middleware:
            response = await mw.after_response(response, messages, config)

        return response
