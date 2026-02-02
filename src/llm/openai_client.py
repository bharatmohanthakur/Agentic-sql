"""
OpenAI LLM Client Implementation
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import (
    LLMClient,
    LLMConfig,
    LLMResponse,
    Message,
    MessageRole,
    TokenUsage,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """
    OpenAI API Client with full feature support

    Supports:
    - GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
    - Function calling
    - Streaming
    - Vision (for multimodal models)
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    async def _get_client(self):
        """Lazy load OpenAI client"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai")

        return self._client

    def _convert_messages(self, messages: List[Message]) -> List[Dict]:
        """Convert internal messages to OpenAI format"""
        converted = []
        for msg in messages:
            m = {
                "role": msg.role.value,
                "content": msg.content,
            }
            if msg.name:
                m["name"] = msg.name
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            converted.append(m)
        return converted

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict]:
        """Convert tools to OpenAI format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    async def _generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using OpenAI API"""
        client = await self._get_client()

        request_params = {
            "model": self.config.model,
            "messages": self._convert_messages(messages),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        if tools:
            request_params["tools"] = self._convert_tools(tools)
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        response = await client.chat.completions.create(**request_params)

        # Extract response data
        choice = response.choices[0]
        message = choice.message

        # Handle tool calls
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            model=response.model,
        )

    async def _stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response tokens"""
        client = await self._get_client()

        request_params = {
            "model": self.config.model,
            "messages": self._convert_messages(messages),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        stream = await client.chat.completions.create(**request_params)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
