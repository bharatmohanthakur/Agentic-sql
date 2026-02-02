"""
Anthropic Claude LLM Client Implementation
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


class AnthropicClient(LLMClient):
    """
    Anthropic Claude API Client

    Supports:
    - Claude 3 Opus, Sonnet, Haiku
    - Tool use
    - Streaming
    - Extended thinking
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    async def _get_client(self):
        """Lazy load Anthropic client"""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                )
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")

        return self._client

    def _convert_messages(self, messages: List[Message]) -> tuple[Optional[str], List[Dict]]:
        """
        Convert internal messages to Anthropic format

        Returns:
            Tuple of (system_message, conversation_messages)
        """
        system = None
        converted = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system = msg.content
                continue

            role = "user" if msg.role == MessageRole.USER else "assistant"

            # Handle tool results
            if msg.role == MessageRole.TOOL:
                converted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
                continue

            # Handle tool calls in assistant messages
            if msg.tool_calls:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    })
                converted.append({"role": role, "content": content})
            else:
                converted.append({"role": role, "content": msg.content})

        return system, converted

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict]:
        """Convert tools to Anthropic format"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    async def _generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using Anthropic API"""
        client = await self._get_client()

        system, converted_messages = self._convert_messages(messages)

        request_params = {
            "model": self.config.model,
            "messages": converted_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if system:
            request_params["system"] = system

        if tools:
            request_params["tools"] = self._convert_tools(tools)

        response = await client.messages.create(**request_params)

        # Extract content and tool calls
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason or "stop",
            usage=TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
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

        system, converted_messages = self._convert_messages(messages)

        request_params = {
            "model": self.config.model,
            "messages": converted_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if system:
            request_params["system"] = system

        async with client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield text
