"""
Azure OpenAI LLM Client Implementation
"""
from __future__ import annotations

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


class AzureOpenAIConfig(LLMConfig):
    """Azure OpenAI specific configuration"""
    azure_endpoint: str = ""
    azure_deployment: str = ""
    api_version: str = "2024-02-01"
    embedding_deployment: str = ""


class AzureOpenAIClient(LLMClient):
    """
    Azure OpenAI API Client

    Supports:
    - GPT-4, GPT-4 Turbo, GPT-4o via Azure
    - Function calling
    - Streaming
    - Embeddings
    """

    def __init__(self, config: AzureOpenAIConfig):
        super().__init__(config)
        self.azure_config = config
        self._client = None
        self._sync_client = None

    async def _get_client(self):
        """Lazy load Azure OpenAI async client"""
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI

                self._client = AsyncAzureOpenAI(
                    api_key=self.azure_config.api_key,
                    azure_endpoint=self.azure_config.azure_endpoint,
                    api_version=self.azure_config.api_version,
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai")

        return self._client

    def _get_sync_client(self):
        """Get synchronous client for embeddings"""
        if self._sync_client is None:
            try:
                from openai import AzureOpenAI

                self._sync_client = AzureOpenAI(
                    api_key=self.azure_config.api_key,
                    azure_endpoint=self.azure_config.azure_endpoint,
                    api_version=self.azure_config.api_version,
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai")

        return self._sync_client

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
        """Generate response using Azure OpenAI API"""
        client = await self._get_client()

        request_params = {
            "model": self.azure_config.azure_deployment,
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
            "model": self.azure_config.azure_deployment,
            "messages": self._convert_messages(messages),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        stream = await client.chat.completions.create(**request_params)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Azure OpenAI"""
        client = self._get_sync_client()

        response = client.embeddings.create(
            model=self.azure_config.embedding_deployment,
            input=text,
        )

        return response.data[0].embedding

    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding asynchronously"""
        client = await self._get_client()

        response = await client.embeddings.create(
            model=self.azure_config.embedding_deployment,
            input=text,
        )

        return response.data[0].embedding

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Simple generation interface for prompts"""
        messages = []

        if system_prompt:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))

        messages.append(Message(role=MessageRole.USER, content=prompt))

        # Temporarily override max_tokens if specified
        original_max_tokens = self.config.max_tokens
        if max_tokens:
            self.config.max_tokens = max_tokens

        try:
            response = await self._generate(messages, **kwargs)
            return response.content
        finally:
            self.config.max_tokens = original_max_tokens
