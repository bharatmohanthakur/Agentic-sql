"""
AWS Bedrock LLM Client - Claude models via AWS Bedrock using boto3
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import (
    LLMClient,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    Message,
    MessageRole,
    TokenUsage,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class BedrockConfig(LLMConfig):
    """
    Configuration for AWS Bedrock client.

    Supports Claude models through AWS Bedrock.
    Authentication uses AWS credentials (environment variables, IAM role, or explicit keys).
    """

    provider: LLMProvider = LLMProvider.BEDROCK

    # AWS Bedrock specific settings
    region_name: str = "us-east-1"
    model: str = "anthropic.claude-3-sonnet-20240229-v1:0"

    # AWS credentials (optional - can use IAM role or environment variables)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None

    # Bedrock-specific options
    anthropic_version: str = "bedrock-2023-05-31"

    # Override defaults for Claude
    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 1.0
    top_k: Optional[int] = None

    # Stop sequences
    stop_sequences: Optional[List[str]] = None


class BedrockClient(LLMClient):
    """
    AWS Bedrock LLM Client using boto3.

    Supports Claude models (Opus, Sonnet, Haiku) through AWS Bedrock.

    Features:
    - Full Claude model support via Bedrock
    - Tool/function calling support
    - Streaming support
    - AWS IAM authentication
    - Cost tracking

    Example:
        ```python
        from llm.bedrock_client import BedrockClient, BedrockConfig

        # Using IAM role (recommended for EC2/Lambda)
        client = BedrockClient(BedrockConfig(
            region_name="us-east-1",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
        ))

        # Using explicit credentials
        client = BedrockClient(BedrockConfig(
            region_name="us-east-1",
            model="anthropic.claude-3-opus-20240229-v1:0",
            aws_access_key_id="AKIA...",
            aws_secret_access_key="...",
        ))

        response = await client.generate("Hello, how are you?")
        ```
    """

    # Available Bedrock Claude models
    AVAILABLE_MODELS = {
        # Claude 3.5 models
        "anthropic.claude-3-5-sonnet-20240620-v1:0": "Claude 3.5 Sonnet",
        "anthropic.claude-3-5-sonnet-20241022-v2:0": "Claude 3.5 Sonnet v2",
        "anthropic.claude-3-5-haiku-20241022-v1:0": "Claude 3.5 Haiku",
        # Claude 3 models
        "anthropic.claude-3-opus-20240229-v1:0": "Claude 3 Opus",
        "anthropic.claude-3-sonnet-20240229-v1:0": "Claude 3 Sonnet",
        "anthropic.claude-3-haiku-20240307-v1:0": "Claude 3 Haiku",
        # Claude 2 models (legacy)
        "anthropic.claude-v2:1": "Claude 2.1",
        "anthropic.claude-v2": "Claude 2.0",
        "anthropic.claude-instant-v1": "Claude Instant",
    }

    def __init__(self, config: BedrockConfig):
        super().__init__(config)
        self.config: BedrockConfig = config
        self._client = None
        self._runtime_client = None

    async def _get_client(self):
        """Lazy load boto3 Bedrock runtime client"""
        if self._runtime_client is None:
            try:
                import boto3
                from botocore.config import Config

                # Build boto3 config
                boto_config = Config(
                    region_name=self.config.region_name,
                    retries={"max_attempts": self.config.retry_attempts, "mode": "adaptive"},
                )

                # Build credentials dict
                credentials = {}
                if self.config.aws_access_key_id:
                    credentials["aws_access_key_id"] = self.config.aws_access_key_id
                if self.config.aws_secret_access_key:
                    credentials["aws_secret_access_key"] = self.config.aws_secret_access_key
                if self.config.aws_session_token:
                    credentials["aws_session_token"] = self.config.aws_session_token

                # Create bedrock-runtime client
                self._runtime_client = boto3.client(
                    "bedrock-runtime",
                    config=boto_config,
                    **credentials,
                )

                logger.info(f"Initialized Bedrock client for region: {self.config.region_name}")

            except ImportError:
                raise ImportError(
                    "boto3 required for AWS Bedrock: pip install boto3"
                )

        return self._runtime_client

    def _convert_messages(self, messages: List[Message]) -> tuple[Optional[str], List[Dict]]:
        """
        Convert internal messages to Bedrock/Claude format.

        Bedrock Claude API expects:
        - System message separate from messages array
        - Messages as list of {role, content}
        - Tool results in specific format

        Returns:
            Tuple of (system_message, messages_list)
        """
        system = None
        converted = []

        for msg in messages:
            # Extract system message
            if msg.role == MessageRole.SYSTEM:
                system = msg.content
                continue

            # Map role
            if msg.role == MessageRole.USER:
                role = "user"
            elif msg.role == MessageRole.ASSISTANT:
                role = "assistant"
            elif msg.role == MessageRole.TOOL:
                # Tool results go as user messages with tool_result content
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
            else:
                role = "user"

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
        """Convert tools to Bedrock/Claude format"""
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
        """Generate response using AWS Bedrock"""
        client = await self._get_client()

        system, converted_messages = self._convert_messages(messages)

        # Build request body for Claude on Bedrock
        request_body = {
            "anthropic_version": self.config.anthropic_version,
            "max_tokens": self.config.max_tokens,
            "messages": converted_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

        # Add optional parameters
        if system:
            request_body["system"] = system

        if self.config.top_k is not None:
            request_body["top_k"] = self.config.top_k

        if self.config.stop_sequences:
            request_body["stop_sequences"] = self.config.stop_sequences

        if tools:
            request_body["tools"] = self._convert_tools(tools)

        # Run synchronous boto3 call in executor
        loop = asyncio.get_event_loop()

        def _invoke():
            response = client.invoke_model(
                modelId=self.config.model,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            return json.loads(response["body"].read())

        response_body = await loop.run_in_executor(None, _invoke)

        # Parse response
        content = ""
        tool_calls = []

        for block in response_body.get("content", []):
            if block.get("type") == "text":
                content = block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        # Extract usage
        usage_data = response_body.get("usage", {})

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response_body.get("stop_reason", "end_turn"),
            usage=TokenUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
            ),
            model=self.config.model,
            metadata={
                "bedrock_model_id": self.config.model,
                "region": self.config.region_name,
            },
        )

    async def _stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response tokens from AWS Bedrock"""
        client = await self._get_client()

        system, converted_messages = self._convert_messages(messages)

        # Build request body
        request_body = {
            "anthropic_version": self.config.anthropic_version,
            "max_tokens": self.config.max_tokens,
            "messages": converted_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

        if system:
            request_body["system"] = system

        if self.config.top_k is not None:
            request_body["top_k"] = self.config.top_k

        if self.config.stop_sequences:
            request_body["stop_sequences"] = self.config.stop_sequences

        # Use invoke_model_with_response_stream for streaming
        loop = asyncio.get_event_loop()

        def _invoke_stream():
            response = client.invoke_model_with_response_stream(
                modelId=self.config.model,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            return response

        response = await loop.run_in_executor(None, _invoke_stream)

        # Process streaming response
        stream = response.get("body")
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    chunk_data = json.loads(chunk.get("bytes", b"{}").decode())

                    # Handle different event types
                    event_type = chunk_data.get("type")

                    if event_type == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield text

                    elif event_type == "message_delta":
                        # End of message
                        pass

    async def list_models(self) -> List[Dict[str, str]]:
        """List available Claude models on Bedrock"""
        return [
            {"model_id": k, "name": v}
            for k, v in self.AVAILABLE_MODELS.items()
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_id": self.config.model,
            "model_name": self.AVAILABLE_MODELS.get(self.config.model, "Unknown"),
            "provider": "AWS Bedrock",
            "region": self.config.region_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
