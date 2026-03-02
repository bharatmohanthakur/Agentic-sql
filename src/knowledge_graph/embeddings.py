"""
Graph Embedding Service - Generates embeddings for knowledge graph elements.

Uses AWS Bedrock (Titan Embeddings) to create vector representations of
graph nodes, edges, and subgraphs for semantic search.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding service"""
    # AWS Bedrock settings
    region_name: str = "us-east-1"
    model_id: str = "amazon.titan-embed-text-v2:0"

    # AWS credentials (optional - can use IAM role)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # Embedding settings
    dimensions: int = 1536
    normalize: bool = True

    # Batching
    batch_size: int = 25
    max_concurrent: int = 5


class GraphEmbeddingService:
    """
    Generates embeddings for knowledge graph elements using AWS Bedrock.

    Supports:
    - Node embeddings (from node properties and labels)
    - Edge embeddings (from relationship type and connected node context)
    - Subgraph embeddings (from path descriptions)
    - Batch embedding generation
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._client = None

    async def _get_client(self):
        """Lazy load Bedrock runtime client"""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config

                boto_config = Config(
                    region_name=self.config.region_name,
                    retries={"max_attempts": 3, "mode": "adaptive"},
                )

                credentials = {}
                if self.config.aws_access_key_id:
                    credentials["aws_access_key_id"] = self.config.aws_access_key_id
                if self.config.aws_secret_access_key:
                    credentials["aws_secret_access_key"] = self.config.aws_secret_access_key

                self._client = boto3.client(
                    "bedrock-runtime",
                    config=boto_config,
                    **credentials,
                )
            except ImportError:
                raise ImportError("boto3 required: pip install boto3")

        return self._client

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text string"""
        client = await self._get_client()

        request_body = {
            "inputText": text,
            "dimensions": self.config.dimensions,
            "normalize": self.config.normalize,
        }

        loop = asyncio.get_event_loop()

        def _invoke():
            response = client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            return json.loads(response["body"].read())

        response = await loop.run_in_executor(None, _invoke)
        return response.get("embedding", [])

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        embeddings = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def _embed_one(text: str) -> List[float]:
            async with semaphore:
                return await self.embed_text(text)

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            batch_results = await asyncio.gather(
                *[_embed_one(text) for text in batch]
            )
            embeddings.extend(batch_results)

        return embeddings

    async def embed_node(self, node_labels: List[str], properties: Dict[str, Any]) -> List[float]:
        """
        Generate embedding for a graph node.

        Combines node labels and properties into a text representation.
        """
        parts = [f"Node type: {', '.join(node_labels)}"]
        for key, value in properties.items():
            if not key.startswith("_") and key != "embedding":
                parts.append(f"{key}: {value}")

        text = ". ".join(parts)
        return await self.embed_text(text)

    async def embed_edge(
        self,
        source_label: str,
        relationship_type: str,
        target_label: str,
        source_name: str = "",
        target_name: str = "",
    ) -> List[float]:
        """
        Generate embedding for a graph edge/relationship.

        Captures the semantic meaning of the relationship.
        """
        text = (
            f"{source_label} {source_name} "
            f"has relationship {relationship_type} "
            f"with {target_label} {target_name}"
        )
        return await self.embed_text(text)

    async def embed_path(self, path_description: str) -> List[float]:
        """Generate embedding for a graph path"""
        return await self.embed_text(f"Graph path: {path_description}")
