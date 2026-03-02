"""
Unified Embedding Service - Manages embeddings across all semantic layers.

Provides a single interface for generating and storing embeddings
for ontology concepts, graph nodes, canonical entities, and table metadata.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingServiceConfig:
    """Configuration for the unified embedding service"""
    region_name: str = "us-east-1"
    model_id: str = "amazon.titan-embed-text-v2:0"
    dimensions: int = 1024
    normalize: bool = True
    batch_size: int = 25

    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


class UnifiedEmbeddingService:
    """
    Unified embedding service for all semantic layers.

    Generates embeddings using AWS Bedrock Titan for:
    - Ontology concepts (classes, fact types, constraints)
    - Knowledge graph nodes (entities, relationships)
    - Canonical entities (Pydantic model instances)
    - Dimensional metadata (tables, columns, descriptions)
    """

    def __init__(self, config: Optional[EmbeddingServiceConfig] = None):
        self.config = config or EmbeddingServiceConfig()
        self._client = None

    async def _get_client(self):
        """Lazy-load Bedrock runtime client"""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config

                credentials = {}
                if self.config.aws_access_key_id:
                    credentials["aws_access_key_id"] = self.config.aws_access_key_id
                if self.config.aws_secret_access_key:
                    credentials["aws_secret_access_key"] = self.config.aws_secret_access_key

                self._client = boto3.client(
                    "bedrock-runtime",
                    config=Config(region_name=self.config.region_name),
                    **credentials,
                )
            except ImportError:
                raise ImportError("boto3 required: pip install boto3")
        return self._client

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a text string"""
        client = await self._get_client()
        loop = asyncio.get_event_loop()

        def _invoke():
            response = client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps({
                    "inputText": text,
                    "dimensions": self.config.dimensions,
                    "normalize": self.config.normalize,
                }),
                contentType="application/json",
                accept="application/json",
            )
            return json.loads(response["body"].read())

        result = await loop.run_in_executor(None, _invoke)
        return result.get("embedding", [])

    # ── Layer-Specific Embedding Methods ──

    async def embed_ontology_concept(
        self,
        concept_name: str,
        concept_type: str,
        description: str = "",
        related_facts: Optional[List[str]] = None,
    ) -> List[float]:
        """Embed an ontology concept (ObjectType or ElementaryFact)"""
        text = f"Ontology concept: {concept_name} (type: {concept_type}). "
        if description:
            text += f"Description: {description}. "
        if related_facts:
            text += f"Related facts: {', '.join(related_facts)}"
        return await self.embed_text(text)

    async def embed_graph_node(
        self,
        labels: List[str],
        properties: Dict[str, Any],
        neighbors: Optional[List[str]] = None,
    ) -> List[float]:
        """Embed a knowledge graph node with relationship context"""
        text = f"Entity: {', '.join(labels)}. "
        for key, value in properties.items():
            if not key.startswith("_") and key != "embedding":
                text += f"{key}: {value}. "
        if neighbors:
            text += f"Connected to: {', '.join(neighbors[:10])}"
        return await self.embed_text(text)

    async def embed_canonical_entity(
        self,
        entity_type: str,
        properties: Dict[str, Any],
    ) -> List[float]:
        """Embed a canonical entity"""
        text = f"Canonical {entity_type}. "
        for key, value in properties.items():
            text += f"{key}: {value}. "
        return await self.embed_text(text)

    async def embed_table_metadata(
        self,
        table_name: str,
        description: str,
        columns: List[Dict[str, str]],
        grain: str = "",
    ) -> List[float]:
        """Embed dimensional model table metadata"""
        text = f"Table: {table_name}. "
        if description:
            text += f"Description: {description}. "
        if grain:
            text += f"Grain: {grain}. "
        col_strs = [f"{c.get('name', '')} ({c.get('type', '')})" for c in columns[:20]]
        text += f"Columns: {', '.join(col_strs)}"
        return await self.embed_text(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        semaphore = asyncio.Semaphore(5)

        async def _embed_one(text: str) -> List[float]:
            async with semaphore:
                return await self.embed_text(text)

        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            batch_results = await asyncio.gather(
                *[_embed_one(text) for text in batch]
            )
            results.extend(batch_results)

        return results
