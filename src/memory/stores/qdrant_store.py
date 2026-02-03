"""
Qdrant Vector Store Implementation

Qdrant is a high-performance vector database optimized for similarity search.
https://qdrant.tech/

Installation:
    pip install qdrant-client

Usage:
    from memory.stores import QdrantMemoryStore

    store = QdrantMemoryStore(
        host="localhost",
        port=6333,
        collection_name="agentic_sql_memories",
    )
    await store.connect()
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from .base import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    """Configuration for Qdrant connection"""
    # Connection settings
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    api_key: Optional[str] = None
    https: bool = False
    url: Optional[str] = None  # Alternative: full URL (e.g., "https://xxx.qdrant.io")

    # Collection settings
    collection_name: str = "agentic_sql_memories"
    vector_size: int = 1536  # OpenAI embedding size
    distance: str = "Cosine"  # Cosine, Euclid, Dot
    on_disk: bool = False  # Store vectors on disk

    # Multi-tenancy / Organization
    namespace: Optional[str] = None  # Logical namespace for isolation
    tenant_id: Optional[str] = None  # Tenant identifier for multi-tenant apps
    group_id: Optional[str] = None   # Group identifier (e.g., project, team)

    # Advanced settings
    shard_number: Optional[int] = None  # Number of shards
    replication_factor: Optional[int] = None  # Replication factor
    write_consistency_factor: Optional[int] = None  # Write consistency
    timeout: int = 30  # Request timeout in seconds


class QdrantMemoryStore(MemoryStore):
    """
    Qdrant Vector Store for semantic memory search.

    Features:
    - High-performance vector similarity search
    - Payload filtering
    - On-disk storage option
    - HNSW indexing for fast retrieval
    """

    def __init__(self, config: QdrantConfig):
        self.config = config
        self._client = None
        self._async_client = None

    async def connect(self) -> None:
        """Connect to Qdrant server"""
        try:
            from qdrant_client import QdrantClient, AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams

            # Connection kwargs
            connect_kwargs = {
                "api_key": self.config.api_key,
                "timeout": self.config.timeout,
            }

            if self.config.url:
                # Use full URL (for Qdrant Cloud)
                connect_kwargs["url"] = self.config.url
            else:
                # Use host/port
                connect_kwargs["host"] = self.config.host
                connect_kwargs["port"] = self.config.port
                connect_kwargs["https"] = self.config.https

            # Sync client for collection management
            self._client = QdrantClient(**connect_kwargs)

            # Async client for operations
            self._async_client = AsyncQdrantClient(**connect_kwargs)

            # Create collection if not exists
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.config.collection_name not in collection_names:
                distance_map = {
                    "Cosine": Distance.COSINE,
                    "Euclid": Distance.EUCLID,
                    "Dot": Distance.DOT,
                }

                # Collection creation kwargs
                create_kwargs = {
                    "collection_name": self.config.collection_name,
                    "vectors_config": VectorParams(
                        size=self.config.vector_size,
                        distance=distance_map.get(self.config.distance, Distance.COSINE),
                        on_disk=self.config.on_disk,
                    ),
                }

                # Add optional sharding/replication settings
                if self.config.shard_number:
                    create_kwargs["shard_number"] = self.config.shard_number
                if self.config.replication_factor:
                    create_kwargs["replication_factor"] = self.config.replication_factor
                if self.config.write_consistency_factor:
                    create_kwargs["write_consistency_factor"] = self.config.write_consistency_factor

                self._client.create_collection(**create_kwargs)
                logger.info(f"Created Qdrant collection: {self.config.collection_name}")

            # Log connection info
            location = self.config.url or f"{self.config.host}:{self.config.port}"
            namespace_info = f" (namespace={self.config.namespace})" if self.config.namespace else ""
            logger.info(f"Connected to Qdrant at {location}{namespace_info}")

        except ImportError:
            raise ImportError(
                "qdrant-client not installed. Install with: pip install qdrant-client"
            )

    async def disconnect(self) -> None:
        """Close Qdrant connection"""
        if self._async_client:
            await self._async_client.close()
        if self._client:
            self._client.close()

    async def store(self, memory: "Memory") -> bool:
        """Store memory with vector embedding"""
        from qdrant_client.models import PointStruct

        if not memory.embedding:
            logger.warning(f"Memory {memory.id} has no embedding, skipping vector store")
            return False

        try:
            # Build payload with multi-tenancy fields
            payload = {
                "content": memory.content,
                "type": memory.type.value,
                "priority": memory.priority.value,
                "entity_id": memory.entity_id,
                "process_id": memory.process_id,
                "session_id": memory.session_id,
                "metadata": json.dumps(memory.metadata),
                "created_at": memory.created_at.isoformat(),
                "access_count": memory.access_count,
                "relevance_score": memory.relevance_score,
            }

            # Add multi-tenancy fields from config
            if self.config.namespace:
                payload["namespace"] = self.config.namespace
            if self.config.tenant_id:
                payload["tenant_id"] = self.config.tenant_id
            if self.config.group_id:
                payload["group_id"] = self.config.group_id

            point = PointStruct(
                id=str(memory.id),
                vector=memory.embedding,
                payload=payload,
            )

            await self._async_client.upsert(
                collection_name=self.config.collection_name,
                points=[point],
            )
            return True

        except Exception as e:
            logger.error(f"Failed to store memory in Qdrant: {e}")
            return False

    async def retrieve(
        self,
        query: str,
        memory_type: Optional["MemoryType"] = None,
        limit: int = 10,
        query_vector: Optional[List[float]] = None,
        score_threshold: float = 0.7,
        namespace: Optional[str] = None,
        tenant_id: Optional[str] = None,
        group_id: Optional[str] = None,
        **kwargs,
    ) -> List["Memory"]:
        """
        Retrieve memories by vector similarity.

        Args:
            query: Text query (not used if query_vector provided)
            memory_type: Filter by memory type
            limit: Max results to return
            query_vector: Pre-computed query embedding
            score_threshold: Minimum similarity score
            namespace: Filter by namespace (overrides config)
            tenant_id: Filter by tenant (overrides config)
            group_id: Filter by group (overrides config)
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from ..manager import Memory, MemoryType, MemoryPriority

        if not query_vector:
            logger.warning("No query vector provided for Qdrant search")
            return []

        try:
            # Build filter conditions
            filter_conditions = []

            # Memory type filter
            if memory_type:
                filter_conditions.append(
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=memory_type.value),
                    )
                )

            # Multi-tenancy filters (use param or fall back to config)
            effective_namespace = namespace or self.config.namespace
            effective_tenant = tenant_id or self.config.tenant_id
            effective_group = group_id or self.config.group_id

            if effective_namespace:
                filter_conditions.append(
                    FieldCondition(
                        key="namespace",
                        match=MatchValue(value=effective_namespace),
                    )
                )

            if effective_tenant:
                filter_conditions.append(
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=effective_tenant),
                    )
                )

            if effective_group:
                filter_conditions.append(
                    FieldCondition(
                        key="group_id",
                        match=MatchValue(value=effective_group),
                    )
                )

            query_filter = Filter(must=filter_conditions) if filter_conditions else None

            # Search
            results = await self._async_client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )

            # Convert to Memory objects
            memories = []
            for hit in results:
                payload = hit.payload
                memories.append(Memory(
                    id=UUID(hit.id),
                    content=payload.get("content", ""),
                    type=MemoryType(payload.get("type", "semantic")),
                    priority=MemoryPriority(payload.get("priority", "medium")),
                    entity_id=payload.get("entity_id"),
                    process_id=payload.get("process_id"),
                    session_id=payload.get("session_id"),
                    metadata=json.loads(payload.get("metadata", "{}")),
                    created_at=datetime.fromisoformat(payload.get("created_at", datetime.utcnow().isoformat())),
                    access_count=payload.get("access_count", 0),
                    relevance_score=hit.score,
                    embedding=None,  # Don't return embedding
                ))

            return memories

        except Exception as e:
            logger.error(f"Failed to search Qdrant: {e}")
            return []

    async def delete(self, memory_id: UUID) -> bool:
        """Delete memory by ID"""
        from qdrant_client.models import PointIdsList

        try:
            await self._async_client.delete(
                collection_name=self.config.collection_name,
                points_selector=PointIdsList(points=[str(memory_id)]),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Qdrant: {e}")
            return False

    async def update(self, memory: "Memory") -> bool:
        """Update memory (upsert)"""
        return await self.store(memory)

    async def count(self) -> int:
        """Get total memory count"""
        try:
            info = await self._async_client.get_collection(self.config.collection_name)
            return info.points_count
        except Exception:
            return 0
