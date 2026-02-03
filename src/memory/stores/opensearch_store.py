"""
OpenSearch Vector Store Implementation

OpenSearch provides both vector search and full-text search capabilities.
https://opensearch.org/

Installation:
    pip install opensearch-py

Usage:
    from memory.stores import OpenSearchMemoryStore

    store = OpenSearchMemoryStore(
        hosts=["https://localhost:9200"],
        index_name="agentic_sql_memories",
        username="admin",
        password="admin",
    )
    await store.connect()
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from .base import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class OpenSearchConfig:
    """Configuration for OpenSearch connection"""
    hosts: List[str] = field(default_factory=lambda: ["https://localhost:9200"])
    index_name: str = "agentic_sql_memories"
    username: Optional[str] = None
    password: Optional[str] = None
    use_ssl: bool = True
    verify_certs: bool = False
    ssl_show_warn: bool = False
    vector_dimension: int = 1536  # OpenAI embedding size
    ef_construction: int = 512  # HNSW parameter
    m: int = 16  # HNSW parameter


class OpenSearchMemoryStore(MemoryStore):
    """
    OpenSearch Store for hybrid vector + full-text search.

    Features:
    - k-NN vector similarity search
    - Full-text search with BM25
    - Hybrid search combining both
    - Rich filtering capabilities
    - Scalable and distributed
    """

    def __init__(self, config: OpenSearchConfig):
        self.config = config
        self._client = None

    async def connect(self) -> None:
        """Connect to OpenSearch cluster"""
        try:
            from opensearchpy import OpenSearch, AsyncOpenSearch

            auth = None
            if self.config.username and self.config.password:
                auth = (self.config.username, self.config.password)

            # Async client
            self._client = AsyncOpenSearch(
                hosts=self.config.hosts,
                http_auth=auth,
                use_ssl=self.config.use_ssl,
                verify_certs=self.config.verify_certs,
                ssl_show_warn=self.config.ssl_show_warn,
            )

            # Create index if not exists
            index_exists = await self._client.indices.exists(index=self.config.index_name)

            if not index_exists:
                index_body = {
                    "settings": {
                        "index": {
                            "knn": True,
                            "knn.algo_param.ef_search": 100,
                        },
                    },
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": self.config.vector_dimension,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "nmslib",
                                    "parameters": {
                                        "ef_construction": self.config.ef_construction,
                                        "m": self.config.m,
                                    },
                                },
                            },
                            "type": {"type": "keyword"},
                            "priority": {"type": "keyword"},
                            "entity_id": {"type": "keyword"},
                            "process_id": {"type": "keyword"},
                            "session_id": {"type": "keyword"},
                            "metadata": {"type": "object", "enabled": False},
                            "created_at": {"type": "date"},
                            "access_count": {"type": "integer"},
                            "relevance_score": {"type": "float"},
                        },
                    },
                }

                await self._client.indices.create(
                    index=self.config.index_name,
                    body=index_body,
                )
                logger.info(f"Created OpenSearch index: {self.config.index_name}")

            logger.info(f"Connected to OpenSearch at {self.config.hosts}")

        except ImportError:
            raise ImportError(
                "opensearch-py not installed. Install with: pip install opensearch-py"
            )

    async def disconnect(self) -> None:
        """Close OpenSearch connection"""
        if self._client:
            await self._client.close()

    async def store(self, memory: "Memory") -> bool:
        """Store memory with vector embedding"""
        try:
            doc = {
                "content": memory.content,
                "type": memory.type.value,
                "priority": memory.priority.value,
                "entity_id": memory.entity_id,
                "process_id": memory.process_id,
                "session_id": memory.session_id,
                "metadata": memory.metadata,
                "created_at": memory.created_at.isoformat(),
                "access_count": memory.access_count,
                "relevance_score": memory.relevance_score,
            }

            if memory.embedding:
                doc["embedding"] = memory.embedding

            await self._client.index(
                index=self.config.index_name,
                id=str(memory.id),
                body=doc,
                refresh=True,
            )
            return True

        except Exception as e:
            logger.error(f"Failed to store memory in OpenSearch: {e}")
            return False

    async def retrieve(
        self,
        query: str,
        memory_type: Optional["MemoryType"] = None,
        limit: int = 10,
        query_vector: Optional[List[float]] = None,
        use_hybrid: bool = True,
        **kwargs,
    ) -> List["Memory"]:
        """
        Retrieve memories using vector search, text search, or hybrid.

        Args:
            query: Text query for BM25 search
            memory_type: Filter by memory type
            limit: Max results to return
            query_vector: Vector for k-NN search
            use_hybrid: Combine vector and text search
        """
        from ..manager import Memory, MemoryType, MemoryPriority

        try:
            # Build query
            must_clauses = []
            filter_clauses = []

            # Filter by type if specified
            if memory_type:
                filter_clauses.append({
                    "term": {"type": memory_type.value}
                })

            # Vector search
            if query_vector:
                knn_query = {
                    "knn": {
                        "embedding": {
                            "vector": query_vector,
                            "k": limit,
                        }
                    }
                }

                if use_hybrid and query:
                    # Hybrid: combine k-NN with text search
                    search_body = {
                        "size": limit,
                        "query": {
                            "bool": {
                                "should": [
                                    knn_query,
                                    {"match": {"content": query}},
                                ],
                                "filter": filter_clauses if filter_clauses else None,
                            }
                        },
                    }
                else:
                    # Pure vector search
                    search_body = {
                        "size": limit,
                        "query": {
                            "bool": {
                                "must": [knn_query],
                                "filter": filter_clauses if filter_clauses else None,
                            }
                        },
                    }
            else:
                # Pure text search
                search_body = {
                    "size": limit,
                    "query": {
                        "bool": {
                            "must": [{"match": {"content": query}}] if query else [],
                            "filter": filter_clauses if filter_clauses else None,
                        }
                    },
                }

            # Execute search
            response = await self._client.search(
                index=self.config.index_name,
                body=search_body,
            )

            # Convert to Memory objects
            memories = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                memories.append(Memory(
                    id=UUID(hit["_id"]),
                    content=source.get("content", ""),
                    type=MemoryType(source.get("type", "semantic")),
                    priority=MemoryPriority(source.get("priority", "medium")),
                    entity_id=source.get("entity_id"),
                    process_id=source.get("process_id"),
                    session_id=source.get("session_id"),
                    metadata=source.get("metadata", {}),
                    created_at=datetime.fromisoformat(source.get("created_at", datetime.utcnow().isoformat())),
                    access_count=source.get("access_count", 0),
                    relevance_score=hit["_score"],
                    embedding=None,
                ))

            return memories

        except Exception as e:
            logger.error(f"Failed to search OpenSearch: {e}")
            return []

    async def delete(self, memory_id: UUID) -> bool:
        """Delete memory by ID"""
        try:
            await self._client.delete(
                index=self.config.index_name,
                id=str(memory_id),
                refresh=True,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete from OpenSearch: {e}")
            return False

    async def update(self, memory: "Memory") -> bool:
        """Update memory (reindex)"""
        return await self.store(memory)

    async def count(self) -> int:
        """Get total memory count"""
        try:
            response = await self._client.count(index=self.config.index_name)
            return response["count"]
        except Exception:
            return 0

    async def search_text(
        self,
        query: str,
        memory_type: Optional["MemoryType"] = None,
        limit: int = 10,
    ) -> List["Memory"]:
        """Full-text search using BM25"""
        return await self.retrieve(
            query=query,
            memory_type=memory_type,
            limit=limit,
            query_vector=None,
            use_hybrid=False,
        )
