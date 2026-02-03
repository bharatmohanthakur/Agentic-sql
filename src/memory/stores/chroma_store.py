"""
ChromaDB Vector Store Implementation

ChromaDB is an open-source embedding database.
https://www.trychroma.com/

Installation:
    pip install chromadb

Usage:
    from memory.stores import ChromaMemoryStore

    store = ChromaMemoryStore(
        path="./chroma_data",
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
class ChromaConfig:
    """Configuration for ChromaDB"""
    path: str = "./chroma_data"  # Persistent storage path
    collection_name: str = "agentic_sql_memories"
    host: Optional[str] = None  # For client-server mode
    port: Optional[int] = None
    distance_fn: str = "cosine"  # cosine, l2, ip


class ChromaMemoryStore(MemoryStore):
    """
    ChromaDB Vector Store for semantic memory search.

    Features:
    - Simple setup, no server required
    - Persistent local storage
    - Metadata filtering
    - Automatic embedding (optional)
    """

    def __init__(self, config: ChromaConfig):
        self.config = config
        self._client = None
        self._collection = None

    async def connect(self) -> None:
        """Connect to ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings

            if self.config.host and self.config.port:
                # Client-server mode
                self._client = chromadb.HttpClient(
                    host=self.config.host,
                    port=self.config.port,
                )
            else:
                # Persistent local mode
                self._client = chromadb.PersistentClient(
                    path=self.config.path,
                    settings=Settings(anonymized_telemetry=False),
                )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_fn},
            )

            logger.info(f"Connected to ChromaDB at {self.config.path}")

        except ImportError:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )

    async def disconnect(self) -> None:
        """Close ChromaDB connection"""
        pass  # ChromaDB handles cleanup automatically

    async def store(self, memory: "Memory") -> bool:
        """Store memory with vector embedding"""
        try:
            metadata = {
                "type": memory.type.value,
                "priority": memory.priority.value,
                "entity_id": memory.entity_id or "",
                "process_id": memory.process_id or "",
                "session_id": memory.session_id or "",
                "created_at": memory.created_at.isoformat(),
                "access_count": memory.access_count,
                "relevance_score": memory.relevance_score,
            }

            if memory.embedding:
                self._collection.upsert(
                    ids=[str(memory.id)],
                    embeddings=[memory.embedding],
                    documents=[memory.content],
                    metadatas=[metadata],
                )
            else:
                self._collection.upsert(
                    ids=[str(memory.id)],
                    documents=[memory.content],
                    metadatas=[metadata],
                )

            return True

        except Exception as e:
            logger.error(f"Failed to store memory in ChromaDB: {e}")
            return False

    async def retrieve(
        self,
        query: str,
        memory_type: Optional["MemoryType"] = None,
        limit: int = 10,
        query_vector: Optional[List[float]] = None,
        **kwargs,
    ) -> List["Memory"]:
        """Retrieve memories by vector similarity"""
        from ..manager import Memory, MemoryType, MemoryPriority

        try:
            where_filter = None
            if memory_type:
                where_filter = {"type": memory_type.value}

            if query_vector:
                results = self._collection.query(
                    query_embeddings=[query_vector],
                    n_results=limit,
                    where=where_filter,
                )
            else:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where=where_filter,
                )

            memories = []
            if results and results["ids"]:
                for i, id_str in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    document = results["documents"][0][i] if results["documents"] else ""
                    distance = results["distances"][0][i] if results["distances"] else 0

                    memories.append(Memory(
                        id=UUID(id_str),
                        content=document,
                        type=MemoryType(metadata.get("type", "semantic")),
                        priority=MemoryPriority(metadata.get("priority", "medium")),
                        entity_id=metadata.get("entity_id") or None,
                        process_id=metadata.get("process_id") or None,
                        session_id=metadata.get("session_id") or None,
                        metadata={},
                        created_at=datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat())),
                        access_count=metadata.get("access_count", 0),
                        relevance_score=1 - distance,  # Convert distance to similarity
                        embedding=None,
                    ))

            return memories

        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            return []

    async def delete(self, memory_id: UUID) -> bool:
        """Delete memory by ID"""
        try:
            self._collection.delete(ids=[str(memory_id)])
            return True
        except Exception as e:
            logger.error(f"Failed to delete from ChromaDB: {e}")
            return False

    async def update(self, memory: "Memory") -> bool:
        """Update memory (upsert)"""
        return await self.store(memory)

    async def count(self) -> int:
        """Get total memory count"""
        try:
            return self._collection.count()
        except Exception:
            return 0
