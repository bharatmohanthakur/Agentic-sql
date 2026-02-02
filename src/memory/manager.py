"""
Hybrid Memory Manager - Combines Graph + Vector + SQL storage
Implements Cognee's ECL pattern and Memori's SQL-native persistence
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memories stored"""
    CONVERSATION = "conversation"  # Chat history
    ENTITY_FACT = "entity_fact"  # Facts about entities
    SCHEMA = "schema"  # Database schema knowledge
    QUERY_PATTERN = "query_pattern"  # Successful SQL patterns
    ERROR_PATTERN = "error_pattern"  # Error patterns to avoid
    USER_PREFERENCE = "user_preference"  # User-specific preferences
    SEMANTIC = "semantic"  # General semantic knowledge


class MemoryPriority(str, Enum):
    """Priority levels for memory retrieval"""
    CRITICAL = "critical"  # Always retrieve
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Memory:
    """Single memory unit"""
    id: UUID = field(default_factory=uuid4)
    type: MemoryType = MemoryType.SEMANTIC
    content: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: MemoryPriority = MemoryPriority.MEDIUM
    entity_id: Optional[str] = None  # For entity-specific memories
    process_id: Optional[str] = None  # For agent/process-specific
    session_id: Optional[str] = None  # For session-specific
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    relevance_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "priority": self.priority.value,
            "entity_id": self.entity_id,
            "process_id": self.process_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "relevance_score": self.relevance_score,
        }


class MemoryConfig(BaseModel):
    """Configuration for memory manager"""
    enable_graph: bool = True
    enable_vector: bool = True
    enable_sql: bool = True
    vector_dimensions: int = 1536  # OpenAI embedding size
    similarity_threshold: float = 0.7
    max_memories_per_query: int = 10
    memory_ttl_days: int = 90
    enable_compression: bool = True
    enable_deduplication: bool = True
    consolidation_interval_hours: int = 24


class MemoryStore(ABC):
    """Abstract base for memory storage backends"""

    @abstractmethod
    async def store(self, memory: Memory) -> bool:
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Memory]:
        pass

    @abstractmethod
    async def delete(self, memory_id: UUID) -> bool:
        pass

    @abstractmethod
    async def update(self, memory: Memory) -> bool:
        pass


class MemoryManager:
    """
    Hybrid Memory Manager implementing:
    1. ECL (Extract, Cognify, Load) pattern from Cognee
    2. Multi-level hierarchy from Memori (entity, process, session)
    3. Graph + Vector hybrid retrieval

    Memory flows through:
    - ADD: Ingest raw data
    - COGNIFY: Extract entities, relationships, embeddings
    - MEMIFY: Enrich with computed properties
    - SEARCH: Retrieve relevant memories
    """

    def __init__(
        self,
        config: MemoryConfig,
        graph_store: Optional[MemoryStore] = None,
        vector_store: Optional[MemoryStore] = None,
        sql_store: Optional[MemoryStore] = None,
        embedding_fn: Optional[callable] = None,
    ):
        self.config = config
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.sql_store = sql_store
        self.embedding_fn = embedding_fn
        self._cache: Dict[str, Memory] = {}
        self._consolidation_lock = asyncio.Lock()

    async def add(
        self,
        content: str,
        memory_type: MemoryType,
        entity_id: Optional[str] = None,
        process_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
    ) -> Memory:
        """
        ADD phase: Ingest raw content into memory system
        """
        # Deduplication check
        if self.config.enable_deduplication:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            if content_hash in self._cache:
                existing = self._cache[content_hash]
                existing.access_count += 1
                existing.updated_at = datetime.utcnow()
                return existing

        memory = Memory(
            type=memory_type,
            content=content,
            entity_id=entity_id,
            process_id=process_id,
            session_id=session_id,
            metadata=metadata or {},
            priority=priority,
        )

        # Cache it
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        self._cache[content_hash] = memory

        logger.debug(f"Added memory: {memory.id} [{memory_type.value}]")
        return memory

    async def cognify(self, memory: Memory) -> Memory:
        """
        COGNIFY phase: Transform raw content into structured knowledge
        - Generate embeddings
        - Extract entities and relationships
        - Build graph connections
        """
        # Generate embedding if function available
        if self.embedding_fn and self.config.enable_vector:
            try:
                memory.embedding = await self._generate_embedding(memory.content)
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")

        # Extract entities and relationships for graph
        if self.config.enable_graph:
            entities = await self._extract_entities(memory.content)
            relationships = await self._extract_relationships(memory.content, entities)
            memory.metadata["entities"] = entities
            memory.metadata["relationships"] = relationships

        logger.debug(f"Cognified memory: {memory.id}")
        return memory

    async def memify(self, memory: Memory) -> Memory:
        """
        MEMIFY phase: Enrich memory with computed properties
        - Calculate relevance scores
        - Add derived relationships
        - Compress if needed
        """
        # Calculate initial relevance score based on type and priority
        base_score = {
            MemoryPriority.CRITICAL: 1.0,
            MemoryPriority.HIGH: 0.8,
            MemoryPriority.MEDIUM: 0.6,
            MemoryPriority.LOW: 0.4,
        }.get(memory.priority, 0.5)

        # Boost score for frequently accessed memories
        access_boost = min(memory.access_count * 0.05, 0.3)
        memory.relevance_score = min(base_score + access_boost, 1.0)

        # Compression for long content
        if self.config.enable_compression and len(memory.content) > 1000:
            memory.metadata["compressed"] = True
            memory.metadata["original_length"] = len(memory.content)

        logger.debug(f"Memified memory: {memory.id} (score: {memory.relevance_score:.2f})")
        return memory

    async def store(self, memory: Memory) -> bool:
        """
        LOAD phase: Persist memory to appropriate stores
        """
        success = True

        # Store in SQL (primary persistence)
        if self.sql_store and self.config.enable_sql:
            try:
                await self.sql_store.store(memory)
            except Exception as e:
                logger.error(f"SQL store failed: {e}")
                success = False

        # Store embedding in vector store
        if self.vector_store and self.config.enable_vector and memory.embedding:
            try:
                await self.vector_store.store(memory)
            except Exception as e:
                logger.error(f"Vector store failed: {e}")
                success = False

        # Store graph relationships
        if self.graph_store and self.config.enable_graph:
            try:
                await self.graph_store.store(memory)
            except Exception as e:
                logger.error(f"Graph store failed: {e}")
                success = False

        return success

    async def ingest(
        self,
        content: str,
        memory_type: MemoryType,
        **kwargs,
    ) -> Memory:
        """
        Complete ECL pipeline: Add -> Cognify -> Memify -> Store
        """
        memory = await self.add(content, memory_type, **kwargs)
        memory = await self.cognify(memory)
        memory = await self.memify(memory)
        await self.store(memory)
        return memory

    async def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        entity_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        include_graph_context: bool = True,
    ) -> List[Memory]:
        """
        SEARCH: Hybrid retrieval using vector similarity + graph traversal

        1. Vector search for semantic similarity
        2. Graph traversal for related context
        3. Merge and rank results
        """
        results: List[Memory] = []
        seen_ids = set()

        # Vector similarity search
        if self.vector_store and self.config.enable_vector:
            try:
                vector_results = await self.vector_store.retrieve(
                    query,
                    memory_type=memory_type,
                    limit=limit,
                )
                for mem in vector_results:
                    if mem.id not in seen_ids:
                        results.append(mem)
                        seen_ids.add(mem.id)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        # Graph context expansion
        if self.graph_store and self.config.enable_graph and include_graph_context:
            try:
                # Get related memories through graph relationships
                for mem in results[:5]:  # Expand top 5 results
                    related = await self._get_graph_context(mem)
                    for rel_mem in related:
                        if rel_mem.id not in seen_ids:
                            rel_mem.relevance_score *= 0.8  # Discount related
                            results.append(rel_mem)
                            seen_ids.add(rel_mem.id)
            except Exception as e:
                logger.warning(f"Graph expansion failed: {e}")

        # Filter by entity/session if specified
        if entity_id:
            results = [m for m in results if m.entity_id == entity_id]
        if session_id:
            results = [m for m in results if m.session_id == session_id]

        # Sort by relevance and limit
        results.sort(key=lambda m: m.relevance_score, reverse=True)
        return results[:limit]

    async def store_interaction(self, context: Any) -> None:
        """Store an agent interaction context as memories"""
        from .base import AgentContext

        if not isinstance(context, AgentContext):
            return

        # Store thoughts
        for thought in context.thoughts:
            await self.ingest(
                content=thought.content,
                memory_type=MemoryType.CONVERSATION,
                session_id=context.conversation_id,
                entity_id=context.user.user_id if context.user else None,
                metadata={"thought_type": thought.type.value},
            )

        # Store successful actions as patterns
        for action in context.actions:
            if action.result and not action.error:
                await self.ingest(
                    content=json.dumps({
                        "tool": action.tool_name,
                        "args": action.arguments,
                        "result_summary": str(action.result)[:500],
                    }),
                    memory_type=MemoryType.QUERY_PATTERN,
                    session_id=context.conversation_id,
                    priority=MemoryPriority.HIGH,
                )

    async def consolidate(self) -> int:
        """
        Consolidate memories: merge duplicates, prune old entries,
        recalculate relevance scores
        """
        async with self._consolidation_lock:
            consolidated_count = 0

            # Merge near-duplicate memories
            # Prune low-relevance, old memories
            # Recalculate scores based on access patterns

            logger.info(f"Consolidated {consolidated_count} memories")
            return consolidated_count

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        if asyncio.iscoroutinefunction(self.embedding_fn):
            return await self.embedding_fn(text)
        return self.embedding_fn(text)

    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        # Placeholder - implement with NER or LLM extraction
        return []

    async def _extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        # Placeholder - implement with relation extraction
        return []

    async def _get_graph_context(self, memory: Memory) -> List[Memory]:
        """Get related memories through graph traversal"""
        if not self.graph_store:
            return []

        # Traverse graph for related nodes
        return await self.graph_store.retrieve(
            str(memory.id),
            limit=5,
        )
