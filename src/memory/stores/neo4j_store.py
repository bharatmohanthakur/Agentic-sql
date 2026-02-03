"""
Neo4j Graph Store Implementation

Neo4j for storing entity relationships and graph traversal.
https://neo4j.com/

Installation:
    pip install neo4j

Usage:
    from memory.stores import Neo4jMemoryStore

    store = Neo4jMemoryStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
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
class Neo4jConfig:
    """Configuration for Neo4j connection"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class Neo4jMemoryStore(MemoryStore):
    """
    Neo4j Graph Store for relationship-based memory.

    Features:
    - Entity relationships
    - Graph traversal for context expansion
    - Cypher query support
    """

    def __init__(self, config: Neo4jConfig):
        self.config = config
        self._driver = None

    async def connect(self) -> None:
        """Connect to Neo4j"""
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
            )

            # Verify connection
            async with self._driver.session(database=self.config.database) as session:
                await session.run("RETURN 1")

            logger.info(f"Connected to Neo4j at {self.config.uri}")

        except ImportError:
            raise ImportError(
                "neo4j not installed. Install with: pip install neo4j"
            )

    async def disconnect(self) -> None:
        """Close Neo4j connection"""
        if self._driver:
            await self._driver.close()

    async def store(self, memory: "Memory") -> bool:
        """Store memory as a node with relationships"""
        try:
            async with self._driver.session(database=self.config.database) as session:
                query = """
                MERGE (m:Memory {id: $id})
                SET m.content = $content,
                    m.type = $type,
                    m.priority = $priority,
                    m.entity_id = $entity_id,
                    m.session_id = $session_id,
                    m.created_at = $created_at,
                    m.access_count = $access_count,
                    m.relevance_score = $relevance_score
                RETURN m
                """

                await session.run(
                    query,
                    id=str(memory.id),
                    content=memory.content,
                    type=memory.type.value,
                    priority=memory.priority.value,
                    entity_id=memory.entity_id,
                    session_id=memory.session_id,
                    created_at=memory.created_at.isoformat(),
                    access_count=memory.access_count,
                    relevance_score=memory.relevance_score,
                )

                # Create relationships from metadata
                if memory.metadata.get("relationships"):
                    for rel in memory.metadata["relationships"]:
                        rel_query = """
                        MATCH (m:Memory {id: $from_id})
                        MERGE (t:Memory {id: $to_id})
                        MERGE (m)-[:RELATES_TO {type: $rel_type}]->(t)
                        """
                        await session.run(
                            rel_query,
                            from_id=str(memory.id),
                            to_id=rel.get("target_id"),
                            rel_type=rel.get("type", "related"),
                        )

            return True

        except Exception as e:
            logger.error(f"Failed to store memory in Neo4j: {e}")
            return False

    async def retrieve(
        self,
        query: str,
        memory_type: Optional["MemoryType"] = None,
        limit: int = 10,
        depth: int = 2,
        **kwargs,
    ) -> List["Memory"]:
        """Retrieve memories via graph traversal"""
        from ..manager import Memory, MemoryType, MemoryPriority

        try:
            async with self._driver.session(database=self.config.database) as session:
                # Find memories and their related nodes
                cypher = """
                MATCH (m:Memory)
                WHERE m.id = $query OR m.content CONTAINS $query
                OPTIONAL MATCH (m)-[:RELATES_TO*1..2]->(related:Memory)
                RETURN m, collect(DISTINCT related) as related
                LIMIT $limit
                """

                if memory_type:
                    cypher = """
                    MATCH (m:Memory {type: $type})
                    WHERE m.id = $query OR m.content CONTAINS $query
                    OPTIONAL MATCH (m)-[:RELATES_TO*1..2]->(related:Memory)
                    RETURN m, collect(DISTINCT related) as related
                    LIMIT $limit
                    """

                result = await session.run(
                    cypher,
                    query=query,
                    type=memory_type.value if memory_type else None,
                    limit=limit,
                )

                memories = []
                async for record in result:
                    node = record["m"]
                    memories.append(Memory(
                        id=UUID(node["id"]),
                        content=node.get("content", ""),
                        type=MemoryType(node.get("type", "semantic")),
                        priority=MemoryPriority(node.get("priority", "medium")),
                        entity_id=node.get("entity_id"),
                        session_id=node.get("session_id"),
                        metadata={"related_count": len(record["related"])},
                        created_at=datetime.fromisoformat(node.get("created_at", datetime.utcnow().isoformat())),
                        access_count=node.get("access_count", 0),
                        relevance_score=node.get("relevance_score", 0.5),
                        embedding=None,
                    ))

                return memories

        except Exception as e:
            logger.error(f"Failed to search Neo4j: {e}")
            return []

    async def delete(self, memory_id: UUID) -> bool:
        """Delete memory and its relationships"""
        try:
            async with self._driver.session(database=self.config.database) as session:
                await session.run(
                    "MATCH (m:Memory {id: $id}) DETACH DELETE m",
                    id=str(memory_id),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Neo4j: {e}")
            return False

    async def update(self, memory: "Memory") -> bool:
        """Update memory node"""
        return await self.store(memory)

    async def count(self) -> int:
        """Get total memory count"""
        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run("MATCH (m:Memory) RETURN count(m) as count")
                record = await result.single()
                return record["count"] if record else 0
        except Exception:
            return 0
