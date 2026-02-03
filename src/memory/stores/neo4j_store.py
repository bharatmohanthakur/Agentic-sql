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
    # Connection settings
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

    # Multi-tenancy / Organization
    namespace: Optional[str] = None  # Logical namespace (used as node label prefix)
    tenant_id: Optional[str] = None  # Tenant identifier for multi-tenant apps
    group_id: Optional[str] = None   # Group identifier (e.g., project, team)

    # Node configuration
    node_label: str = "Memory"  # Base label for memory nodes
    use_namespace_label: bool = True  # Add namespace as additional label

    # Connection pool settings
    max_connection_lifetime: int = 3600  # Max connection lifetime in seconds
    max_connection_pool_size: int = 100  # Max connections in pool
    connection_timeout: int = 30  # Connection timeout in seconds


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

    def _get_node_labels(self) -> str:
        """Get node labels including namespace if configured"""
        labels = [self.config.node_label]
        if self.config.namespace and self.config.use_namespace_label:
            labels.append(f"NS_{self.config.namespace}")
        return ":".join(labels)

    async def connect(self) -> None:
        """Connect to Neo4j"""
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout,
            )

            # Verify connection
            async with self._driver.session(database=self.config.database) as session:
                await session.run("RETURN 1")

            # Create indexes for better performance
            async with self._driver.session(database=self.config.database) as session:
                # Index on memory ID
                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (m:{self.config.node_label}) ON (m.id)"
                )
                # Index on namespace if using multi-tenancy
                if self.config.namespace:
                    await session.run(
                        f"CREATE INDEX IF NOT EXISTS FOR (m:{self.config.node_label}) ON (m.namespace)"
                    )
                # Index on tenant_id
                if self.config.tenant_id:
                    await session.run(
                        f"CREATE INDEX IF NOT EXISTS FOR (m:{self.config.node_label}) ON (m.tenant_id)"
                    )

            namespace_info = f" (namespace={self.config.namespace})" if self.config.namespace else ""
            logger.info(f"Connected to Neo4j at {self.config.uri}{namespace_info}")

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
            node_labels = self._get_node_labels()

            async with self._driver.session(database=self.config.database) as session:
                # Build query with dynamic labels
                query = f"""
                MERGE (m:{node_labels} {{id: $id}})
                SET m.content = $content,
                    m.type = $type,
                    m.priority = $priority,
                    m.entity_id = $entity_id,
                    m.process_id = $process_id,
                    m.session_id = $session_id,
                    m.created_at = $created_at,
                    m.access_count = $access_count,
                    m.relevance_score = $relevance_score,
                    m.namespace = $namespace,
                    m.tenant_id = $tenant_id,
                    m.group_id = $group_id
                RETURN m
                """

                await session.run(
                    query,
                    id=str(memory.id),
                    content=memory.content,
                    type=memory.type.value,
                    priority=memory.priority.value,
                    entity_id=memory.entity_id,
                    process_id=memory.process_id,
                    session_id=memory.session_id,
                    created_at=memory.created_at.isoformat(),
                    access_count=memory.access_count,
                    relevance_score=memory.relevance_score,
                    # Multi-tenancy fields
                    namespace=self.config.namespace,
                    tenant_id=self.config.tenant_id,
                    group_id=self.config.group_id,
                )

                # Create relationships from metadata
                if memory.metadata.get("relationships"):
                    for rel in memory.metadata["relationships"]:
                        rel_query = f"""
                        MATCH (m:{node_labels} {{id: $from_id}})
                        MERGE (t:{node_labels} {{id: $to_id}})
                        MERGE (m)-[:RELATES_TO {{type: $rel_type}}]->(t)
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
        namespace: Optional[str] = None,
        tenant_id: Optional[str] = None,
        group_id: Optional[str] = None,
        **kwargs,
    ) -> List["Memory"]:
        """
        Retrieve memories via graph traversal.

        Args:
            query: Search query (ID or content match)
            memory_type: Filter by memory type
            limit: Max results to return
            depth: Graph traversal depth (default 2)
            namespace: Filter by namespace (overrides config)
            tenant_id: Filter by tenant (overrides config)
            group_id: Filter by group (overrides config)
        """
        from ..manager import Memory, MemoryType, MemoryPriority

        try:
            node_labels = self._get_node_labels()

            # Use param or fall back to config
            effective_namespace = namespace or self.config.namespace
            effective_tenant = tenant_id or self.config.tenant_id
            effective_group = group_id or self.config.group_id

            # Build WHERE conditions
            where_conditions = ["(m.id = $query OR m.content CONTAINS $query)"]
            if effective_namespace:
                where_conditions.append("m.namespace = $namespace")
            if effective_tenant:
                where_conditions.append("m.tenant_id = $tenant_id")
            if effective_group:
                where_conditions.append("m.group_id = $group_id")
            if memory_type:
                where_conditions.append("m.type = $type")

            where_clause = " AND ".join(where_conditions)

            async with self._driver.session(database=self.config.database) as session:
                cypher = f"""
                MATCH (m:{node_labels})
                WHERE {where_clause}
                OPTIONAL MATCH (m)-[:RELATES_TO*1..{depth}]->(related:{node_labels})
                RETURN m, collect(DISTINCT related) as related
                LIMIT $limit
                """

                result = await session.run(
                    cypher,
                    query=query,
                    type=memory_type.value if memory_type else None,
                    namespace=effective_namespace,
                    tenant_id=effective_tenant,
                    group_id=effective_group,
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
                        process_id=node.get("process_id"),
                        session_id=node.get("session_id"),
                        metadata={
                            "related_count": len(record["related"]),
                            "namespace": node.get("namespace"),
                            "tenant_id": node.get("tenant_id"),
                            "group_id": node.get("group_id"),
                        },
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
            node_labels = self._get_node_labels()

            # Build delete query with namespace filter for safety
            where_conditions = ["m.id = $id"]
            if self.config.namespace:
                where_conditions.append("m.namespace = $namespace")

            where_clause = " AND ".join(where_conditions)

            async with self._driver.session(database=self.config.database) as session:
                await session.run(
                    f"MATCH (m:{node_labels}) WHERE {where_clause} DETACH DELETE m",
                    id=str(memory_id),
                    namespace=self.config.namespace,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Neo4j: {e}")
            return False

    async def update(self, memory: "Memory") -> bool:
        """Update memory node"""
        return await self.store(memory)

    async def count(
        self,
        namespace: Optional[str] = None,
        tenant_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> int:
        """
        Get total memory count.

        Args:
            namespace: Filter by namespace (overrides config)
            tenant_id: Filter by tenant (overrides config)
            group_id: Filter by group (overrides config)
        """
        try:
            node_labels = self._get_node_labels()

            # Use param or fall back to config
            effective_namespace = namespace or self.config.namespace
            effective_tenant = tenant_id or self.config.tenant_id
            effective_group = group_id or self.config.group_id

            # Build WHERE conditions
            where_conditions = []
            if effective_namespace:
                where_conditions.append("m.namespace = $namespace")
            if effective_tenant:
                where_conditions.append("m.tenant_id = $tenant_id")
            if effective_group:
                where_conditions.append("m.group_id = $group_id")

            where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(
                    f"MATCH (m:{node_labels}) {where_clause} RETURN count(m) as count",
                    namespace=effective_namespace,
                    tenant_id=effective_tenant,
                    group_id=effective_group,
                )
                record = await result.single()
                return record["count"] if record else 0
        except Exception:
            return 0
