"""
Knowledge Base - Unified knowledge storage and retrieval
Combines vector search, graph traversal, and semantic understanding
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class KnowledgeType(str, Enum):
    """Types of knowledge"""
    SCHEMA = "schema"  # Database schema
    QUERY_PATTERN = "query_pattern"  # SQL patterns
    BUSINESS_RULE = "business_rule"  # Domain rules
    TERMINOLOGY = "terminology"  # Business terms
    RELATIONSHIP = "relationship"  # Entity relationships
    METRIC = "metric"  # Business metrics
    DOCUMENTATION = "documentation"  # External docs


@dataclass
class KnowledgeItem:
    """A single knowledge item"""
    id: str = field(default_factory=lambda: str(uuid4()))
    type: KnowledgeType = KnowledgeType.SCHEMA

    # Content
    content: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)

    # Embeddings for similarity search
    embedding: Optional[List[float]] = None

    # Graph connections
    related_to: List[str] = field(default_factory=list)  # Other item IDs
    tags: List[str] = field(default_factory=list)

    # Metadata
    source: str = ""
    confidence: float = 1.0
    use_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "structured_data": self.structured_data,
            "tags": self.tags,
            "confidence": self.confidence,
        }


class SemanticIndex:
    """
    Semantic search index using embeddings
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable] = None,
        similarity_threshold: float = 0.7,
    ):
        self.embedding_fn = embedding_fn
        self.threshold = similarity_threshold
        self._items: Dict[str, KnowledgeItem] = {}
        self._embeddings: Dict[str, List[float]] = {}

    async def add(self, item: KnowledgeItem) -> None:
        """Add item to index"""
        self._items[item.id] = item

        if self.embedding_fn and not item.embedding:
            item.embedding = await self._generate_embedding(item.content)

        if item.embedding:
            self._embeddings[item.id] = item.embedding

    async def search(
        self,
        query: str,
        limit: int = 10,
        knowledge_type: Optional[KnowledgeType] = None,
    ) -> List[KnowledgeItem]:
        """Search for similar items"""
        if not self.embedding_fn:
            # Fallback to keyword search
            return self._keyword_search(query, limit, knowledge_type)

        query_embedding = await self._generate_embedding(query)

        # Calculate similarities
        scored = []
        for item_id, embedding in self._embeddings.items():
            item = self._items[item_id]

            if knowledge_type and item.type != knowledge_type:
                continue

            similarity = self._cosine_similarity(query_embedding, embedding)

            if similarity >= self.threshold:
                scored.append((similarity, item))

        # Sort by similarity
        scored.sort(key=lambda x: x[0], reverse=True)

        # Update use counts
        results = []
        for _, item in scored[:limit]:
            item.use_count += 1
            results.append(item)

        return results

    def _keyword_search(
        self,
        query: str,
        limit: int,
        knowledge_type: Optional[KnowledgeType],
    ) -> List[KnowledgeItem]:
        """Simple keyword-based search"""
        query_words = set(query.lower().split())
        scored = []

        for item in self._items.values():
            if knowledge_type and item.type != knowledge_type:
                continue

            content_words = set(item.content.lower().split())
            overlap = len(query_words & content_words)

            if overlap > 0:
                score = overlap * item.confidence
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if asyncio.iscoroutinefunction(self.embedding_fn):
            return await self.embedding_fn(text)
        return self.embedding_fn(text)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


class KnowledgeGraph:
    """
    Graph-based knowledge representation

    Enables:
    - Relationship traversal
    - Path finding
    - Pattern matching
    """

    def __init__(self):
        self._nodes: Dict[str, KnowledgeItem] = {}
        self._edges: Dict[str, List[Tuple[str, str]]] = {}  # node -> [(neighbor, rel_type)]
        self._reverse_edges: Dict[str, List[Tuple[str, str]]] = {}

    def add_node(self, item: KnowledgeItem) -> None:
        """Add a node to the graph"""
        self._nodes[item.id] = item
        self._edges[item.id] = []
        self._reverse_edges[item.id] = []

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relationship: str = "related_to",
    ) -> None:
        """Add an edge between nodes"""
        if from_id in self._nodes and to_id in self._nodes:
            self._edges[from_id].append((to_id, relationship))
            self._reverse_edges[to_id].append((from_id, relationship))

    def get_neighbors(
        self,
        node_id: str,
        relationship: Optional[str] = None,
        depth: int = 1,
    ) -> List[KnowledgeItem]:
        """Get neighboring nodes"""
        if node_id not in self._nodes:
            return []

        visited = {node_id}
        current_level = [node_id]
        results = []

        for _ in range(depth):
            next_level = []

            for nid in current_level:
                for neighbor_id, rel in self._edges.get(nid, []):
                    if neighbor_id not in visited:
                        if relationship is None or rel == relationship:
                            visited.add(neighbor_id)
                            next_level.append(neighbor_id)
                            results.append(self._nodes[neighbor_id])

            current_level = next_level

        return results

    def find_path(
        self,
        from_id: str,
        to_id: str,
    ) -> Optional[List[str]]:
        """Find path between two nodes using BFS"""
        if from_id not in self._nodes or to_id not in self._nodes:
            return None

        visited = {from_id}
        queue = [(from_id, [from_id])]

        while queue:
            current, path = queue.pop(0)

            for neighbor_id, _ in self._edges.get(current, []):
                if neighbor_id == to_id:
                    return path + [neighbor_id]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def get_subgraph(
        self,
        center_id: str,
        radius: int = 2,
    ) -> Dict[str, Any]:
        """Get subgraph around a node"""
        nodes = [self._nodes[center_id]] if center_id in self._nodes else []
        nodes.extend(self.get_neighbors(center_id, depth=radius))

        node_ids = {n.id for n in nodes}

        edges = []
        for node in nodes:
            for neighbor_id, rel in self._edges.get(node.id, []):
                if neighbor_id in node_ids:
                    edges.append({
                        "from": node.id,
                        "to": neighbor_id,
                        "relationship": rel,
                    })

        return {
            "nodes": [n.to_dict() for n in nodes],
            "edges": edges,
        }


class KnowledgeBase:
    """
    Unified knowledge base combining semantic search and graph

    Features:
    - Multi-type knowledge storage
    - Semantic similarity search
    - Graph-based reasoning
    - Automatic knowledge extraction
    - Continuous learning
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable] = None,
        llm_client: Optional[Any] = None,
    ):
        self.semantic_index = SemanticIndex(embedding_fn)
        self.graph = KnowledgeGraph()
        self.llm = llm_client
        self._type_stats: Dict[KnowledgeType, int] = {}

    async def add(
        self,
        content: str,
        knowledge_type: KnowledgeType,
        structured_data: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        related_to: Optional[List[str]] = None,
        source: str = "manual",
    ) -> KnowledgeItem:
        """Add knowledge to the base"""
        item = KnowledgeItem(
            type=knowledge_type,
            content=content,
            structured_data=structured_data or {},
            tags=tags or [],
            related_to=related_to or [],
            source=source,
        )

        # Add to semantic index
        await self.semantic_index.add(item)

        # Add to graph
        self.graph.add_node(item)

        # Create edges for relationships
        for related_id in item.related_to:
            self.graph.add_edge(item.id, related_id)

        # Update stats
        self._type_stats[knowledge_type] = self._type_stats.get(knowledge_type, 0) + 1

        logger.debug(f"Added knowledge: {item.id} [{knowledge_type.value}]")

        return item

    async def search(
        self,
        query: str,
        knowledge_type: Optional[KnowledgeType] = None,
        include_related: bool = True,
        limit: int = 10,
    ) -> List[KnowledgeItem]:
        """
        Search for relevant knowledge

        Combines semantic search with graph traversal
        """
        # Semantic search
        results = await self.semantic_index.search(
            query,
            limit=limit,
            knowledge_type=knowledge_type,
        )

        if not include_related:
            return results

        # Expand with graph neighbors
        expanded = list(results)
        seen_ids = {r.id for r in results}

        for result in results[:5]:  # Expand top 5
            neighbors = self.graph.get_neighbors(result.id, depth=1)

            for neighbor in neighbors:
                if neighbor.id not in seen_ids:
                    neighbor.confidence *= 0.8  # Discount related items
                    expanded.append(neighbor)
                    seen_ids.add(neighbor.id)

        # Sort by confidence and limit
        expanded.sort(key=lambda x: x.confidence, reverse=True)

        return expanded[:limit]

    async def ingest_schema(
        self,
        schema_profiles: Dict[str, Any],
    ) -> int:
        """Ingest database schema as knowledge"""
        count = 0

        for table_name, profile in schema_profiles.items():
            # Add table
            table_item = await self.add(
                content=f"Table {table_name}: {profile.description if hasattr(profile, 'description') else ''}",
                knowledge_type=KnowledgeType.SCHEMA,
                structured_data={
                    "table_name": table_name,
                    "table_type": profile.table_type.value if hasattr(profile, 'table_type') else "entity",
                    "row_count": profile.row_count if hasattr(profile, 'row_count') else 0,
                },
                tags=["table", table_name],
            )
            count += 1

            # Add columns
            columns = profile.columns if hasattr(profile, 'columns') else []
            for col in columns:
                col_name = col.name if hasattr(col, 'name') else str(col)
                col_item = await self.add(
                    content=f"Column {table_name}.{col_name}",
                    knowledge_type=KnowledgeType.SCHEMA,
                    structured_data={
                        "table_name": table_name,
                        "column_name": col_name,
                        "data_type": col.data_type if hasattr(col, 'data_type') else "unknown",
                    },
                    tags=["column", table_name, col_name],
                    related_to=[table_item.id],
                )
                count += 1

            # Add relationships
            foreign_keys = profile.foreign_keys if hasattr(profile, 'foreign_keys') else []
            for fk in foreign_keys:
                await self.add(
                    content=f"Relationship: {table_name}.{fk['column']} -> {fk['references']}",
                    knowledge_type=KnowledgeType.RELATIONSHIP,
                    structured_data=fk,
                    tags=["foreign_key", table_name],
                    related_to=[table_item.id],
                )
                count += 1

        logger.info(f"Ingested {count} schema items")
        return count

    async def ingest_query_patterns(
        self,
        patterns: List[Dict[str, str]],
    ) -> int:
        """Ingest successful query patterns"""
        count = 0

        for pattern in patterns:
            await self.add(
                content=f"Q: {pattern.get('question', '')}\nSQL: {pattern.get('sql', '')}",
                knowledge_type=KnowledgeType.QUERY_PATTERN,
                structured_data=pattern,
                tags=pattern.get('tables', []),
                source="learned",
            )
            count += 1

        logger.info(f"Ingested {count} query patterns")
        return count

    async def ingest_business_rules(
        self,
        rules: List[Dict[str, Any]],
    ) -> int:
        """Ingest business rules and terminology"""
        count = 0

        for rule in rules:
            await self.add(
                content=rule.get("description", ""),
                knowledge_type=KnowledgeType.BUSINESS_RULE,
                structured_data=rule,
                tags=rule.get("tags", []),
                source="documentation",
            )
            count += 1

        logger.info(f"Ingested {count} business rules")
        return count

    async def extract_knowledge_from_text(
        self,
        text: str,
        source: str = "document",
    ) -> int:
        """Extract and store knowledge from unstructured text"""
        if not self.llm:
            return 0

        prompt = f"""
        Extract structured knowledge from this text.
        Identify:
        1. Business terms and their definitions
        2. Data relationships
        3. Business rules
        4. Metrics and KPIs

        Text:
        {text}

        Return as JSON array with objects containing:
        - type: "terminology" | "relationship" | "rule" | "metric"
        - content: description
        - entities: list of related entities

        JSON:
        """

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=1000)

            # Parse JSON
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return 0

            items = json.loads(json_match.group())
            count = 0

            for item in items:
                type_map = {
                    "terminology": KnowledgeType.TERMINOLOGY,
                    "relationship": KnowledgeType.RELATIONSHIP,
                    "rule": KnowledgeType.BUSINESS_RULE,
                    "metric": KnowledgeType.METRIC,
                }

                knowledge_type = type_map.get(
                    item.get("type", ""),
                    KnowledgeType.DOCUMENTATION,
                )

                await self.add(
                    content=item.get("content", ""),
                    knowledge_type=knowledge_type,
                    structured_data=item,
                    tags=item.get("entities", []),
                    source=source,
                )
                count += 1

            return count

        except Exception as e:
            logger.warning(f"Knowledge extraction failed: {e}")
            return 0

    def get_context_for_query(
        self,
        tables: List[str],
    ) -> str:
        """Get relevant context for specific tables"""
        context_parts = []

        for table in tables:
            # Get table knowledge
            table_items = [
                item for item in self.semantic_index._items.values()
                if table in item.tags and item.type == KnowledgeType.SCHEMA
            ]

            for item in table_items:
                context_parts.append(item.content)

            # Get related business rules
            rule_items = [
                item for item in self.semantic_index._items.values()
                if table in item.tags and item.type == KnowledgeType.BUSINESS_RULE
            ]

            for item in rule_items:
                context_parts.append(f"Rule: {item.content}")

        return "\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            "total_items": len(self.semantic_index._items),
            "by_type": dict(self._type_stats),
            "graph_nodes": len(self.graph._nodes),
            "graph_edges": sum(len(e) for e in self.graph._edges.values()),
        }
