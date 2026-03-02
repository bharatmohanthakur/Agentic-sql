"""
GraphRAG - Combines graph traversal with LLM reasoning.

Retrieval-Augmented Generation using the knowledge graph as the retrieval layer.
Provides richer context than flat vector search by including relationship semantics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .graph_model import GraphNode, GraphPath
from .neo4j_knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG"""
    # Retrieval settings
    max_context_nodes: int = 20
    max_traversal_depth: int = 3
    vector_search_top_k: int = 10
    min_similarity_score: float = 0.5

    # Context building
    include_node_properties: bool = True
    include_relationship_types: bool = True
    include_paths: bool = True
    max_context_tokens: int = 4000

    # Community detection
    enable_community_context: bool = False


class RetrievedContext(BaseModel):
    """Context retrieved from the knowledge graph"""
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    paths: List[str] = Field(default_factory=list)  # Human-readable path descriptions
    community_summaries: List[str] = Field(default_factory=list)
    text_context: str = ""  # Formatted text for LLM consumption
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphRAG:
    """
    GraphRAG retriever that combines:
    1. Vector similarity search (find semantically similar nodes)
    2. Graph traversal (expand context through relationships)
    3. Context formatting (prepare graph context for LLM)

    This provides richer context than flat RAG because it includes
    the relationships between entities, not just similar documents.
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        embedding_fn: Any = None,
        config: Optional[GraphRAGConfig] = None,
    ):
        self.kg = knowledge_graph
        self.embedding_fn = embedding_fn  # async fn(text) -> List[float]
        self.config = config or GraphRAGConfig()

    async def retrieve(
        self,
        query: str,
        node_labels: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievedContext:
        """
        Retrieve relevant context from the knowledge graph for a query.

        Steps:
        1. Generate query embedding
        2. Vector search for similar nodes
        3. Expand context through graph traversal
        4. Format context for LLM consumption
        """
        context = RetrievedContext()

        # Step 1: Vector search if embedding function available
        seed_nodes: List[Tuple[GraphNode, float]] = []
        if self.embedding_fn:
            query_embedding = await self.embedding_fn(query)

            for label in (node_labels or ["Node"]):
                results = await self.kg.vector_search(
                    label=label,
                    query_embedding=query_embedding,
                    top_k=self.config.vector_search_top_k,
                    min_score=self.config.min_similarity_score,
                )
                seed_nodes.extend(results)

        # Step 2: Text-based search fallback
        if not seed_nodes and node_labels:
            for label in node_labels:
                nodes = await self.kg.find_nodes(
                    label=label,
                    limit=self.config.vector_search_top_k,
                )
                seed_nodes.extend((n, 1.0) for n in nodes)

        # Step 3: Expand context through graph traversal
        expanded_nodes = set()
        expanded_edges = []
        paths = []

        for node, score in seed_nodes[:self.config.max_context_nodes]:
            expanded_nodes.add(node.id)
            context.nodes.append({
                "id": node.id,
                "labels": node.labels,
                "properties": node.properties,
                "similarity_score": score,
            })

            # Traverse neighbors
            neighbor_paths = await self.kg.find_neighbors(
                node_id=node.id,
                depth=self.config.max_traversal_depth,
            )

            for path in neighbor_paths:
                if path.length > 0:
                    paths.append(path.to_text())
                    for n in path.nodes:
                        if n.id not in expanded_nodes:
                            expanded_nodes.add(n.id)
                            context.nodes.append({
                                "id": n.id,
                                "labels": n.labels,
                                "properties": n.properties,
                            })
                    for e in path.edges:
                        context.edges.append({
                            "source": e.source_id,
                            "target": e.target_id,
                            "type": e.relationship_type,
                        })

        context.paths = paths[:20]  # Limit path descriptions

        # Step 4: Format context for LLM
        context.text_context = self._format_context(context)
        context.metadata = {
            "total_nodes": len(context.nodes),
            "total_edges": len(context.edges),
            "total_paths": len(context.paths),
            "seed_nodes": len(seed_nodes),
        }

        return context

    def _format_context(self, context: RetrievedContext) -> str:
        """Format graph context as text for LLM consumption"""
        parts = ["## Knowledge Graph Context\n"]

        # Entities
        if context.nodes:
            parts.append("### Entities:")
            for node in context.nodes[:self.config.max_context_nodes]:
                labels = ", ".join(node.get("labels", []))
                name = node.get("properties", {}).get("name", node["id"])
                score = node.get("similarity_score", "")
                score_str = f" (relevance: {score:.2f})" if score else ""
                parts.append(f"- [{labels}] {name}{score_str}")

                if self.config.include_node_properties:
                    for k, v in node.get("properties", {}).items():
                        if k != "name" and not k.startswith("_"):
                            parts.append(f"  - {k}: {v}")

        # Relationships
        if context.edges and self.config.include_relationship_types:
            parts.append("\n### Relationships:")
            seen = set()
            for edge in context.edges:
                key = (edge["source"], edge["type"], edge["target"])
                if key not in seen:
                    seen.add(key)
                    parts.append(
                        f"- ({edge['source']})-[:{edge['type']}]->({edge['target']})"
                    )

        # Paths
        if context.paths and self.config.include_paths:
            parts.append("\n### Traversal Paths:")
            for path_text in context.paths[:10]:
                parts.append(f"- {path_text}")

        return "\n".join(parts)
