"""
Data Lineage Tracker - Traces data from dimensional model back to ontology.

Maintains lineage relationships across all 4 semantic layers:
Ontology → Knowledge Graph → Canonical → Dimensional
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LineageEdge(BaseModel):
    """A single lineage edge connecting two data elements"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_fqn: str          # Fully qualified name: schema.table.column
    target_fqn: str
    transformation: str = ""  # SQL expression or description
    layer_from: str = ""     # ontology, knowledge_graph, canonical, dimensional
    layer_to: str = ""
    scd_type: Optional[str] = None
    business_rule: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LineagePath(BaseModel):
    """A complete lineage path from source to target"""
    edges: List[LineageEdge] = Field(default_factory=list)
    source_layer: str = ""
    target_layer: str = ""
    total_hops: int = 0
    semantic_loss: List[str] = Field(default_factory=list)  # What meaning was lost

    def to_text(self) -> str:
        """Human-readable lineage path"""
        parts = []
        for edge in self.edges:
            parts.append(
                f"[{edge.layer_from}] {edge.source_fqn} "
                f"→ [{edge.layer_to}] {edge.target_fqn}"
            )
            if edge.transformation:
                parts.append(f"  Transform: {edge.transformation}")
        return "\n".join(parts)


class LineageTracker:
    """
    Tracks data lineage across the 4 semantic layers.

    Stores lineage edges in a graph (Neo4j) and provides
    traversal for impact analysis and provenance queries.

    Lineage flow:
    Ontology facts → Knowledge graph nodes → Canonical entities → Dimensional columns
    """

    def __init__(self, neo4j_driver: Any = None):
        self._driver = neo4j_driver
        self._edges: List[LineageEdge] = []  # In-memory fallback

    async def add_edge(self, edge: LineageEdge) -> None:
        """Add a lineage edge"""
        if self._driver:
            await self._store_edge_neo4j(edge)
        else:
            self._edges.append(edge)

    async def trace_backward(
        self,
        target_fqn: str,
        max_depth: int = 10,
    ) -> LineagePath:
        """
        Trace lineage backward from a dimensional column to its ontology source.

        Walks the lineage graph from target back to source.
        """
        if self._driver:
            return await self._trace_neo4j(target_fqn, "backward", max_depth)

        # In-memory fallback
        path = LineagePath(target_layer="dimensional")
        visited: Set[str] = set()
        current = target_fqn

        for _ in range(max_depth):
            if current in visited:
                break
            visited.add(current)

            edge = next(
                (e for e in self._edges if e.target_fqn == current),
                None,
            )
            if not edge:
                break

            path.edges.insert(0, edge)
            current = edge.source_fqn

        path.total_hops = len(path.edges)
        if path.edges:
            path.source_layer = path.edges[0].layer_from

        return path

    async def trace_forward(
        self,
        source_fqn: str,
        max_depth: int = 10,
    ) -> List[LineagePath]:
        """
        Trace lineage forward from a source to all downstream targets.

        Used for impact analysis: "If this source changes, what breaks?"
        """
        paths = []
        visited: Set[str] = set()

        async def _traverse(fqn: str, current_path: List[LineageEdge], depth: int):
            if depth > max_depth or fqn in visited:
                return
            visited.add(fqn)

            downstream = [e for e in self._edges if e.source_fqn == fqn]
            if not downstream:
                if current_path:
                    path = LineagePath(
                        edges=list(current_path),
                        source_layer=current_path[0].layer_from,
                        target_layer=current_path[-1].layer_to,
                        total_hops=len(current_path),
                    )
                    paths.append(path)
                return

            for edge in downstream:
                current_path.append(edge)
                await _traverse(edge.target_fqn, current_path, depth + 1)
                current_path.pop()

        await _traverse(source_fqn, [], 0)
        return paths

    async def get_semantic_context(
        self,
        dimensional_column: str,
    ) -> Dict[str, Any]:
        """
        Get the full semantic context for a dimensional column.

        Traces back to ontology and returns what the column "means"
        in business terms.
        """
        path = await self.trace_backward(dimensional_column)

        context = {
            "column": dimensional_column,
            "lineage_depth": path.total_hops,
            "source_layer": path.source_layer,
            "lineage_path": path.to_text(),
            "transformations": [
                e.transformation for e in path.edges if e.transformation
            ],
            "business_rules": [
                e.business_rule for e in path.edges if e.business_rule
            ],
        }

        # Extract ontology context from the first edge
        if path.edges:
            first_edge = path.edges[0]
            context["ontology_source"] = first_edge.source_fqn
            context["ontology_layer"] = first_edge.layer_from

        return context

    async def _store_edge_neo4j(self, edge: LineageEdge) -> None:
        """Store lineage edge in Neo4j"""
        try:
            async with self._driver.session() as session:
                await session.run(
                    """
                    MERGE (s:DataElement {fqn: $source_fqn})
                    SET s.layer = $layer_from
                    MERGE (t:DataElement {fqn: $target_fqn})
                    SET t.layer = $layer_to
                    MERGE (s)-[r:TRANSFORMS_TO]->(t)
                    SET r.id = $id,
                        r.transformation = $transformation,
                        r.scd_type = $scd_type,
                        r.business_rule = $business_rule,
                        r.created_at = $created_at
                    """,
                    source_fqn=edge.source_fqn,
                    target_fqn=edge.target_fqn,
                    layer_from=edge.layer_from,
                    layer_to=edge.layer_to,
                    id=edge.id,
                    transformation=edge.transformation,
                    scd_type=edge.scd_type,
                    business_rule=edge.business_rule,
                    created_at=edge.created_at.isoformat(),
                )
        except Exception as e:
            logger.error(f"Failed to store lineage edge: {e}")

    async def _trace_neo4j(
        self,
        fqn: str,
        direction: str,
        max_depth: int,
    ) -> LineagePath:
        """Trace lineage using Neo4j graph traversal"""
        try:
            if direction == "backward":
                query = f"""
                MATCH path = (source)-[:TRANSFORMS_TO*1..{max_depth}]->(target {{fqn: $fqn}})
                RETURN path
                ORDER BY length(path) ASC
                LIMIT 1
                """
            else:
                query = f"""
                MATCH path = (source {{fqn: $fqn}})-[:TRANSFORMS_TO*1..{max_depth}]->(target)
                RETURN path
                LIMIT 10
                """

            async with self._driver.session() as session:
                result = await session.run(query, fqn=fqn)
                record = await result.single()

                if not record:
                    return LineagePath()

                neo4j_path = record["path"]
                edges = []
                for rel in neo4j_path.relationships:
                    edges.append(LineageEdge(
                        source_fqn=rel.start_node.get("fqn", ""),
                        target_fqn=rel.end_node.get("fqn", ""),
                        transformation=rel.get("transformation", ""),
                        layer_from=rel.start_node.get("layer", ""),
                        layer_to=rel.end_node.get("layer", ""),
                        scd_type=rel.get("scd_type"),
                        business_rule=rel.get("business_rule", ""),
                    ))

                return LineagePath(
                    edges=edges,
                    total_hops=len(edges),
                    source_layer=edges[0].layer_from if edges else "",
                    target_layer=edges[-1].layer_to if edges else "",
                )

        except Exception as e:
            logger.error(f"Neo4j lineage trace failed: {e}")
            return LineagePath()
