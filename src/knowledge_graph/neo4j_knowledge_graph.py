"""
Neo4j Knowledge Graph - Operations for building and querying the knowledge graph.

Bridges between ontology facts (Layer 1) and queryable graph structures.
Supports Cypher query generation, graph traversal, and vector search.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .graph_model import GraphEdge, GraphNode, GraphPath, GraphSchema

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGraphConfig:
    """Configuration for Neo4j Knowledge Graph"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

    # Vector index settings
    enable_vector_index: bool = True
    vector_dimensions: int = 1536  # AWS Titan embedding dimensions
    similarity_function: str = "cosine"

    # Performance settings
    max_connection_pool_size: int = 50
    connection_timeout: int = 30

    # Namespace
    namespace: Optional[str] = None


class KnowledgeGraph:
    """
    Neo4j Knowledge Graph for Layer 2 of the Semantic Space Funnel.

    Provides:
    - Node/edge CRUD operations
    - Cypher query execution
    - Graph traversal (BFS/DFS, shortest path)
    - Vector similarity search on node embeddings
    - Schema management
    - Bulk import from ontology facts
    """

    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self._driver = None
        self._schema: Optional[GraphSchema] = None

    async def connect(self) -> None:
        """Initialize Neo4j connection and set up indexes"""
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout,
            )

            async with self._driver.session(database=self.config.database) as session:
                await session.run("RETURN 1")

            logger.info(f"Knowledge graph connected to Neo4j at {self.config.uri}")

        except ImportError:
            raise ImportError("neo4j required: pip install neo4j")

    async def disconnect(self) -> None:
        """Close Neo4j connection"""
        if self._driver:
            await self._driver.close()

    async def set_schema(self, schema: GraphSchema) -> None:
        """Apply a graph schema, creating indexes and constraints"""
        self._schema = schema

        async with self._driver.session(database=self.config.database) as session:
            # Create indexes for each node label
            for label, props in schema.node_labels.items():
                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.id)"
                )
                # Create unique constraints
                for prop in schema.unique_properties.get(label, []):
                    await session.run(
                        f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) "
                        f"REQUIRE n.{prop} IS UNIQUE"
                    )

            # Create vector index if enabled
            if self.config.enable_vector_index:
                for label in schema.node_labels:
                    try:
                        await session.run(
                            f"""
                            CREATE VECTOR INDEX IF NOT EXISTS {label.lower()}_embedding_idx
                            FOR (n:{label})
                            ON (n.embedding)
                            OPTIONS {{
                                indexConfig: {{
                                    `vector.dimensions`: $dims,
                                    `vector.similarity_function`: $sim
                                }}
                            }}
                            """,
                            dims=self.config.vector_dimensions,
                            sim=self.config.similarity_function,
                        )
                    except Exception as e:
                        logger.warning(f"Could not create vector index for {label}: {e}")

        logger.info(f"Applied schema '{schema.name}' with {len(schema.node_labels)} labels")

    # ── Node Operations ──

    async def create_node(self, node: GraphNode) -> bool:
        """Create or update a node in the knowledge graph"""
        try:
            labels = ":".join(node.labels)
            async with self._driver.session(database=self.config.database) as session:
                props = {**node.properties, "id": node.id}
                if node.source_ontology_type:
                    props["_ontology_type"] = node.source_ontology_type

                set_clause = ", ".join(f"n.{k} = ${k}" for k in props)
                params = {**props}

                query = f"MERGE (n:{labels} {{id: $id}}) SET {set_clause}"

                if node.embedding:
                    query += ", n.embedding = $embedding"
                    params["embedding"] = node.embedding

                await session.run(query, **params)
            return True
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            return False

    async def create_edge(self, edge: GraphEdge) -> bool:
        """Create a relationship between two nodes"""
        try:
            async with self._driver.session(database=self.config.database) as session:
                props_clause = ""
                if edge.properties:
                    prop_str = ", ".join(f"{k}: ${k}" for k in edge.properties)
                    props_clause = f" {{{prop_str}}}"

                query = f"""
                MATCH (a {{id: $source_id}})
                MATCH (b {{id: $target_id}})
                MERGE (a)-[r:{edge.relationship_type}{props_clause}]->(b)
                SET r.id = $edge_id, r.weight = $weight
                """
                if edge.source_fact_type:
                    query += ", r._fact_type = $fact_type"

                await session.run(
                    query,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_id=edge.id,
                    weight=edge.weight,
                    fact_type=edge.source_fact_type,
                    **edge.properties,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to create edge: {e}")
            return False

    # ── Query Operations ──

    async def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a raw Cypher query"""
        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(query, **(parameters or {}))
                records = []
                async for record in result:
                    records.append(dict(record))
                return records
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            return []

    async def find_nodes(
        self,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[GraphNode]:
        """Find nodes by label and optional property filters"""
        where_clause = ""
        params: Dict[str, Any] = {"limit": limit}

        if properties:
            conditions = []
            for i, (k, v) in enumerate(properties.items()):
                param_name = f"p{i}"
                conditions.append(f"n.{k} = ${param_name}")
                params[param_name] = v
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        MATCH (n:{label})
        {where_clause}
        RETURN n
        LIMIT $limit
        """

        records = await self.execute_cypher(query, params)
        return [self._record_to_node(r["n"], [label]) for r in records]

    async def find_neighbors(
        self,
        node_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",
        depth: int = 1,
    ) -> List[GraphPath]:
        """Find neighbor nodes within N hops"""
        rel_filter = f":{relationship_type}" if relationship_type else ""

        if direction == "outgoing":
            pattern = f"(n)-[r{rel_filter}*1..{depth}]->(m)"
        elif direction == "incoming":
            pattern = f"(n)<-[r{rel_filter}*1..{depth}]-(m)"
        else:
            pattern = f"(n)-[r{rel_filter}*1..{depth}]-(m)"

        query = f"""
        MATCH p = {pattern}
        WHERE n.id = $node_id
        RETURN p
        LIMIT 50
        """

        records = await self.execute_cypher(query, {"node_id": node_id})
        paths = []
        for record in records:
            path = self._record_to_path(record.get("p"))
            if path:
                paths.append(path)
        return paths

    async def shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 10,
    ) -> Optional[GraphPath]:
        """Find shortest path between two nodes"""
        query = f"""
        MATCH p = shortestPath(
            (a {{id: $source_id}})-[*..{max_depth}]-(b {{id: $target_id}})
        )
        RETURN p
        """
        records = await self.execute_cypher(
            query, {"source_id": source_id, "target_id": target_id}
        )
        if records:
            return self._record_to_path(records[0].get("p"))
        return None

    # ── Vector Search ──

    async def vector_search(
        self,
        label: str,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.5,
    ) -> List[Tuple[GraphNode, float]]:
        """
        Search for similar nodes using vector embeddings.

        Uses Neo4j vector index for approximate nearest neighbor search.
        """
        index_name = f"{label.lower()}_embedding_idx"
        query = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        WHERE score >= $min_score
        RETURN node, score
        ORDER BY score DESC
        """

        records = await self.execute_cypher(
            query,
            {
                "index_name": index_name,
                "top_k": top_k,
                "embedding": query_embedding,
                "min_score": min_score,
            },
        )

        results = []
        for record in records:
            node = self._record_to_node(record["node"], [label])
            score = record["score"]
            results.append((node, score))
        return results

    # ── Bulk Operations ──

    async def import_from_ontology(
        self,
        fact_types: list,
        fact_instances: list,
        schema: GraphSchema,
    ) -> Dict[str, int]:
        """
        Bulk import from ontology Layer 1 into knowledge graph.

        Converts elementary facts to nodes and edges.
        """
        await self.set_schema(schema)

        node_count = 0
        edge_count = 0

        for instance in fact_instances:
            fact_name = instance.fact_type
            rel_type = schema.fact_mappings.get(fact_name)

            if rel_type and len(instance.values) >= 2:
                # Binary fact → create nodes and edge
                role_names = list(instance.values.keys())
                source_value = instance.values[role_names[0]]
                target_value = instance.values[role_names[1]]

                # Find corresponding fact type to get object types
                fact_type = next(
                    (ft for ft in fact_types if ft.name == fact_name),
                    None,
                )
                if not fact_type or len(fact_type.roles) < 2:
                    continue

                source_label = fact_type.roles[0].object_type
                target_label = fact_type.roles[1].object_type

                # Create source node
                source_node = GraphNode(
                    labels=[source_label],
                    properties={"name": str(source_value)},
                    source_ontology_type=source_label,
                )
                await self.create_node(source_node)
                node_count += 1

                # Create target node
                target_node = GraphNode(
                    labels=[target_label],
                    properties={"name": str(target_value)},
                    source_ontology_type=target_label,
                )
                await self.create_node(target_node)
                node_count += 1

                # Create edge
                edge = GraphEdge(
                    source_id=source_node.id,
                    target_id=target_node.id,
                    relationship_type=rel_type,
                    source_fact_type=fact_name,
                )
                await self.create_edge(edge)
                edge_count += 1

        logger.info(f"Imported {node_count} nodes and {edge_count} edges from ontology")
        return {"nodes": node_count, "edges": edge_count}

    async def get_graph_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the knowledge graph"""
        label_counts = await self.execute_cypher(
            "CALL db.labels() YIELD label "
            "CALL { WITH label MATCH (n) WHERE label IN labels(n) RETURN count(n) AS cnt } "
            "RETURN label, cnt"
        )
        rel_counts = await self.execute_cypher(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "CALL { WITH relationshipType MATCH ()-[r]->() "
            "WHERE type(r) = relationshipType RETURN count(r) AS cnt } "
            "RETURN relationshipType, cnt"
        )

        return {
            "node_labels": {r["label"]: r["cnt"] for r in label_counts},
            "relationship_types": {r["relationshipType"]: r["cnt"] for r in rel_counts},
            "schema": self._schema.to_cypher_schema() if self._schema else None,
        }

    # ── Internal Helpers ──

    def _record_to_node(self, neo4j_node: Any, labels: List[str]) -> GraphNode:
        """Convert Neo4j node record to GraphNode"""
        props = dict(neo4j_node) if hasattr(neo4j_node, '__iter__') else {}
        node_id = props.pop("id", str(id(neo4j_node)))
        embedding = props.pop("embedding", None)
        ontology_type = props.pop("_ontology_type", None)

        return GraphNode(
            id=node_id,
            labels=labels,
            properties=props,
            embedding=embedding,
            source_ontology_type=ontology_type,
        )

    def _record_to_path(self, neo4j_path: Any) -> Optional[GraphPath]:
        """Convert Neo4j path to GraphPath"""
        if not neo4j_path:
            return None
        try:
            nodes = [
                self._record_to_node(n, list(n.labels))
                for n in neo4j_path.nodes
            ]
            edges = [
                GraphEdge(
                    source_id=str(r.start_node.element_id),
                    target_id=str(r.end_node.element_id),
                    relationship_type=r.type,
                    properties=dict(r),
                )
                for r in neo4j_path.relationships
            ]
            return GraphPath(nodes=nodes, edges=edges)
        except Exception:
            return None
