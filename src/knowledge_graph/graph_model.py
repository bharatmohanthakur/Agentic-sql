"""
Knowledge Graph Model - Node, Edge, Property type definitions.

Represents domain knowledge as a property graph compatible with Neo4j.
Bridges between ontology facts (Layer 1) and queryable graph structures.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PropertyType(str, Enum):
    """Supported property value types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    LIST = "list"
    VECTOR = "vector"  # For embedding vectors


class GraphProperty(BaseModel):
    """A property on a node or edge"""
    name: str
    value: Any
    property_type: PropertyType = PropertyType.STRING
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """
    A node in the knowledge graph.

    Nodes represent entities (Person, Address, Product) with labels and properties.
    Maps to Neo4j nodes.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    labels: List[str] = Field(default_factory=list)  # Neo4j node labels
    properties: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None  # Vector embedding
    source_ontology_type: Optional[str] = None  # Link back to ObjectType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def primary_label(self) -> str:
        return self.labels[0] if self.labels else "Node"


class GraphEdge(BaseModel):
    """
    An edge (relationship) in the knowledge graph.

    Edges represent relationships between nodes with a type and properties.
    Maps to Neo4j relationships.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str              # Source node ID
    target_id: str              # Target node ID
    relationship_type: str      # Neo4j relationship type (e.g., "HAS_ADDRESS")
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_fact_type: Optional[str] = None  # Link back to ElementaryFact
    weight: float = 1.0         # Edge weight for graph algorithms
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphPath(BaseModel):
    """A path through the knowledge graph (sequence of nodes and edges)"""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    total_weight: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.edges)

    def to_text(self) -> str:
        """Convert path to human-readable text"""
        parts = []
        for i, node in enumerate(self.nodes):
            parts.append(f"({node.primary_label}: {node.properties.get('name', node.id)})")
            if i < len(self.edges):
                parts.append(f"-[:{self.edges[i].relationship_type}]->")
        return "".join(parts)


class GraphSchema(BaseModel):
    """
    Schema definition for the knowledge graph.

    Defines allowed node labels, relationship types, and their properties.
    Derived from the ontology layer.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    version: str = "1.0.0"

    # Schema elements
    node_labels: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    # label -> {property_name: property_type}

    relationship_types: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # rel_type -> {source_label, target_label, properties}

    # Constraints
    unique_properties: Dict[str, List[str]] = Field(default_factory=dict)
    # label -> [property_names that must be unique]

    # Ontology mapping
    ontology_id: Optional[str] = None
    type_mappings: Dict[str, str] = Field(default_factory=dict)
    # ObjectType.name -> node_label

    fact_mappings: Dict[str, str] = Field(default_factory=dict)
    # ElementaryFact.name -> relationship_type

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_node_properties(self, label: str) -> Dict[str, str]:
        """Get expected properties for a node label"""
        return self.node_labels.get(label, {})

    def get_relationship_info(self, rel_type: str) -> Optional[Dict[str, Any]]:
        """Get info about a relationship type"""
        return self.relationship_types.get(rel_type)

    def to_cypher_schema(self) -> str:
        """Generate Cypher schema description for LLM context"""
        lines = ["// Knowledge Graph Schema", ""]

        lines.append("// Node Labels:")
        for label, props in self.node_labels.items():
            prop_str = ", ".join(f"{k}: {v}" for k, v in props.items())
            lines.append(f"(:{label} {{{prop_str}}})")

        lines.append("\n// Relationships:")
        for rel_type, info in self.relationship_types.items():
            source = info.get("source_label", "?")
            target = info.get("target_label", "?")
            lines.append(f"(:{source})-[:{rel_type}]->(:{target})")

        return "\n".join(lines)

    @classmethod
    def from_ontology(
        cls,
        ontology_name: str,
        ontology_id: str,
        object_types: list,
        fact_types: list,
    ) -> "GraphSchema":
        """
        Create a GraphSchema from an OntologyDefinition.

        Maps ObjectTypes to node labels and ElementaryFacts to relationship types.
        """
        node_labels = {}
        relationship_types = {}
        type_mappings = {}
        fact_mappings = {}

        # Map object types to node labels
        for ot in object_types:
            label = ot.name
            props = {"name": "string", "id": "string"}
            if ot.reference_mode:
                props[ot.reference_mode] = "string"
            node_labels[label] = props
            type_mappings[ot.name] = label

        # Map binary facts to relationships
        for ft in fact_types:
            if len(ft.roles) == 2:
                rel_type = ft.name.upper()
                source_label = ft.roles[0].object_type
                target_label = ft.roles[1].object_type
                relationship_types[rel_type] = {
                    "source_label": source_label,
                    "target_label": target_label,
                    "reading": ft.reading,
                }
                fact_mappings[ft.name] = rel_type

        return cls(
            name=f"{ontology_name}_graph",
            ontology_id=ontology_id,
            node_labels=node_labels,
            relationship_types=relationship_types,
            type_mappings=type_mappings,
            fact_mappings=fact_mappings,
        )
