"""
Canonical Data Model - Standardized entity definitions.

The canonical model serves as the integration contract between systems.
Defined as Pydantic models for runtime validation and JSON Schema generation.

Key principle: Application-independent, entity-centric, relationship-aware.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CanonicalMetadata(BaseModel):
    """
    Metadata embedded in every canonical entity.

    Tracks schema version, source system, and lineage.
    """
    schema_version: str = "1.0.0"
    source_system: str = ""
    source_id: Optional[str] = None
    canonical_timestamp: datetime = Field(default_factory=datetime.utcnow)
    lineage: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class CanonicalRelationship(BaseModel):
    """
    A relationship between canonical entities.

    Tracks how entities connect, derived from ontology fact types.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    relationship_type: str          # e.g., "has_address", "placed_order"
    source_entity_type: str         # e.g., "Person"
    target_entity_type: str         # e.g., "Address"
    source_entity_id: str
    target_entity_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_fact_type: Optional[str] = None  # Link to ontology ElementaryFact
    metadata: CanonicalMetadata = Field(default_factory=CanonicalMetadata)


class CanonicalEntity(BaseModel):
    """
    Base canonical entity - the integration contract.

    All domain entities inherit from this base.
    Provides validation, serialization, and schema generation.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    entity_type: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[CanonicalRelationship] = Field(default_factory=list)
    metadata: CanonicalMetadata = Field(default_factory=CanonicalMetadata)

    # Ontology traceability
    source_ontology_type: Optional[str] = None  # Link to ObjectType

    def get_property(self, name: str, default: Any = None) -> Any:
        """Get a property value"""
        return self.properties.get(name, default)

    def set_property(self, name: str, value: Any) -> None:
        """Set a property value"""
        self.properties[name] = value

    def add_relationship(
        self,
        relationship_type: str,
        target_entity_type: str,
        target_entity_id: str,
        **properties,
    ) -> CanonicalRelationship:
        """Add a relationship to another entity"""
        rel = CanonicalRelationship(
            relationship_type=relationship_type,
            source_entity_type=self.entity_type,
            target_entity_type=target_entity_type,
            source_entity_id=self.id,
            target_entity_id=target_entity_id,
            properties=properties,
        )
        self.relationships.append(rel)
        return rel

    def to_flat(self, prefix: str = "") -> Dict[str, Any]:
        """
        Flatten to dimensional format (for Snowflake star schema).

        Converts hierarchical canonical structure to flat key-value pairs.
        Example: {"name": {"first": "Jane"}} → {"name_first": "Jane"}
        """
        flat = {"id": self.id, "entity_type": self.entity_type}

        def _flatten(obj: Any, key_prefix: str) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{key_prefix}_{k}" if key_prefix else k
                    _flatten(v, new_key)
            elif isinstance(obj, list):
                flat[key_prefix] = obj  # Lists stay as-is
            else:
                flat[key_prefix] = obj

        _flatten(self.properties, prefix)
        return flat

    def to_graph_node(self) -> Dict[str, Any]:
        """Convert to knowledge graph node format"""
        return {
            "id": self.id,
            "labels": [self.entity_type],
            "properties": {**self.properties, "canonical_id": self.id},
        }

    def to_graph_edges(self) -> List[Dict[str, Any]]:
        """Convert relationships to knowledge graph edge format"""
        return [
            {
                "source_id": rel.source_entity_id,
                "target_id": rel.target_entity_id,
                "relationship_type": rel.relationship_type.upper(),
                "properties": rel.properties,
            }
            for rel in self.relationships
        ]


class CanonicalDataset(BaseModel):
    """
    A collection of canonical entities and their relationships.

    Represents a batch of entities for bulk operations.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""
    entities: List[CanonicalEntity] = Field(default_factory=list)
    metadata: CanonicalMetadata = Field(default_factory=CanonicalMetadata)

    def add_entity(self, entity: CanonicalEntity) -> None:
        """Add an entity to the dataset"""
        self.entities.append(entity)

    def get_entities_by_type(self, entity_type: str) -> List[CanonicalEntity]:
        """Get all entities of a specific type"""
        return [e for e in self.entities if e.entity_type == entity_type]

    def get_all_relationships(self) -> List[CanonicalRelationship]:
        """Get all relationships across all entities"""
        rels = []
        for entity in self.entities:
            rels.extend(entity.relationships)
        return rels

    @property
    def entity_types(self) -> Set[str]:
        """Get all unique entity types"""
        return {e.entity_type for e in self.entities}

    @property
    def entity_count(self) -> int:
        return len(self.entities)
