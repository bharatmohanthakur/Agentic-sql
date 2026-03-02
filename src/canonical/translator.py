"""
Model Translator - Converts between ontology, canonical, graph, and dimensional formats.

Bridges all 4 layers of the semantic space funnel.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TranslationDirection(str, Enum):
    """Supported translation directions"""
    ONTOLOGY_TO_CANONICAL = "ontology_to_canonical"
    CANONICAL_TO_GRAPH = "canonical_to_graph"
    CANONICAL_TO_DIMENSIONAL = "canonical_to_dimensional"
    GRAPH_TO_CANONICAL = "graph_to_canonical"
    DIMENSIONAL_TO_CANONICAL = "dimensional_to_canonical"


class ModelTranslator:
    """
    Translates between data representations across semantic layers.

    Layer 1 (Ontology)     ↔ Layer 3 (Canonical)
    Layer 2 (Graph)        ↔ Layer 3 (Canonical)
    Layer 3 (Canonical)    ↔ Layer 4 (Dimensional)

    The Canonical Model (Layer 3) is always the pivot format.
    """

    def translate(
        self,
        data: Any,
        direction: TranslationDirection,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Translate data between formats"""
        handlers = {
            TranslationDirection.ONTOLOGY_TO_CANONICAL: self._ontology_to_canonical,
            TranslationDirection.CANONICAL_TO_GRAPH: self._canonical_to_graph,
            TranslationDirection.CANONICAL_TO_DIMENSIONAL: self._canonical_to_dimensional,
            TranslationDirection.GRAPH_TO_CANONICAL: self._graph_to_canonical,
            TranslationDirection.DIMENSIONAL_TO_CANONICAL: self._dimensional_to_canonical,
        }

        handler = handlers.get(direction)
        if not handler:
            raise ValueError(f"Unsupported direction: {direction}")

        return handler(data, context or {})

    def _ontology_to_canonical(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert ontology fact instances to canonical entities.

        Input: {"fact_type": "PersonHasAddress",
                "values": {"person": "John", "address": "123 Main St"}}
        Output: {"entity_type": "Person",
                 "properties": {"name": "John"},
                 "relationships": [{"type": "has_address", "target_type": "Address", ...}]}
        """
        fact_type = data.get("fact_type", "")
        values = data.get("values", {})
        roles = data.get("roles", [])

        if len(roles) >= 2:
            # Binary fact → entity with relationship
            subject_role = roles[0]
            object_role = roles[1]

            return {
                "entity_type": subject_role.get("object_type", ""),
                "properties": {
                    subject_role.get("name", "value"): values.get(subject_role.get("name", ""), ""),
                },
                "relationships": [{
                    "relationship_type": fact_type.lower(),
                    "target_entity_type": object_role.get("object_type", ""),
                    "target_value": values.get(object_role.get("name", ""), ""),
                    "source_fact_type": fact_type,
                }],
            }

        # Unary or simple value
        return {
            "entity_type": data.get("object_type", "Entity"),
            "properties": values,
        }

    def _canonical_to_graph(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert canonical entity to graph node + edges.

        Input: CanonicalEntity dict
        Output: {"nodes": [...], "edges": [...]}
        """
        entity_type = data.get("entity_type", "Entity")
        entity_id = data.get("id", "")
        properties = data.get("properties", {})
        relationships = data.get("relationships", [])

        nodes = [{
            "id": entity_id,
            "labels": [entity_type],
            "properties": {**properties, "canonical_id": entity_id},
        }]

        edges = []
        for rel in relationships:
            target_id = rel.get("target_entity_id", rel.get("target_value", ""))
            edges.append({
                "source_id": entity_id,
                "target_id": target_id,
                "relationship_type": rel.get("relationship_type", "RELATED_TO").upper(),
                "properties": rel.get("properties", {}),
            })

            # Create target node if enough info
            target_type = rel.get("target_entity_type", "Entity")
            if target_id:
                nodes.append({
                    "id": target_id,
                    "labels": [target_type],
                    "properties": {"name": str(target_id)},
                })

        return {"nodes": nodes, "edges": edges}

    def _canonical_to_dimensional(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Flatten canonical entity to dimensional (star schema) format.

        Input: {"entity_type": "Person", "properties": {"name": {"first": "Jane", "last": "Doe"}}}
        Output: {"table": "DIM_PERSON", "columns": {"FIRST_NAME": "Jane", "LAST_NAME": "Doe"}}
        """
        entity_type = data.get("entity_type", "Entity")
        properties = data.get("properties", {})
        entity_id = data.get("id", "")

        # Flatten nested properties
        flat = {"ID": entity_id}
        self._flatten_dict(properties, flat, "")

        # Convert to dimensional naming (UPPER_SNAKE_CASE)
        dimensional = {}
        for key, value in flat.items():
            dim_key = key.upper().replace(".", "_").replace(" ", "_")
            dimensional[dim_key] = value

        table_name = f"DIM_{entity_type.upper()}"

        return {
            "table": table_name,
            "columns": dimensional,
            "entity_type": entity_type,
            "source_canonical_id": entity_id,
        }

    def _graph_to_canonical(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert graph node + edges to canonical entity.

        Input: {"node": {...}, "edges": [...]}
        Output: CanonicalEntity dict
        """
        node = data.get("node", {})
        edges = data.get("edges", [])

        labels = node.get("labels", [])
        properties = dict(node.get("properties", {}))
        properties.pop("canonical_id", None)

        relationships = []
        for edge in edges:
            relationships.append({
                "relationship_type": edge.get("relationship_type", "").lower(),
                "source_entity_type": labels[0] if labels else "",
                "target_entity_type": edge.get("target_label", ""),
                "source_entity_id": node.get("id", ""),
                "target_entity_id": edge.get("target_id", ""),
            })

        return {
            "id": node.get("id", ""),
            "entity_type": labels[0] if labels else "Entity",
            "properties": properties,
            "relationships": relationships,
        }

    def _dimensional_to_canonical(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert dimensional (flat row) to canonical entity.

        Input: {"table": "DIM_PERSON", "columns": {"FIRST_NAME": "Jane", ...}}
        Output: CanonicalEntity dict
        """
        table = data.get("table", "")
        columns = data.get("columns", {})

        # Infer entity type from table name
        entity_type = table.replace("DIM_", "").replace("FACT_", "").title()

        # Group flat columns into nested properties
        properties = {}
        for key, value in columns.items():
            if key == "ID":
                continue
            parts = key.lower().split("_")
            if len(parts) > 1:
                # Reconstruct nesting
                group = parts[0]
                field = "_".join(parts[1:])
                if group not in properties:
                    properties[group] = {}
                if isinstance(properties[group], dict):
                    properties[group][field] = value
                else:
                    properties[key.lower()] = value
            else:
                properties[key.lower()] = value

        return {
            "id": columns.get("ID", ""),
            "entity_type": entity_type,
            "properties": properties,
        }

    def _flatten_dict(
        self,
        obj: Any,
        result: Dict[str, Any],
        prefix: str,
    ) -> None:
        """Recursively flatten a nested dict"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}_{key}" if prefix else key
                self._flatten_dict(value, result, new_prefix)
        elif isinstance(obj, list):
            result[prefix] = obj
        else:
            result[prefix] = obj
