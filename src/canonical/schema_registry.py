"""
Schema Registry - Version tracking and compatibility checking for canonical models.

Manages schema evolution with backward/forward compatibility checks.
Schemas are stored as JSON Schema generated from Pydantic models.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CompatibilityMode(str, Enum):
    """Schema compatibility modes"""
    BACKWARD = "backward"        # New consumers can read old data
    FORWARD = "forward"          # Old consumers can read new data
    FULL = "full"                # Both directions
    NONE = "none"                # No compatibility checks


class SchemaVersion(BaseModel):
    """A versioned schema entry in the registry"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    entity_type: str
    version: str                  # Semantic version: MAJOR.MINOR.PATCH
    json_schema: Dict[str, Any]   # JSON Schema definition
    pydantic_model_name: str = ""  # Source Pydantic model class name
    description: str = ""
    changelog: str = ""
    is_deprecated: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SchemaRegistry:
    """
    Registry for canonical model schemas.

    Tracks versions, checks compatibility, and provides schema lookup.

    Usage:
        registry = SchemaRegistry()
        registry.register("Person", "1.0.0", CanonicalPerson.model_json_schema())
        registry.register("Person", "1.1.0", CanonicalPersonV2.model_json_schema())

        schema = registry.get_latest("Person")
        is_compatible = registry.check_compatibility("Person", new_schema)
    """

    def __init__(self, compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARD):
        self.compatibility_mode = compatibility_mode
        self._schemas: Dict[str, List[SchemaVersion]] = {}  # entity_type -> versions

    def register(
        self,
        entity_type: str,
        version: str,
        json_schema: Dict[str, Any],
        description: str = "",
        changelog: str = "",
    ) -> SchemaVersion:
        """Register a new schema version"""
        if entity_type not in self._schemas:
            self._schemas[entity_type] = []

        # Check compatibility with previous version
        if self._schemas[entity_type] and self.compatibility_mode != CompatibilityMode.NONE:
            latest = self._schemas[entity_type][-1]
            errors = self._check_compatibility(
                latest.json_schema, json_schema, self.compatibility_mode
            )
            if errors:
                raise ValueError(
                    f"Schema incompatible ({self.compatibility_mode.value}): {errors}"
                )

        sv = SchemaVersion(
            entity_type=entity_type,
            version=version,
            json_schema=json_schema,
            description=description,
            changelog=changelog,
        )
        self._schemas[entity_type].append(sv)
        logger.info(f"Registered schema: {entity_type} v{version}")
        return sv

    def get_latest(self, entity_type: str) -> Optional[SchemaVersion]:
        """Get the latest schema version for an entity type"""
        versions = self._schemas.get(entity_type, [])
        if not versions:
            return None
        return versions[-1]

    def get_version(self, entity_type: str, version: str) -> Optional[SchemaVersion]:
        """Get a specific schema version"""
        for sv in self._schemas.get(entity_type, []):
            if sv.version == version:
                return sv
        return None

    def get_all_versions(self, entity_type: str) -> List[SchemaVersion]:
        """Get all versions of a schema"""
        return self._schemas.get(entity_type, [])

    def list_entity_types(self) -> List[str]:
        """List all registered entity types"""
        return list(self._schemas.keys())

    def check_compatibility(
        self,
        entity_type: str,
        new_schema: Dict[str, Any],
    ) -> List[str]:
        """Check if a new schema is compatible with the latest version"""
        latest = self.get_latest(entity_type)
        if not latest:
            return []  # No previous version, always compatible

        return self._check_compatibility(
            latest.json_schema, new_schema, self.compatibility_mode
        )

    def validate_data(
        self,
        entity_type: str,
        data: Dict[str, Any],
        version: Optional[str] = None,
    ) -> List[str]:
        """Validate data against a schema version"""
        if version:
            sv = self.get_version(entity_type, version)
        else:
            sv = self.get_latest(entity_type)

        if not sv:
            return [f"No schema found for {entity_type}"]

        return self._validate_against_schema(data, sv.json_schema)

    def to_summary(self) -> Dict[str, Any]:
        """Get a summary of all registered schemas for LLM context"""
        summary = {}
        for entity_type, versions in self._schemas.items():
            latest = versions[-1] if versions else None
            summary[entity_type] = {
                "latest_version": latest.version if latest else None,
                "total_versions": len(versions),
                "properties": list(
                    latest.json_schema.get("properties", {}).keys()
                ) if latest else [],
                "required": latest.json_schema.get("required", []) if latest else [],
            }
        return summary

    def _check_compatibility(
        self,
        old_schema: Dict[str, Any],
        new_schema: Dict[str, Any],
        mode: CompatibilityMode,
    ) -> List[str]:
        """Check compatibility between two schemas"""
        errors = []
        old_props = old_schema.get("properties", {})
        new_props = new_schema.get("properties", {})
        old_required = set(old_schema.get("required", []))
        new_required = set(new_schema.get("required", []))

        if mode in (CompatibilityMode.BACKWARD, CompatibilityMode.FULL):
            # Backward: new consumers can read old data
            # Cannot add new required fields (old data won't have them)
            new_required_fields = new_required - old_required
            for field in new_required_fields:
                if field not in old_props:
                    errors.append(
                        f"Backward incompatible: new required field '{field}' "
                        f"not present in old schema"
                    )

        if mode in (CompatibilityMode.FORWARD, CompatibilityMode.FULL):
            # Forward: old consumers can read new data
            # Cannot remove fields that old consumers expect
            removed_fields = set(old_props.keys()) - set(new_props.keys())
            for field in removed_fields:
                if field in old_required:
                    errors.append(
                        f"Forward incompatible: required field '{field}' "
                        f"was removed"
                    )

        return errors

    def _validate_against_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> List[str]:
        """Simple validation against JSON schema"""
        errors = []
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        for field, value in data.items():
            if field in properties:
                prop_schema = properties[field]
                expected_type = prop_schema.get("type")
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Field '{field}' should be string, got {type(value).__name__}")
                elif expected_type == "integer" and not isinstance(value, int):
                    errors.append(f"Field '{field}' should be integer, got {type(value).__name__}")

        return errors
