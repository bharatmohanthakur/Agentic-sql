"""
Layer 3: Canonical Model (MEDIUM Semantic Space)

Serves as the "lowest common denominator" for integration.
Standardized structures that bridge knowledge graphs and dimensional models.

Example: { "Person": { "Name": { "First": "Jane", "Last": "Doe" }, "Address": {...} } }

Components:
- canonical_model: Pydantic canonical data model definitions
- schema_registry: Schema versioning and registry
- schema_agent: Agent for schema validation and transformation
- translator: Model translation between layers
"""

from .canonical_model import (
    CanonicalEntity,
    CanonicalMetadata,
    CanonicalRelationship,
    CanonicalDataset,
)
from .schema_registry import SchemaRegistry, SchemaVersion
from .schema_agent import SchemaAgent, SchemaAgentConfig
from .translator import (
    ModelTranslator,
    TranslationDirection,
)

__all__ = [
    "CanonicalEntity",
    "CanonicalMetadata",
    "CanonicalRelationship",
    "CanonicalDataset",
    "SchemaRegistry",
    "SchemaVersion",
    "SchemaAgent",
    "SchemaAgentConfig",
    "ModelTranslator",
    "TranslationDirection",
]
