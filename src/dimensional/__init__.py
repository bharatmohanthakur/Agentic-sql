"""
Layer 4: Dimensional Model (LOWEST Semantic Space)

Flattened & optimized for reporting. Implements Kimball star/snowflake schemas.
This is where "premature compression" occurs - semantic meaning is sacrificed for query speed.

Example: Dim_Employee (StreetAddress column) - flattens Person+Address+City into one field.

Components:
- dimensional_model: Star/snowflake schema definitions
- snowflake_client: Snowflake DDL, query, and metadata operations
- sql_generation_agent: Semantic-aware SQL generation for dimensional queries
- lineage: Data lineage tracking from dimensional back to ontology
"""

from .dimensional_model import (
    DimensionTable,
    FactTable,
    BridgeTable,
    StarSchema,
    SCDType,
)
from .snowflake_client import SnowflakeClient, SnowflakeConfig
from .sql_generation_agent import DimensionalSQLAgent, DimensionalSQLAgentConfig
from .lineage import LineageTracker, LineageEdge, LineagePath

__all__ = [
    "DimensionTable",
    "FactTable",
    "BridgeTable",
    "StarSchema",
    "SCDType",
    "SnowflakeClient",
    "SnowflakeConfig",
    "DimensionalSQLAgent",
    "DimensionalSQLAgentConfig",
    "LineageTracker",
    "LineageEdge",
    "LineagePath",
]
