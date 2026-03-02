"""
Dimensional Model - Star/Snowflake schema definitions (Kimball methodology).

Defines fact tables, dimension tables, bridge tables, and their relationships.
Tracks the "premature compression" from canonical models.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SCDType(str, Enum):
    """Slowly Changing Dimension types"""
    TYPE_0 = "type_0"  # Retain original - no changes
    TYPE_1 = "type_1"  # Overwrite - no history
    TYPE_2 = "type_2"  # Add new row - full history
    TYPE_3 = "type_3"  # Add previous-value column
    TYPE_6 = "type_6"  # Hybrid 1+2+3


class ColumnDefinition(BaseModel):
    """A column in a dimensional table"""
    name: str
    data_type: str = "VARCHAR"
    description: str = ""
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_table: Optional[str] = None
    is_nullable: bool = True
    is_measure: bool = False          # For fact table numeric measures
    scd_type: Optional[SCDType] = None
    # Ontology traceability
    ontology_concept: Optional[str] = None  # Link to ObjectType/FactType
    semantic_loss_notes: str = ""            # What meaning was compressed
    canonical_source: Optional[str] = None   # canonical_model.field path


class DimensionTable(BaseModel):
    """
    A dimension table in the star schema.

    Stores descriptive attributes (who, what, where, when).
    May implement SCD for historical tracking.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str                     # e.g., "Dim_Employee"
    description: str = ""
    business_key: str = ""        # Natural key column
    surrogate_key: str = ""       # Surrogate key column
    columns: List[ColumnDefinition] = Field(default_factory=list)
    scd_type: SCDType = SCDType.TYPE_1
    is_conformed: bool = False    # Shared across multiple fact tables
    grain: str = ""               # One row per...
    # Ontology traceability
    source_ontology_types: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_ddl(self, schema_name: str = "analytics") -> str:
        """Generate Snowflake DDL for this dimension"""
        cols = []
        for col in self.columns:
            nullable = "" if col.is_nullable else " NOT NULL"
            pk = " PRIMARY KEY" if col.is_primary_key else ""
            cols.append(f"    {col.name} {col.data_type}{nullable}{pk}")

        # Add SCD columns if Type 2
        if self.scd_type == SCDType.TYPE_2:
            cols.extend([
                "    effective_date DATE NOT NULL",
                "    expiration_date DATE",
                "    is_current BOOLEAN DEFAULT TRUE",
            ])

        cols.append("    _loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()")

        col_str = ",\n".join(cols)
        return f"CREATE TABLE IF NOT EXISTS {schema_name}.{self.name} (\n{col_str}\n);"


class FactTable(BaseModel):
    """
    A fact table in the star schema.

    Stores measurements/events with foreign keys to dimensions.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str                     # e.g., "Fact_Sales"
    description: str = ""
    grain: str = ""               # e.g., "One row per order line item per day"
    fact_type: str = "transaction"  # transaction, periodic_snapshot, accumulating
    dimension_keys: List[str] = Field(default_factory=list)  # FK column names
    measures: List[ColumnDefinition] = Field(default_factory=list)
    degenerate_dimensions: List[str] = Field(default_factory=list)  # e.g., OrderNumber
    columns: List[ColumnDefinition] = Field(default_factory=list)
    clustering_keys: List[str] = Field(default_factory=list)
    # Ontology traceability
    source_ontology_facts: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_ddl(self, schema_name: str = "analytics") -> str:
        """Generate Snowflake DDL for this fact table"""
        cols = []
        for col in self.columns:
            nullable = "" if col.is_nullable else " NOT NULL"
            cols.append(f"    {col.name} {col.data_type}{nullable}")

        cols.append("    _loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()")
        col_str = ",\n".join(cols)

        ddl = f"CREATE TABLE IF NOT EXISTS {schema_name}.{self.name} (\n{col_str}\n)"

        if self.clustering_keys:
            keys = ", ".join(self.clustering_keys)
            ddl += f"\nCLUSTER BY ({keys})"

        return ddl + ";"


class BridgeTable(BaseModel):
    """
    A bridge table for many-to-many relationships.

    Resolves M:M that star schemas can't directly represent.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str                     # e.g., "B_Has_Skill"
    description: str = ""
    dimension_key_1: str = ""     # First FK
    dimension_key_2: str = ""     # Second FK
    weight_factor: bool = True    # Include weight factor column
    columns: List[ColumnDefinition] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StarSchema(BaseModel):
    """
    Complete star schema definition.

    Contains all fact tables, dimension tables, and bridge tables
    that form a dimensional model for a business process.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    business_process: str = ""

    fact_tables: List[FactTable] = Field(default_factory=list)
    dimension_tables: List[DimensionTable] = Field(default_factory=list)
    bridge_tables: List[BridgeTable] = Field(default_factory=list)

    # Ontology traceability
    ontology_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_dimension(self, name: str) -> Optional[DimensionTable]:
        """Look up a dimension by name"""
        for d in self.dimension_tables:
            if d.name == name:
                return d
        return None

    def get_fact(self, name: str) -> Optional[FactTable]:
        """Look up a fact table by name"""
        for f in self.fact_tables:
            if f.name == name:
                return f
        return None

    def get_conformed_dimensions(self) -> List[DimensionTable]:
        """Get all conformed dimensions"""
        return [d for d in self.dimension_tables if d.is_conformed]

    def to_schema_context(self) -> str:
        """Generate schema description for LLM context"""
        lines = [f"Star Schema: {self.name}", f"Business Process: {self.business_process}", ""]

        lines.append("FACT TABLES:")
        for ft in self.fact_tables:
            lines.append(f"  {ft.name} (grain: {ft.grain})")
            for col in ft.columns:
                measure_tag = " [MEASURE]" if col.is_measure else ""
                fk_tag = f" [FK→{col.foreign_key_table}]" if col.is_foreign_key else ""
                lines.append(f"    - {col.name}: {col.data_type}{measure_tag}{fk_tag}")

        lines.append("\nDIMENSION TABLES:")
        for dt in self.dimension_tables:
            conformed_tag = " [CONFORMED]" if dt.is_conformed else ""
            scd_tag = f" [{dt.scd_type.value.upper()}]"
            lines.append(f"  {dt.name}{conformed_tag}{scd_tag} (grain: {dt.grain})")
            for col in dt.columns:
                lines.append(f"    - {col.name}: {col.data_type}")

        if self.bridge_tables:
            lines.append("\nBRIDGE TABLES:")
            for bt in self.bridge_tables:
                lines.append(f"  {bt.name}: {bt.dimension_key_1} ↔ {bt.dimension_key_2}")

        return "\n".join(lines)

    def generate_all_ddl(self, schema_name: str = "analytics") -> str:
        """Generate complete DDL for the star schema"""
        ddl_parts = [f"-- Star Schema: {self.name}", f"-- {self.description}", ""]

        for dt in self.dimension_tables:
            ddl_parts.append(f"-- Dimension: {dt.name}")
            ddl_parts.append(dt.to_ddl(schema_name))
            ddl_parts.append("")

        for ft in self.fact_tables:
            ddl_parts.append(f"-- Fact: {ft.name}")
            ddl_parts.append(ft.to_ddl(schema_name))
            ddl_parts.append("")

        return "\n".join(ddl_parts)
