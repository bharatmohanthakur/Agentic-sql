"""
Auto-Discovery System - Intelligent schema discovery and profiling
Automatically discovers and understands database structure when connected
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ColumnType(str, Enum):
    """Semantic column types"""
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    IDENTIFIER = "identifier"  # UUID, code, etc.
    NAME = "name"
    DESCRIPTION = "description"
    AMOUNT = "amount"
    QUANTITY = "quantity"
    PRICE = "price"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    STATUS = "status"
    CATEGORY = "category"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    URL = "url"
    JSON = "json"
    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    TEXT = "text"


class TableType(str, Enum):
    """Semantic table types"""
    ENTITY = "entity"  # Core business entity (users, products)
    JUNCTION = "junction"  # Many-to-many relationship
    LOOKUP = "lookup"  # Reference/lookup table
    TRANSACTION = "transaction"  # Transactional data
    LOG = "log"  # Audit/log table
    STAGING = "staging"  # ETL staging
    ARCHIVE = "archive"  # Historical data
    METADATA = "metadata"  # System metadata


@dataclass
class ColumnProfile:
    """Detailed column profile"""
    name: str
    data_type: str
    semantic_type: ColumnType = ColumnType.TEXT
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: Optional[str] = None  # table.column

    # Statistics
    distinct_count: int = 0
    null_count: int = 0
    sample_values: List[Any] = field(default_factory=list)
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_length: float = 0.0

    # Inferred metadata
    description: str = ""
    business_name: str = ""  # Human-readable name
    tags: List[str] = field(default_factory=list)


@dataclass
class TableProfile:
    """Detailed table profile"""
    name: str
    schema_name: str = "public"
    table_type: TableType = TableType.ENTITY
    columns: List[ColumnProfile] = field(default_factory=list)

    # Statistics
    row_count: int = 0
    size_bytes: int = 0

    # Relationships
    primary_key: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    referenced_by: List[str] = field(default_factory=list)

    # Inferred metadata
    description: str = ""
    business_name: str = ""
    domain: str = ""  # e.g., "sales", "inventory", "users"
    tags: List[str] = field(default_factory=list)

    # Sample queries
    common_queries: List[str] = field(default_factory=list)


class DatabaseDialect(str, Enum):
    """Supported database dialects for schema discovery"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MSSQL = "mssql"
    SQLITE = "sqlite"


class SchemaDiscovery:
    """
    Intelligent Schema Discovery Engine

    Automatically discovers:
    - All tables and columns
    - Primary and foreign keys
    - Implicit relationships (naming conventions)
    - Column semantics (types, business meaning)
    - Table domains and categories

    Supports multiple database types: PostgreSQL, MySQL, MSSQL, SQLite
    """

    def __init__(
        self,
        db_executor: Callable,
        llm_client: Optional[Any] = None,
        cache_ttl_seconds: int = 3600,
        dialect: DatabaseDialect = DatabaseDialect.POSTGRESQL,
    ):
        self.db_executor = db_executor
        self.llm = llm_client
        self.cache_ttl = cache_ttl_seconds
        self.dialect = dialect
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None

    async def discover(
        self,
        schemas: Optional[List[str]] = None,
        include_stats: bool = True,
        include_samples: bool = True,
        max_tables: int = 100,
    ) -> Dict[str, TableProfile]:
        """
        Full schema discovery

        Returns dict of table_name -> TableProfile
        """
        logger.info("Starting schema discovery...")
        start_time = datetime.utcnow()

        # 1. Get all tables
        tables = await self._discover_tables(schemas, max_tables)
        logger.info(f"Found {len(tables)} tables")

        # 2. Get columns for each table
        profiles: Dict[str, TableProfile] = {}

        for table_info in tables:
            table_name = table_info["name"]
            schema_name = table_info.get("schema", "public")

            profile = TableProfile(
                name=table_name,
                schema_name=schema_name,
            )

            # Get columns
            profile.columns = await self._discover_columns(schema_name, table_name)

            # Get constraints
            await self._discover_constraints(profile)

            # Get statistics
            if include_stats:
                await self._get_table_stats(profile)

            # Get samples
            if include_samples:
                await self._get_sample_values(profile)

            profiles[table_name] = profile

        # 3. Infer implicit relationships
        await self._infer_relationships(profiles)

        # 4. Classify tables
        self._classify_tables(profiles)

        # 5. Infer column semantics
        await self._infer_column_semantics(profiles)

        # 6. Generate descriptions using LLM
        if self.llm:
            await self._generate_descriptions(profiles)

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Schema discovery completed in {elapsed:.2f}s")

        return profiles

    async def _discover_tables(
        self,
        schemas: Optional[List[str]],
        max_tables: int,
    ) -> List[Dict[str, str]]:
        """Discover all tables in database - supports multiple dialects"""

        if self.dialect == DatabaseDialect.MSSQL:
            # MSSQL query - includes views
            sql = f"""
            SELECT TOP {max_tables}
                TABLE_SCHEMA as [schema],
                TABLE_NAME as name,
                TABLE_TYPE as type
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'dbo'
            ORDER BY TABLE_NAME
            """
        elif self.dialect == DatabaseDialect.MYSQL:
            sql = f"""
            SELECT
                TABLE_SCHEMA as `schema`,
                TABLE_NAME as name,
                TABLE_TYPE as type
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
            LIMIT {max_tables}
            """
        elif self.dialect == DatabaseDialect.SQLITE:
            sql = f"""
            SELECT name, 'main' as schema, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
            LIMIT {max_tables}
            """
        else:
            # PostgreSQL (default)
            sql = f"""
            SELECT
                table_schema as schema,
                table_name as name,
                table_type as type
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            LIMIT {max_tables}
            """

        if schemas and self.dialect not in [DatabaseDialect.SQLITE]:
            schemas_str = ",".join(f"'{s}'" for s in schemas)
            if self.dialect == DatabaseDialect.MSSQL:
                sql = sql.replace("TABLE_SCHEMA = 'dbo'", f"TABLE_SCHEMA IN ({schemas_str})")
            else:
                sql += f" AND table_schema IN ({schemas_str})"

        try:
            result = await self.db_executor(sql)
            tables = []
            for r in (result or []):
                tables.append({
                    "name": r.get("name") or r.get("TABLE_NAME"),
                    "schema": r.get("schema") or r.get("TABLE_SCHEMA", "dbo"),
                    "type": r.get("type") or r.get("TABLE_TYPE", "TABLE"),
                })
            logger.info(f"Discovered {len(tables)} tables/views")
            return tables
        except Exception as e:
            logger.warning(f"Table discovery failed: {e}")
            return []

    async def _discover_columns(
        self,
        schema_name: str,
        table_name: str,
    ) -> List[ColumnProfile]:
        """Discover columns for a table - supports multiple dialects"""

        if self.dialect == DatabaseDialect.MSSQL:
            sql = f"""
            SELECT
                COLUMN_NAME as column_name,
                DATA_TYPE as data_type,
                IS_NULLABLE as is_nullable,
                COLUMN_DEFAULT as column_default,
                CHARACTER_MAXIMUM_LENGTH as max_length
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{schema_name}'
            AND TABLE_NAME = '{table_name}'
            ORDER BY ORDINAL_POSITION
            """
        elif self.dialect == DatabaseDialect.SQLITE:
            sql = f"PRAGMA table_info('{table_name}')"
        else:
            # PostgreSQL / MySQL
            sql = f"""
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = '{schema_name}'
            AND table_name = '{table_name}'
            ORDER BY ordinal_position
            """

        try:
            result = await self.db_executor(sql)
            columns = []

            for row in result:
                if self.dialect == DatabaseDialect.SQLITE:
                    col = ColumnProfile(
                        name=row.get("name", ""),
                        data_type=row.get("type", "TEXT"),
                        nullable=row.get("notnull", 0) == 0,
                    )
                else:
                    col = ColumnProfile(
                        name=row.get("column_name") or row.get("COLUMN_NAME", ""),
                        data_type=row.get("data_type") or row.get("DATA_TYPE", ""),
                        nullable=(row.get("is_nullable") or row.get("IS_NULLABLE", "YES")) == "YES",
                    )
                columns.append(col)

            logger.debug(f"Discovered {len(columns)} columns for {table_name}")
            return columns
        except Exception as e:
            logger.warning(f"Column discovery failed for {table_name}: {e}")
            return []

    async def _discover_constraints(self, profile: TableProfile) -> None:
        """Discover primary and foreign key constraints - supports multiple dialects"""

        # Primary key - different SQL per dialect
        if self.dialect == DatabaseDialect.MSSQL:
            pk_sql = f"""
            SELECT COLUMN_NAME as column_name
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + CONSTRAINT_NAME), 'IsPrimaryKey') = 1
            AND TABLE_NAME = '{profile.name}'
            """
        else:
            pk_sql = f"""
            SELECT a.attname as column_name
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = '{profile.schema_name}.{profile.name}'::regclass
            AND i.indisprimary
            """

        try:
            result = await self.db_executor(pk_sql)
            for row in (result or []):
                col_name = row.get("column_name") or row.get("COLUMN_NAME")
                if col_name:
                    profile.primary_key.append(col_name)
                    for col in profile.columns:
                        if col.name == col_name:
                            col.is_primary_key = True
                            col.semantic_type = ColumnType.PRIMARY_KEY
        except Exception as e:
            logger.debug(f"PK discovery skipped for {profile.name}: {e}")

        # Foreign keys
        if self.dialect == DatabaseDialect.MSSQL:
            fk_sql = f"""
            SELECT
                COL_NAME(fc.parent_object_id, fc.parent_column_id) AS column_name,
                OBJECT_NAME(fc.referenced_object_id) AS foreign_table,
                COL_NAME(fc.referenced_object_id, fc.referenced_column_id) AS foreign_column
            FROM sys.foreign_key_columns fc
            WHERE OBJECT_NAME(fc.parent_object_id) = '{profile.name}'
            """
        else:
            fk_sql = f"""
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table,
                ccu.column_name AS foreign_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_name = '{profile.name}'
            """

        try:
            result = await self.db_executor(fk_sql)
            for row in (result or []):
                col_name = row.get("column_name") or row.get("COLUMN_NAME")
                fk_table = row.get("foreign_table") or row.get("FOREIGN_TABLE")
                fk_col = row.get("foreign_column") or row.get("FOREIGN_COLUMN")
                if col_name and fk_table:
                    profile.foreign_keys.append({
                        "column": col_name,
                        "references": f"{fk_table}.{fk_col}",
                    })
                    for col in profile.columns:
                        if col.name == col_name:
                            col.is_foreign_key = True
                            col.references = f"{fk_table}.{fk_col}"
                            col.semantic_type = ColumnType.FOREIGN_KEY
        except Exception as e:
            logger.debug(f"FK discovery skipped for {profile.name}: {e}")

    async def _get_table_stats(self, profile: TableProfile) -> None:
        """Get row count and size - supports multiple dialects"""
        try:
            table_ref = f"[{profile.schema_name}].[{profile.name}]" if self.dialect == DatabaseDialect.MSSQL else profile.name
            result = await self.db_executor(f"SELECT COUNT(*) as cnt FROM {table_ref}")
            profile.row_count = result[0].get("cnt", 0) if result else 0
            logger.debug(f"Table {profile.name} has {profile.row_count} rows")
        except Exception as e:
            logger.debug(f"Stats skipped for {profile.name}: {e}")

    async def _get_sample_values(self, profile: TableProfile) -> None:
        """Get sample values for each column - supports multiple dialects"""
        # Skip large text columns
        skip_types = ["text", "ntext", "image", "xml", "varbinary"]

        for col in profile.columns:
            if any(st in col.data_type.lower() for st in skip_types):
                continue
            if "max" in col.data_type.lower():
                continue

            try:
                if self.dialect == DatabaseDialect.MSSQL:
                    sql = f"""
                    SELECT DISTINCT TOP 5 [{col.name}] as val
                    FROM [{profile.schema_name}].[{profile.name}]
                    WHERE [{col.name}] IS NOT NULL
                    """
                else:
                    sql = f"""
                    SELECT DISTINCT "{col.name}" as val
                    FROM {profile.name}
                    WHERE "{col.name}" IS NOT NULL
                    LIMIT 5
                    """
                result = await self.db_executor(sql)
                col.sample_values = [r.get("val") for r in (result or [])]
            except Exception as e:
                logger.debug(f"Sample values skipped for {profile.name}.{col.name}: {e}")

    async def _infer_relationships(
        self,
        profiles: Dict[str, TableProfile],
    ) -> None:
        """Infer implicit relationships from naming conventions"""
        table_names = set(profiles.keys())

        for table_name, profile in profiles.items():
            for col in profile.columns:
                # Pattern: column ends with _id and matches a table name
                if col.name.endswith("_id"):
                    potential_table = col.name[:-3]  # Remove _id

                    # Check singular/plural forms
                    candidates = [
                        potential_table,
                        potential_table + "s",
                        potential_table + "es",
                        potential_table.rstrip("s"),
                    ]

                    for candidate in candidates:
                        if candidate in table_names and not col.is_foreign_key:
                            col.is_foreign_key = True
                            col.references = f"{candidate}.id"
                            col.semantic_type = ColumnType.FOREIGN_KEY
                            profile.foreign_keys.append({
                                "column": col.name,
                                "references": f"{candidate}.id",
                                "inferred": True,
                            })

                            # Mark reverse relationship
                            if candidate in profiles:
                                profiles[candidate].referenced_by.append(table_name)
                            break

    def _classify_tables(self, profiles: Dict[str, TableProfile]) -> None:
        """Classify tables by their type"""
        for table_name, profile in profiles.items():
            fk_count = len(profile.foreign_keys)
            col_count = len(profile.columns)

            # Junction table: mostly foreign keys
            if fk_count >= 2 and col_count <= fk_count + 3:
                profile.table_type = TableType.JUNCTION

            # Lookup table: few columns, few rows, mostly static
            elif col_count <= 4 and profile.row_count < 100:
                profile.table_type = TableType.LOOKUP

            # Log tables: contain timestamp + action patterns
            elif any(kw in table_name.lower() for kw in ["log", "audit", "history"]):
                profile.table_type = TableType.LOG

            # Transaction tables: contain date/amount patterns
            elif any(c.semantic_type in [ColumnType.AMOUNT, ColumnType.TIMESTAMP] for c in profile.columns):
                profile.table_type = TableType.TRANSACTION

            # Default: entity table
            else:
                profile.table_type = TableType.ENTITY

    async def _infer_column_semantics(
        self,
        profiles: Dict[str, TableProfile],
    ) -> None:
        """Infer semantic types for columns"""
        for profile in profiles.values():
            for col in profile.columns:
                if col.semantic_type not in [ColumnType.PRIMARY_KEY, ColumnType.FOREIGN_KEY]:
                    col.semantic_type = self._infer_semantic_type(col)
                    col.business_name = self._generate_business_name(col.name)

    def _infer_semantic_type(self, col: ColumnProfile) -> ColumnType:
        """Infer semantic type from column name and data type"""
        name_lower = col.name.lower()
        data_type = col.data_type.lower()

        # Date/time types
        if any(t in data_type for t in ["timestamp", "datetime"]):
            return ColumnType.TIMESTAMP
        if "date" in data_type:
            return ColumnType.DATE

        # Boolean
        if "bool" in data_type:
            return ColumnType.BOOLEAN

        # JSON
        if "json" in data_type:
            return ColumnType.JSON

        # Name patterns
        name_patterns = {
            ColumnType.EMAIL: ["email", "e_mail"],
            ColumnType.PHONE: ["phone", "mobile", "tel"],
            ColumnType.URL: ["url", "link", "website"],
            ColumnType.ADDRESS: ["address", "street", "city", "state", "zip", "country"],
            ColumnType.NAME: ["name", "first_name", "last_name", "full_name"],
            ColumnType.DESCRIPTION: ["description", "desc", "notes", "comment", "bio"],
            ColumnType.STATUS: ["status", "state", "is_", "has_"],
            ColumnType.CATEGORY: ["category", "type", "kind", "class"],
            ColumnType.AMOUNT: ["amount", "total", "sum", "balance"],
            ColumnType.PRICE: ["price", "cost", "fee", "rate"],
            ColumnType.QUANTITY: ["quantity", "qty", "count", "number"],
        }

        for semantic_type, patterns in name_patterns.items():
            if any(p in name_lower for p in patterns):
                return semantic_type

        # Data type fallbacks
        if any(t in data_type for t in ["int", "numeric", "decimal", "float"]):
            return ColumnType.NUMERIC

        return ColumnType.TEXT

    def _generate_business_name(self, column_name: str) -> str:
        """Convert column_name to Business Name"""
        # Replace underscores with spaces
        words = column_name.replace("_", " ").split()
        # Capitalize each word
        return " ".join(word.capitalize() for word in words)

    async def _generate_descriptions(
        self,
        profiles: Dict[str, TableProfile],
    ) -> None:
        """Use LLM to generate human-readable descriptions"""
        if not self.llm:
            return

        for table_name, profile in profiles.items():
            prompt = f"""
            Generate a brief, clear description for this database table:

            Table: {table_name}
            Columns: {', '.join(c.name for c in profile.columns)}
            Type: {profile.table_type.value}
            Row count: {profile.row_count}
            Foreign keys: {profile.foreign_keys}

            Respond with just the description (1-2 sentences):
            """

            try:
                response = await self.llm.generate(prompt=prompt, max_tokens=100)
                profile.description = response.strip()
            except Exception as e:
                logger.warning(f"Description generation failed for {table_name}: {e}")


class RelationshipInference:
    """
    Infers complex relationships between tables
    - Many-to-many through junction tables
    - Hierarchical relationships
    - Temporal relationships
    """

    def __init__(self, profiles: Dict[str, TableProfile]):
        self.profiles = profiles
        self.graph: Dict[str, List[Tuple[str, str]]] = {}  # table -> [(related_table, rel_type)]

    def build_relationship_graph(self) -> Dict[str, Any]:
        """Build complete relationship graph"""
        # Direct relationships
        for table_name, profile in self.profiles.items():
            self.graph[table_name] = []

            for fk in profile.foreign_keys:
                ref_table = fk["references"].split(".")[0]
                self.graph[table_name].append((ref_table, "belongs_to"))

        # Inverse relationships
        for table_name, profile in self.profiles.items():
            for fk in profile.foreign_keys:
                ref_table = fk["references"].split(".")[0]
                if ref_table in self.graph:
                    self.graph[ref_table].append((table_name, "has_many"))

        # Many-to-many through junction tables
        for table_name, profile in self.profiles.items():
            if profile.table_type == TableType.JUNCTION:
                # Find the two tables it connects
                connected = [fk["references"].split(".")[0] for fk in profile.foreign_keys]
                if len(connected) >= 2:
                    for i, t1 in enumerate(connected):
                        for t2 in connected[i+1:]:
                            if t1 in self.graph:
                                self.graph[t1].append((t2, f"many_to_many_via_{table_name}"))
                            if t2 in self.graph:
                                self.graph[t2].append((t1, f"many_to_many_via_{table_name}"))

        return self.graph

    def find_path(self, from_table: str, to_table: str) -> Optional[List[str]]:
        """Find join path between two tables using BFS"""
        if from_table not in self.graph or to_table not in self.graph:
            return None

        visited = {from_table}
        queue = [(from_table, [from_table])]

        while queue:
            current, path = queue.pop(0)

            for neighbor, _ in self.graph.get(current, []):
                if neighbor == to_table:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None


class DataProfiler:
    """
    Deep data profiling for understanding data patterns
    """

    def __init__(self, db_executor: Callable):
        self.db_executor = db_executor

    async def profile_column(
        self,
        table: str,
        column: str,
    ) -> Dict[str, Any]:
        """Get detailed statistics for a column"""
        stats = {}

        # Basic stats
        sql = f"""
        SELECT
            COUNT(*) as total_count,
            COUNT(DISTINCT "{column}") as distinct_count,
            COUNT(*) - COUNT("{column}") as null_count,
            MIN("{column}") as min_val,
            MAX("{column}") as max_val
        FROM {table}
        """

        try:
            result = await self.db_executor(sql)
            if result:
                stats.update(result[0])
        except:
            pass

        # Value distribution
        sql = f"""
        SELECT "{column}" as val, COUNT(*) as cnt
        FROM {table}
        WHERE "{column}" IS NOT NULL
        GROUP BY "{column}"
        ORDER BY cnt DESC
        LIMIT 10
        """

        try:
            result = await self.db_executor(sql)
            stats["top_values"] = list(result) if result else []
        except:
            pass

        return stats
