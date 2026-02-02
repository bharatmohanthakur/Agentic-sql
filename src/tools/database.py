"""
Database Tools - SQL execution, schema retrieval, validation
Enterprise-grade with row-level security and audit logging
"""
from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from ..core.base import UserContext
from ..core.registry import (
    Tool,
    ToolCategory,
    ToolResult,
    ToolSchema,
    PermissionLevel,
)

logger = logging.getLogger(__name__)


class DatabaseConnection(Protocol):
    """Protocol for database connections"""

    async def execute(self, sql: str, params: Optional[Dict] = None) -> List[Dict]:
        ...

    async def fetch_schema(self) -> Dict[str, Any]:
        ...

    async def explain(self, sql: str) -> str:
        ...


class GetSchemaTool(Tool):
    """
    Retrieves database schema information
    Respects user permissions for schema visibility
    """

    name = "get_schema"
    description = "Get database schema including tables, columns, and relationships"
    category = ToolCategory.DATABASE
    permission_level = PermissionLevel.AUTHENTICATED

    def __init__(self, db_connection: DatabaseConnection):
        super().__init__()
        self.db = db_connection
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "include_columns": {
                    "type": "boolean",
                    "description": "Include column details",
                    "default": True,
                },
                "include_relationships": {
                    "type": "boolean",
                    "description": "Include foreign key relationships",
                    "default": True,
                },
                "table_filter": {
                    "type": "string",
                    "description": "Filter tables by pattern",
                },
            },
            required=[],
        )

    async def execute(
        self,
        user_context: UserContext,
        include_columns: bool = True,
        include_relationships: bool = True,
        table_filter: Optional[str] = None,
    ) -> ToolResult:
        try:
            # Check cache
            cache_key = f"{include_columns}_{include_relationships}_{table_filter}"
            if self._is_cache_valid(cache_key):
                return ToolResult(success=True, data=self._cache[cache_key])

            # Fetch schema
            raw_schema = await self.db.fetch_schema()

            # Filter tables based on user permissions
            allowed_tables = self._filter_by_permissions(
                raw_schema.get("tables", []),
                user_context,
            )

            # Apply table filter pattern
            if table_filter:
                pattern = re.compile(table_filter, re.IGNORECASE)
                allowed_tables = [t for t in allowed_tables if pattern.search(t["name"])]

            result = {
                "tables": allowed_tables,
                "columns": {},
                "relationships": [],
            }

            if include_columns:
                for table in allowed_tables:
                    table_name = table["name"]
                    result["columns"][table_name] = raw_schema.get("columns", {}).get(
                        table_name, []
                    )

            if include_relationships:
                result["relationships"] = self._filter_relationships(
                    raw_schema.get("relationships", []),
                    [t["name"] for t in allowed_tables],
                )

            # Update cache
            self._cache[cache_key] = result
            self._cache_time = datetime.utcnow()

            return ToolResult(success=True, data=result)

        except Exception as e:
            logger.exception("Schema retrieval failed")
            return ToolResult(success=False, error=str(e))

    def _is_cache_valid(self, cache_key: str) -> bool:
        if cache_key not in self._cache or self._cache_time is None:
            return False
        elapsed = (datetime.utcnow() - self._cache_time).total_seconds()
        return elapsed < self._cache_ttl_seconds

    def _filter_by_permissions(
        self,
        tables: List[Dict],
        user_context: UserContext,
    ) -> List[Dict]:
        """Filter tables based on user permissions"""
        if "admin" in user_context.roles:
            return tables

        allowed_schemas = user_context.permissions.get("allowed_schemas", [])
        allowed_tables = user_context.permissions.get("allowed_tables", [])

        if not allowed_schemas and not allowed_tables:
            return tables  # No restrictions

        filtered = []
        for table in tables:
            table_name = table["name"]
            schema_name = table.get("schema", "public")

            if schema_name in allowed_schemas or table_name in allowed_tables:
                filtered.append(table)

        return filtered

    def _filter_relationships(
        self,
        relationships: List[Dict],
        allowed_tables: List[str],
    ) -> List[Dict]:
        """Filter relationships to only include allowed tables"""
        return [
            rel for rel in relationships
            if rel.get("from_table") in allowed_tables
            and rel.get("to_table") in allowed_tables
        ]


class ExecuteSQLTool(Tool):
    """
    Executes SQL queries with security checks and row-level security
    """

    name = "execute_sql"
    description = "Execute a SQL query and return results"
    category = ToolCategory.DATABASE
    permission_level = PermissionLevel.AUTHENTICATED
    rate_limit = 60  # 60 queries per minute per user

    # Blocked patterns for security
    BLOCKED_PATTERNS = [
        r'\bDROP\b',
        r'\bDELETE\b',
        r'\bTRUNCATE\b',
        r'\bUPDATE\b',
        r'\bINSERT\b',
        r'\bALTER\b',
        r'\bCREATE\b',
        r'\bGRANT\b',
        r'\bREVOKE\b',
        r'\bEXEC\b',
        r'\bEXECUTE\b',
        r'--',  # SQL comments (potential injection)
        r'/\*',  # Block comments
    ]

    def __init__(
        self,
        db_connection: DatabaseConnection,
        max_rows: int = 1000,
        timeout_seconds: float = 30.0,
        allow_writes: bool = False,
    ):
        super().__init__()
        self.db = db_connection
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        self.allow_writes = allow_writes

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "sql": {
                    "type": "string",
                    "description": "The SQL query to execute",
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters for prepared statements",
                },
            },
            required=["sql"],
        )

    async def execute(
        self,
        user_context: UserContext,
        sql: str,
        params: Optional[Dict] = None,
    ) -> ToolResult:
        start_time = datetime.utcnow()

        # Security validation
        validation_error = self._validate_sql(sql, user_context)
        if validation_error:
            return ToolResult(success=False, error=validation_error)

        # Apply row-level security
        sql = self._apply_rls(sql, user_context)

        # Add LIMIT if not present
        sql = self._ensure_limit(sql)

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.db.execute(sql, params),
                timeout=self.timeout_seconds,
            )

            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Process results
            data = list(result) if result else []
            columns = list(data[0].keys()) if data else []
            truncated = len(data) > self.max_rows

            if truncated:
                data = data[:self.max_rows]

            # Audit log
            logger.info(
                f"SQL executed | User: {user_context.user_id} | "
                f"Rows: {len(data)} | Time: {execution_time:.2f}ms"
            )

            return ToolResult(
                success=True,
                data={
                    "data": data,
                    "columns": columns,
                    "row_count": len(data),
                    "truncated": truncated,
                    "execution_time_ms": execution_time,
                },
            )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Query timeout after {self.timeout_seconds}s",
            )
        except Exception as e:
            logger.exception("SQL execution failed")
            return ToolResult(success=False, error=str(e))

    def _validate_sql(self, sql: str, user_context: UserContext) -> Optional[str]:
        """Validate SQL for security issues"""
        sql_upper = sql.upper()

        # Check blocked patterns (unless admin and writes allowed)
        if not (self.allow_writes and "admin" in user_context.roles):
            for pattern in self.BLOCKED_PATTERNS:
                if re.search(pattern, sql, re.IGNORECASE):
                    return f"Operation not allowed: {pattern}"

        # Check for multiple statements
        if sql.count(';') > 1:
            return "Multiple statements not allowed"

        return None

    def _apply_rls(self, sql: str, user_context: UserContext) -> str:
        """Apply row-level security filters"""
        filters = user_context.get_sql_filters()
        if not filters:
            return sql

        # Parse and inject WHERE clauses
        for table, conditions in filters.items():
            for column, value in conditions.items():
                # Escape value to prevent injection
                escaped_value = str(value).replace("'", "''")
                filter_clause = f"{table}.{column} = '{escaped_value}'"

                # Inject into query
                if re.search(r'\bWHERE\b', sql, re.IGNORECASE):
                    sql = re.sub(
                        r'\bWHERE\b',
                        f'WHERE {filter_clause} AND ',
                        sql,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                else:
                    # Add WHERE before ORDER BY, GROUP BY, LIMIT, or end
                    sql = re.sub(
                        r'(\s+ORDER\s+BY|\s+GROUP\s+BY|\s+LIMIT|\s+HAVING|\s*$)',
                        f' WHERE {filter_clause} \\1',
                        sql,
                        count=1,
                        flags=re.IGNORECASE,
                    )

        return sql

    def _ensure_limit(self, sql: str) -> str:
        """Ensure query has a LIMIT clause"""
        if not re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
            sql = f"{sql.rstrip().rstrip(';')} LIMIT {self.max_rows + 1}"
        return sql


class ValidateSQLTool(Tool):
    """
    Validates SQL syntax without executing
    """

    name = "validate_sql"
    description = "Validate SQL syntax and check for potential issues"
    category = ToolCategory.VALIDATION
    permission_level = PermissionLevel.AUTHENTICATED

    def __init__(self, db_connection: DatabaseConnection):
        super().__init__()
        self.db = db_connection

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "sql": {
                    "type": "string",
                    "description": "The SQL query to validate",
                },
            },
            required=["sql"],
        )

    async def execute(
        self,
        user_context: UserContext,
        sql: str,
    ) -> ToolResult:
        issues = []

        # Syntax check via EXPLAIN
        try:
            await self.db.explain(sql)
        except Exception as e:
            issues.append(f"Syntax error: {e}")

        # Check for common issues
        issues.extend(self._check_common_issues(sql))

        return ToolResult(
            success=len(issues) == 0,
            data={
                "valid": len(issues) == 0,
                "issues": issues,
            },
        )

    def _check_common_issues(self, sql: str) -> List[str]:
        """Check for common SQL issues"""
        issues = []

        # SELECT * warning
        if re.search(r'SELECT\s+\*', sql, re.IGNORECASE):
            issues.append("Warning: SELECT * may return unnecessary columns")

        # Missing WHERE on large tables
        if not re.search(r'\bWHERE\b', sql, re.IGNORECASE):
            if not re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
                issues.append("Warning: No WHERE or LIMIT clause")

        # Cartesian join detection
        if re.search(r',\s*\w+\s+WHERE', sql, re.IGNORECASE):
            if not re.search(r'JOIN|=', sql, re.IGNORECASE):
                issues.append("Warning: Possible cartesian product")

        return issues


class ExplainQueryTool(Tool):
    """
    Explains query execution plan
    """

    name = "explain_query"
    description = "Get the execution plan for a SQL query"
    category = ToolCategory.ANALYSIS
    permission_level = PermissionLevel.ELEVATED
    required_roles = ["analyst", "admin"]

    def __init__(self, db_connection: DatabaseConnection):
        super().__init__()
        self.db = db_connection

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "sql": {
                    "type": "string",
                    "description": "The SQL query to explain",
                },
                "analyze": {
                    "type": "boolean",
                    "description": "Run EXPLAIN ANALYZE for actual execution stats",
                    "default": False,
                },
            },
            required=["sql"],
        )

    async def execute(
        self,
        user_context: UserContext,
        sql: str,
        analyze: bool = False,
    ) -> ToolResult:
        try:
            explain_sql = f"EXPLAIN {'ANALYZE ' if analyze else ''}{sql}"
            result = await self.db.explain(explain_sql)

            return ToolResult(
                success=True,
                data={
                    "plan": result,
                    "analyzed": analyze,
                },
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))
