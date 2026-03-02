"""
Snowflake Client - DDL, query, and metadata operations for the dimensional model.

Provides Snowflake-specific functionality for the lowest semantic layer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SnowflakeConfig:
    """Configuration for Snowflake connection"""
    account: str = ""
    user: str = ""
    password: str = ""
    warehouse: str = "COMPUTE_WH"
    database: str = "ANALYTICS"
    schema: str = "PUBLIC"
    role: str = "ANALYST"

    # Optional authentication
    authenticator: Optional[str] = None  # "externalbrowser", "snowflake_jwt"
    private_key_path: Optional[str] = None


class SnowflakeClient:
    """
    Snowflake operations client for the dimensional model layer.

    Provides:
    - Schema introspection (table/column metadata)
    - Query execution with result formatting
    - DDL generation and execution
    - Metadata retrieval for LLM context
    """

    def __init__(self, config: SnowflakeConfig):
        self.config = config
        self._connection = None

    async def connect(self) -> None:
        """Establish Snowflake connection"""
        try:
            import snowflake.connector

            loop = asyncio.get_event_loop()

            def _connect():
                params = {
                    "account": self.config.account,
                    "user": self.config.user,
                    "password": self.config.password,
                    "warehouse": self.config.warehouse,
                    "database": self.config.database,
                    "schema": self.config.schema,
                    "role": self.config.role,
                }
                if self.config.authenticator:
                    params["authenticator"] = self.config.authenticator

                return snowflake.connector.connect(**params)

            self._connection = await loop.run_in_executor(None, _connect)
            logger.info(f"Connected to Snowflake: {self.config.account}")

        except ImportError:
            raise ImportError(
                "snowflake-connector-python required: "
                "pip install snowflake-connector-python"
            )

    async def disconnect(self) -> None:
        """Close Snowflake connection"""
        if self._connection:
            self._connection.close()

    async def execute_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results as dicts"""
        loop = asyncio.get_event_loop()

        def _execute():
            cursor = self._connection.cursor()
            try:
                cursor.execute(sql, params or {})
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            finally:
                cursor.close()

        return await loop.run_in_executor(None, _execute)

    async def get_table_metadata(
        self,
        schema: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get metadata for all tables in a schema"""
        target_schema = schema or self.config.schema
        sql = f"""
        SELECT
            TABLE_NAME,
            TABLE_TYPE,
            ROW_COUNT,
            BYTES,
            COMMENT
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = '{target_schema}'
        ORDER BY TABLE_NAME
        """
        return await self.execute_query(sql)

    async def get_column_metadata(
        self,
        table_name: str,
        schema: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get column metadata for a table"""
        target_schema = schema or self.config.schema
        sql = f"""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            COMMENT,
            ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{target_schema}'
          AND TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """
        return await self.execute_query(sql)

    async def get_schema_context(
        self,
        schema: Optional[str] = None,
        include_comments: bool = True,
    ) -> str:
        """
        Build schema context for LLM consumption.

        Returns a formatted string describing all tables and columns.
        """
        tables = await self.get_table_metadata(schema)
        lines = [f"Database: {self.config.database}", f"Schema: {schema or self.config.schema}", ""]

        for table in tables:
            table_name = table["TABLE_NAME"]
            comment = table.get("COMMENT", "") or ""
            row_count = table.get("ROW_COUNT", "?")
            lines.append(f"TABLE: {table_name} (~{row_count} rows)")
            if comment and include_comments:
                lines.append(f"  Description: {comment}")

            columns = await self.get_column_metadata(table_name, schema)
            for col in columns:
                nullable = "NULL" if col["IS_NULLABLE"] == "YES" else "NOT NULL"
                col_comment = col.get("COMMENT", "") or ""
                desc = f" -- {col_comment}" if col_comment and include_comments else ""
                lines.append(
                    f"  - {col['COLUMN_NAME']}: {col['DATA_TYPE']} {nullable}{desc}"
                )
            lines.append("")

        return "\n".join(lines)

    async def get_sample_data(
        self,
        table_name: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get sample rows from a table"""
        sql = f"SELECT * FROM {table_name} LIMIT {limit}"
        return await self.execute_query(sql)

    async def validate_sql(self, sql: str) -> List[str]:
        """Validate SQL using EXPLAIN"""
        errors = []
        try:
            await self.execute_query(f"EXPLAIN {sql}")
        except Exception as e:
            errors.append(str(e))
        return errors
