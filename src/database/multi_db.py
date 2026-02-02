"""
Multi-Database Manager - Unified interface for multiple databases
Supports 100+ tables across multiple database types
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    CLICKHOUSE = "clickhouse"
    DUCKDB = "duckdb"
    MSSQL = "mssql"
    ORACLE = "oracle"


class ConnectionConfig(BaseModel):
    """Database connection configuration"""
    name: str  # Unique identifier
    db_type: DatabaseType
    host: str = "localhost"
    port: int = 5432
    database: str = ""
    username: str = ""
    password: str = ""

    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0

    # SSL settings
    ssl_enabled: bool = False
    ssl_ca: Optional[str] = None

    # Additional options
    options: Dict[str, Any] = Field(default_factory=dict)

    def get_connection_string(self) -> str:
        """Generate connection string"""
        if self.db_type == DatabaseType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == DatabaseType.MYSQL:
            return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == DatabaseType.SQLITE:
            return f"sqlite:///{self.database}"
        else:
            return f"{self.db_type.value}://{self.host}:{self.port}/{self.database}"


class DatabaseAdapter(ABC):
    """Abstract database adapter"""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._connection = None

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection"""
        pass

    @abstractmethod
    async def execute(
        self,
        sql: str,
        params: Optional[Dict] = None,
    ) -> List[Dict]:
        """Execute query and return results"""
        pass

    @abstractmethod
    async def get_schema(self) -> Dict[str, Any]:
        """Get database schema"""
        pass

    @abstractmethod
    async def explain(self, sql: str) -> str:
        """Get query execution plan"""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if connected"""
        return self._connection is not None


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL database adapter"""

    async def connect(self) -> None:
        try:
            import asyncpg

            self._connection = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
            )
            logger.info(f"Connected to PostgreSQL: {self.config.database}")
        except ImportError:
            raise ImportError("asyncpg required: pip install asyncpg")

    async def disconnect(self) -> None:
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def execute(
        self,
        sql: str,
        params: Optional[Dict] = None,
    ) -> List[Dict]:
        if not self._connection:
            await self.connect()

        result = await self._connection.fetch(sql)
        return [dict(row) for row in result]

    async def get_schema(self) -> Dict[str, Any]:
        tables_sql = """
        SELECT table_name, table_schema
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        AND table_type = 'BASE TABLE'
        """

        tables = await self.execute(tables_sql)

        columns_sql = """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        """

        columns = await self.execute(columns_sql)

        # Group columns by table
        columns_by_table = {}
        for col in columns:
            table = col["table_name"]
            if table not in columns_by_table:
                columns_by_table[table] = []
            columns_by_table[table].append(col)

        return {
            "tables": tables,
            "columns": columns_by_table,
        }

    async def explain(self, sql: str) -> str:
        result = await self.execute(f"EXPLAIN {sql}")
        return "\n".join(str(row) for row in result)


class MySQLAdapter(DatabaseAdapter):
    """MySQL database adapter"""

    async def connect(self) -> None:
        try:
            import aiomysql

            self._connection = await aiomysql.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                db=self.config.database,
            )
            logger.info(f"Connected to MySQL: {self.config.database}")
        except ImportError:
            raise ImportError("aiomysql required: pip install aiomysql")

    async def disconnect(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    async def execute(
        self,
        sql: str,
        params: Optional[Dict] = None,
    ) -> List[Dict]:
        if not self._connection:
            await self.connect()

        async with self._connection.cursor() as cursor:
            await cursor.execute(sql)
            columns = [d[0] for d in cursor.description or []]
            rows = await cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]

    async def get_schema(self) -> Dict[str, Any]:
        tables = await self.execute("SHOW TABLES")
        return {"tables": tables, "columns": {}}

    async def explain(self, sql: str) -> str:
        result = await self.execute(f"EXPLAIN {sql}")
        return str(result)


class SQLiteAdapter(DatabaseAdapter):
    """SQLite database adapter"""

    async def connect(self) -> None:
        try:
            import aiosqlite

            self._connection = await aiosqlite.connect(self.config.database)
            logger.info(f"Connected to SQLite: {self.config.database}")
        except ImportError:
            raise ImportError("aiosqlite required: pip install aiosqlite")

    async def disconnect(self) -> None:
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def execute(
        self,
        sql: str,
        params: Optional[Dict] = None,
    ) -> List[Dict]:
        if not self._connection:
            await self.connect()

        self._connection.row_factory = lambda c, r: dict(
            zip([col[0] for col in c.description], r)
        )

        async with self._connection.execute(sql) as cursor:
            return await cursor.fetchall()

    async def get_schema(self) -> Dict[str, Any]:
        tables = await self.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return {"tables": tables, "columns": {}}

    async def explain(self, sql: str) -> str:
        result = await self.execute(f"EXPLAIN QUERY PLAN {sql}")
        return str(result)


class MSSQLAdapter(DatabaseAdapter):
    """MS SQL Server database adapter using pyodbc"""

    async def connect(self) -> None:
        try:
            import pyodbc
            import asyncio

            # Build ODBC connection string
            conn_str = (
                f'DRIVER={{ODBC Driver 18 for SQL Server}};'
                f'SERVER={self.config.host};'
                f'DATABASE={self.config.database};'
                f'UID={self.config.username};'
                f'PWD={self.config.password};'
                'TrustServerCertificate=yes;'
            )

            # Run in executor since pyodbc is sync
            loop = asyncio.get_event_loop()
            self._connection = await loop.run_in_executor(
                None, lambda: pyodbc.connect(conn_str)
            )
            logger.info(f"Connected to MS SQL Server: {self.config.database}")

        except ImportError:
            raise ImportError("pyodbc required: pip install pyodbc")

    async def disconnect(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    async def execute(
        self,
        sql: str,
        params: Optional[Dict] = None,
    ) -> List[Dict]:
        import asyncio

        if not self._connection:
            await self.connect()

        def _execute():
            cursor = self._connection.cursor()
            cursor.execute(sql)

            # Check if query returns results
            if cursor.description:
                columns = [column[0] for column in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _execute)

    async def get_schema(self) -> Dict[str, Any]:
        """Get MSSQL schema including views"""
        # Get tables and views
        tables_sql = """
        SELECT TABLE_NAME as table_name, TABLE_SCHEMA as table_schema, TABLE_TYPE as table_type
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = 'dbo'
        """
        tables = await self.execute(tables_sql)

        # Get columns
        columns_sql = """
        SELECT TABLE_NAME as table_name, COLUMN_NAME as column_name,
               DATA_TYPE as data_type, IS_NULLABLE as is_nullable,
               CHARACTER_MAXIMUM_LENGTH as max_length
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'dbo'
        """
        columns = await self.execute(columns_sql)

        # Group columns by table
        columns_by_table = {}
        for col in columns:
            table = col["table_name"]
            if table not in columns_by_table:
                columns_by_table[table] = []
            columns_by_table[table].append(col)

        return {
            "tables": tables,
            "columns": columns_by_table,
        }

    async def explain(self, sql: str) -> str:
        # MSSQL uses SET SHOWPLAN_TEXT for explain
        result = await self.execute(f"SET SHOWPLAN_TEXT ON; {sql}; SET SHOWPLAN_TEXT OFF;")
        return str(result)


@dataclass
class DatabaseStats:
    """Statistics for a database"""
    name: str
    db_type: DatabaseType
    table_count: int = 0
    total_rows: int = 0
    query_count: int = 0
    avg_query_time_ms: float = 0.0
    last_query_time: Optional[datetime] = None
    is_healthy: bool = True


class MultiDatabaseManager:
    """
    Manages multiple database connections

    Features:
    - Unified query interface across databases
    - Automatic connection pooling
    - Cross-database query routing
    - Health monitoring
    - Scalable to 100+ tables
    """

    # Adapter mapping
    ADAPTERS = {
        DatabaseType.POSTGRESQL: PostgreSQLAdapter,
        DatabaseType.MYSQL: MySQLAdapter,
        DatabaseType.SQLITE: SQLiteAdapter,
        DatabaseType.MSSQL: MSSQLAdapter,
    }

    def __init__(self):
        self._databases: Dict[str, DatabaseAdapter] = {}
        self._configs: Dict[str, ConnectionConfig] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, DatabaseStats] = {}
        self._table_to_db: Dict[str, str] = {}  # table_name -> db_name

    async def add_database(self, config: ConnectionConfig) -> None:
        """Add a new database connection"""
        adapter_class = self.ADAPTERS.get(config.db_type)

        if not adapter_class:
            raise ValueError(f"Unsupported database type: {config.db_type}")

        adapter = adapter_class(config)
        await adapter.connect()

        self._databases[config.name] = adapter
        self._configs[config.name] = config
        self._stats[config.name] = DatabaseStats(
            name=config.name,
            db_type=config.db_type,
        )

        # Discover schema
        await self._discover_schema(config.name)

        logger.info(f"Added database: {config.name} ({config.db_type.value})")

    async def remove_database(self, name: str) -> None:
        """Remove a database connection"""
        if name in self._databases:
            await self._databases[name].disconnect()
            del self._databases[name]
            del self._configs[name]
            del self._schemas[name]
            del self._stats[name]

            # Remove table mappings
            self._table_to_db = {
                t: d for t, d in self._table_to_db.items() if d != name
            }

    async def _discover_schema(self, db_name: str) -> None:
        """Discover schema for a database"""
        adapter = self._databases[db_name]
        schema = await adapter.get_schema()

        self._schemas[db_name] = schema

        # Update table mappings
        for table in schema.get("tables", []):
            table_name = table.get("table_name") or table.get("name")
            if table_name:
                # Handle duplicates with prefix
                if table_name in self._table_to_db:
                    prefixed = f"{db_name}.{table_name}"
                    self._table_to_db[prefixed] = db_name
                else:
                    self._table_to_db[table_name] = db_name

        # Update stats
        self._stats[db_name].table_count = len(schema.get("tables", []))

    async def execute(
        self,
        sql: str,
        db_name: Optional[str] = None,
        params: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Execute query on appropriate database

        If db_name not specified, routes based on tables in query
        """
        start_time = datetime.utcnow()

        # Determine target database
        if db_name:
            target_db = db_name
        else:
            target_db = self._route_query(sql)

        if not target_db or target_db not in self._databases:
            raise ValueError(f"Cannot determine target database for query")

        adapter = self._databases[target_db]

        try:
            result = await adapter.execute(sql, params)

            # Update stats
            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
            stats = self._stats[target_db]
            stats.query_count += 1
            stats.avg_query_time_ms = (
                (stats.avg_query_time_ms * (stats.query_count - 1) + elapsed)
                / stats.query_count
            )
            stats.last_query_time = datetime.utcnow()

            return result

        except Exception as e:
            self._stats[target_db].is_healthy = False
            raise

    def _route_query(self, sql: str) -> Optional[str]:
        """Determine which database to route query to"""
        import re

        # Extract table names from SQL
        sql_upper = sql.upper()

        # Find tables after FROM and JOIN
        table_patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'INTO\s+(\w+)',
            r'UPDATE\s+(\w+)',
        ]

        tables = set()
        for pattern in table_patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.update(m.lower() for m in matches)

        # Find which database has these tables
        db_matches = {}
        for table in tables:
            if table in self._table_to_db:
                db = self._table_to_db[table]
                db_matches[db] = db_matches.get(db, 0) + 1

        if db_matches:
            # Return database with most matching tables
            return max(db_matches, key=db_matches.get)

        # Default to first database
        return next(iter(self._databases.keys()), None)

    async def get_unified_schema(self) -> Dict[str, Any]:
        """Get unified schema across all databases"""
        unified = {
            "databases": {},
            "tables": [],
            "columns": {},
        }

        for db_name, schema in self._schemas.items():
            unified["databases"][db_name] = {
                "type": self._configs[db_name].db_type.value,
                "table_count": len(schema.get("tables", [])),
            }

            for table in schema.get("tables", []):
                table_name = table.get("table_name") or table.get("name")
                if table_name:
                    unified["tables"].append({
                        "name": table_name,
                        "database": db_name,
                        "schema": table.get("table_schema", "public"),
                    })

                    # Columns
                    table_cols = schema.get("columns", {}).get(table_name, [])
                    unified["columns"][f"{db_name}.{table_name}"] = table_cols

        return unified

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all connections"""
        results = {}

        for db_name, adapter in self._databases.items():
            try:
                # Simple query to test connection
                await adapter.execute("SELECT 1")
                self._stats[db_name].is_healthy = True
                results[db_name] = True
            except Exception as e:
                logger.warning(f"Health check failed for {db_name}: {e}")
                self._stats[db_name].is_healthy = False
                results[db_name] = False

        return results

    def get_stats(self) -> Dict[str, DatabaseStats]:
        """Get statistics for all databases"""
        return self._stats

    async def refresh_schemas(self) -> None:
        """Refresh schemas for all databases"""
        for db_name in self._databases:
            await self._discover_schema(db_name)

        logger.info(f"Refreshed schemas for {len(self._databases)} databases")
