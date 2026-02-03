---
layout: default
title: Database Support - Agentic SQL
---

# Database Support

Agentic SQL auto-detects and adapts to any SQL database.

---

## Supported Databases

| Database | Adapter | Status | Auto-Detected Features |
|----------|---------|--------|------------------------|
| MS SQL Server | `MSSQLAdapter` | ✅ Ready | TOP, GETDATE, square brackets, T-SQL |
| PostgreSQL | `PostgreSQLAdapter` | ✅ Ready | LIMIT, NOW(), double quotes, PL/pgSQL |
| MySQL | `MySQLAdapter` | ✅ Ready | LIMIT, backticks, NOW(), AUTO_INCREMENT |
| SQLite | `SQLiteAdapter` | ✅ Ready | LIMIT, datetime functions |
| Snowflake | `SnowflakeAdapter` | 🔜 Coming | LIMIT, warehouse syntax |
| BigQuery | `BigQueryAdapter` | 🔜 Coming | LIMIT, backticks, standard SQL |
| Oracle | `OracleAdapter` | 🔜 Coming | ROWNUM, SYSDATE, PL/SQL |

---

## Auto-Detection

When you connect, the agent automatically:

1. **Probes the database** with test queries
2. **Detects the SQL dialect** (TOP vs LIMIT, etc.)
3. **Discovers schema** (tables, columns, types)
4. **Learns naming conventions** (PascalCase, snake_case)
5. **Identifies relationships** (foreign keys, joins)

```python
stats = await agent.connect(db_executor=db.execute)

# Output:
# {
#     "dialect": "mssql",
#     "tables": 25,
#     "schema_insights": 19,
# }
```

---

## MS SQL Server

### Installation

```bash
pip install pyodbc
# Also install ODBC Driver 17 or 18 for SQL Server
```

### Connection

```python
from src.database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

db = MSSQLAdapter(ConnectionConfig(
    name="my_mssql_db",
    db_type=DatabaseType.MSSQL,
    host="server.database.windows.net,1433",  # Include port
    database="MyDatabase",
    username="user",
    password="password",
))

await db.connect()
```

### Azure SQL Database

```python
db = MSSQLAdapter(ConnectionConfig(
    name="azure_db",
    db_type=DatabaseType.MSSQL,
    host="myserver.database.windows.net,1433",
    database="MyDatabase",
    username="admin@myserver",
    password="password",
))
```

### Windows Authentication

```python
db = MSSQLAdapter(ConnectionConfig(
    name="local_db",
    db_type=DatabaseType.MSSQL,
    host="localhost",
    database="MyDatabase",
    # Omit username/password for Windows Auth
))
```

### Auto-Detected Features

- `TOP` instead of `LIMIT`
- `GETDATE()` for current timestamp
- `[brackets]` for identifiers
- T-SQL specific functions
- `DATEPART`, `DATEDIFF` for dates

---

## PostgreSQL

### Installation

```bash
pip install -e ".[postgres]"
# Installs: asyncpg, psycopg2-binary
```

### Connection

```python
from src.database.multi_db import PostgreSQLAdapter, ConnectionConfig, DatabaseType

db = PostgreSQLAdapter(ConnectionConfig(
    name="my_postgres_db",
    db_type=DatabaseType.POSTGRESQL,
    host="localhost",
    port=5432,
    database="mydb",
    username="postgres",
    password="password",
))

await db.connect()
```

### Connection String

```python
db = PostgreSQLAdapter(ConnectionConfig(
    name="my_db",
    db_type=DatabaseType.POSTGRESQL,
    connection_string="postgresql://user:pass@localhost:5432/mydb",
))
```

### Auto-Detected Features

- `LIMIT` and `OFFSET`
- `NOW()` for current timestamp
- `"double quotes"` for identifiers
- Array types
- JSON/JSONB operations
- PL/pgSQL functions

---

## MySQL

### Installation

```bash
pip install -e ".[mysql]"
# Installs: aiomysql
```

### Connection

```python
from src.database.multi_db import MySQLAdapter, ConnectionConfig, DatabaseType

db = MySQLAdapter(ConnectionConfig(
    name="my_mysql_db",
    db_type=DatabaseType.MYSQL,
    host="localhost",
    port=3306,
    database="mydb",
    username="root",
    password="password",
))

await db.connect()
```

### Auto-Detected Features

- `LIMIT` syntax
- `` `backticks` `` for identifiers
- `NOW()` for current timestamp
- `AUTO_INCREMENT`
- MySQL-specific functions

---

## SQLite

### Installation

```bash
pip install -e ".[sqlite]"
# Installs: aiosqlite
```

### File Database

```python
from src.database.multi_db import SQLiteAdapter, ConnectionConfig, DatabaseType

db = SQLiteAdapter(ConnectionConfig(
    name="my_sqlite_db",
    db_type=DatabaseType.SQLITE,
    database="/path/to/database.db",
))

await db.connect()
```

### In-Memory Database

```python
db = SQLiteAdapter(ConnectionConfig(
    name="memory_db",
    db_type=DatabaseType.SQLITE,
    database=":memory:",
))

await db.connect()

# Create tables
await db.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT
    )
""")
```

### Auto-Detected Features

- `LIMIT` syntax
- SQLite datetime functions
- No strict types
- `ROWID` auto-increment

---

## Using with MetaAgent

Once connected, use any database the same way:

```python
from src.intelligence.meta_agent import MetaAgent

# Create agent
agent = MetaAgent(llm_client=llm)

# Connect (agent auto-discovers everything!)
stats = await agent.connect(db_executor=db.execute)

print(f"Dialect: {stats['dialect']}")
print(f"Tables: {stats['tables']}")

# Query works the same regardless of database
result = await agent.query("How many users are there?")
```

---

## Dialect-Specific SQL

The agent automatically generates correct SQL for each dialect:

### Counting Rows

| Question | MS SQL | PostgreSQL/MySQL/SQLite |
|----------|--------|-------------------------|
| "Show first 10 users" | `SELECT TOP 10 * FROM users` | `SELECT * FROM users LIMIT 10` |

### Date Functions

| Question | MS SQL | PostgreSQL | MySQL |
|----------|--------|------------|-------|
| "Orders today" | `WHERE order_date >= CAST(GETDATE() AS DATE)` | `WHERE order_date >= CURRENT_DATE` | `WHERE order_date >= CURDATE()` |

### Identifier Quoting

| Database | Style |
|----------|-------|
| MS SQL | `[TableName].[ColumnName]` |
| PostgreSQL | `"TableName"."ColumnName"` |
| MySQL | `` `TableName`.`ColumnName` `` |

---

## Connection Pooling

For production, use connection pooling:

```python
# PostgreSQL with asyncpg pool
db = PostgreSQLAdapter(ConnectionConfig(
    name="pooled_db",
    db_type=DatabaseType.POSTGRESQL,
    connection_string="postgresql://user:pass@localhost/db",
    pool_size=10,
    pool_max_overflow=20,
))
```

---

## Multiple Databases

You can connect to multiple databases:

```python
# Connect to MSSQL
mssql_db = MSSQLAdapter(mssql_config)
await mssql_db.connect()

# Connect to PostgreSQL
pg_db = PostgreSQLAdapter(pg_config)
await pg_db.connect()

# Create agents for each
mssql_agent = MetaAgent(llm_client=llm)
await mssql_agent.connect(db_executor=mssql_db.execute)

pg_agent = MetaAgent(llm_client=llm)
await pg_agent.connect(db_executor=pg_db.execute)

# Query each with correct dialect
mssql_result = await mssql_agent.query("Show top 10 orders")
pg_result = await pg_agent.query("Show top 10 orders")
```

---

<p align="center">
  <a href="memory-system">Memory System →</a>
</p>
