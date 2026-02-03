# Database Support

Agentic SQL works with any SQL database.

---

## Supported Databases

| Database | Adapter | Status |
|----------|---------|--------|
| [MS SQL Server](mssql.md) | `MSSQLAdapter` | :material-check-circle:{ .status-ready } Ready |
| [PostgreSQL](postgresql.md) | `PostgreSQLAdapter` | :material-check-circle:{ .status-ready } Ready |
| [MySQL](mysql.md) | `MySQLAdapter` | :material-check-circle:{ .status-ready } Ready |
| [SQLite](sqlite.md) | `SQLiteAdapter` | :material-check-circle:{ .status-ready } Ready |
| [Snowflake](snowflake.md) | `SnowflakeAdapter` | :material-check-circle:{ .status-ready } Ready |
| BigQuery | `BigQueryAdapter` | :material-clock:{ .status-coming } Coming |
| Redshift | `RedshiftAdapter` | :material-clock:{ .status-coming } Coming |

---

## Auto-Detection

The agent automatically detects your database type and adapts:

```python
stats = await agent.connect(db_executor=db.execute)
print(f"Detected: {stats['dialect']}")  # "mssql", "postgresql", etc.
```

---

## Dialect Differences

| Feature | MS SQL | PostgreSQL | MySQL | SQLite | Snowflake |
|---------|--------|------------|-------|--------|-----------|
| Row limit | `TOP n` | `LIMIT n` | `LIMIT n` | `LIMIT n` | `LIMIT n` |
| Current date | `GETDATE()` | `NOW()` | `NOW()` | `datetime('now')` | `CURRENT_TIMESTAMP()` |
| Identifiers | `[brackets]` | `"quotes"` | `` `backticks` `` | `"quotes"` | `"QUOTES"` |
| Boolean | `1/0` | `true/false` | `1/0` | `1/0` | `true/false` |
| String concat | `+` | `\|\|` | `CONCAT()` | `\|\|` | `\|\|` or `CONCAT()` |

---

## Quick Start

=== "MS SQL Server"

    ```python
    from src.database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

    db = MSSQLAdapter(ConnectionConfig(
        name="mydb",
        db_type=DatabaseType.MSSQL,
        host="server.database.windows.net,1433",
        database="MyDB",
        username="user",
        password="pass",
    ))
    await db.connect()
    ```

=== "PostgreSQL"

    ```python
    from src.database.multi_db import PostgreSQLAdapter, ConnectionConfig, DatabaseType

    db = PostgreSQLAdapter(ConnectionConfig(
        name="mydb",
        db_type=DatabaseType.POSTGRESQL,
        host="localhost",
        port=5432,
        database="mydb",
        username="postgres",
        password="pass",
    ))
    await db.connect()
    ```

=== "MySQL"

    ```python
    from src.database.multi_db import MySQLAdapter, ConnectionConfig, DatabaseType

    db = MySQLAdapter(ConnectionConfig(
        name="mydb",
        db_type=DatabaseType.MYSQL,
        host="localhost",
        port=3306,
        database="mydb",
        username="root",
        password="pass",
    ))
    await db.connect()
    ```

=== "SQLite"

    ```python
    from src.database.multi_db import SQLiteAdapter, ConnectionConfig, DatabaseType

    db = SQLiteAdapter(ConnectionConfig(
        name="mydb",
        db_type=DatabaseType.SQLITE,
        database="/path/to/db.sqlite",  # or ":memory:"
    ))
    await db.connect()
    ```

=== "Snowflake"

    ```python
    from src.database.multi_db import SnowflakeAdapter, ConnectionConfig, DatabaseType

    db = SnowflakeAdapter(ConnectionConfig(
        name="mydb",
        db_type=DatabaseType.SNOWFLAKE,
        database="MY_DATABASE",
        username="user",
        password="pass",
        options={
            "account": "abc12345.us-east-1",
            "warehouse": "COMPUTE_WH",
            "role": "ANALYST",
        },
    ))
    await db.connect()
    ```
