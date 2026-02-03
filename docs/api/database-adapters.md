# Database Adapters

Connect to different database types.

---

## ConnectionConfig

Base configuration for all database connections.

```python
from src.database.multi_db import ConnectionConfig, DatabaseType

config = ConnectionConfig(
    name: str,                         # Connection identifier
    db_type: DatabaseType,             # Database type enum
    host: Optional[str] = None,        # Server hostname
    port: Optional[int] = None,        # Server port
    database: str,                     # Database name
    username: Optional[str] = None,    # Username
    password: Optional[str] = None,    # Password
    connection_string: Optional[str] = None,  # Alternative to host/port
)
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier for the connection |
| `db_type` | `DatabaseType` | Yes | Database type enum |
| `host` | `str` | No | Server hostname |
| `port` | `int` | No | Server port |
| `database` | `str` | Yes | Database name or file path |
| `username` | `str` | No | Database username |
| `password` | `str` | No | Database password |
| `connection_string` | `str` | No | Full connection string (alternative) |

---

## DatabaseType

```python
from src.database.multi_db import DatabaseType

DatabaseType.MSSQL       # MS SQL Server
DatabaseType.POSTGRESQL  # PostgreSQL
DatabaseType.MYSQL       # MySQL
DatabaseType.SQLITE      # SQLite
DatabaseType.SNOWFLAKE   # Snowflake
DatabaseType.BIGQUERY    # Google BigQuery (coming soon)
DatabaseType.REDSHIFT    # Amazon Redshift (coming soon)
DatabaseType.CLICKHOUSE  # ClickHouse (coming soon)
```

---

## MSSQLAdapter

### Class Definition

```python
from src.database.multi_db import MSSQLAdapter

adapter = MSSQLAdapter(config: ConnectionConfig)
```

### Methods

#### connect()

```python
async def connect() -> None
```

Establish connection to MS SQL Server.

#### execute()

```python
async def execute(query: str, params: tuple = None) -> List[Dict]
```

Execute SQL query and return results.

### Example

```python
from src.database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

db = MSSQLAdapter(ConnectionConfig(
    name="production",
    db_type=DatabaseType.MSSQL,
    host="server.database.windows.net,1433",
    database="MyDatabase",
    username="admin",
    password="password",
))

await db.connect()

# Execute query
results = await db.execute("SELECT TOP 10 * FROM users")
for row in results:
    print(row)
```

### Azure SQL Database

```python
db = MSSQLAdapter(ConnectionConfig(
    name="azure",
    db_type=DatabaseType.MSSQL,
    host="myserver.database.windows.net,1433",
    database="MyDB",
    username="admin@myserver",
    password="password",
))
```

### Windows Authentication

```python
db = MSSQLAdapter(ConnectionConfig(
    name="local",
    db_type=DatabaseType.MSSQL,
    host="localhost",
    database="MyDB",
    # Omit username/password for Windows Auth
))
```

---

## PostgreSQLAdapter

### Example

```python
from src.database.multi_db import PostgreSQLAdapter, ConnectionConfig, DatabaseType

db = PostgreSQLAdapter(ConnectionConfig(
    name="postgres",
    db_type=DatabaseType.POSTGRESQL,
    host="localhost",
    port=5432,
    database="mydb",
    username="postgres",
    password="password",
))

await db.connect()
results = await db.execute("SELECT * FROM users LIMIT 10")
```

### Connection String

```python
db = PostgreSQLAdapter(ConnectionConfig(
    name="postgres",
    db_type=DatabaseType.POSTGRESQL,
    connection_string="postgresql://user:pass@localhost:5432/mydb",
))
```

---

## MySQLAdapter

### Example

```python
from src.database.multi_db import MySQLAdapter, ConnectionConfig, DatabaseType

db = MySQLAdapter(ConnectionConfig(
    name="mysql",
    db_type=DatabaseType.MYSQL,
    host="localhost",
    port=3306,
    database="mydb",
    username="root",
    password="password",
))

await db.connect()
results = await db.execute("SELECT * FROM users LIMIT 10")
```

---

## SQLiteAdapter

### File Database

```python
from src.database.multi_db import SQLiteAdapter, ConnectionConfig, DatabaseType

db = SQLiteAdapter(ConnectionConfig(
    name="sqlite",
    db_type=DatabaseType.SQLITE,
    database="/path/to/database.db",
))

await db.connect()
```

### In-Memory Database

```python
db = SQLiteAdapter(ConnectionConfig(
    name="memory",
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

---

## SnowflakeAdapter

### Example

```python
from src.database.multi_db import SnowflakeAdapter, ConnectionConfig, DatabaseType

db = SnowflakeAdapter(ConnectionConfig(
    name="snowflake",
    db_type=DatabaseType.SNOWFLAKE,
    database="MY_DATABASE",
    username="my_user",
    password="my_password",
    options={
        "account": "abc12345.us-east-1",
        "warehouse": "COMPUTE_WH",
        "role": "ANALYST",
        "schema": "PUBLIC",
    },
))

await db.connect()
results = await db.execute("SELECT * FROM my_table LIMIT 10")
```

### Snowflake Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `account` | `str` | Yes | Account identifier (e.g., `abc12345.us-east-1`) |
| `warehouse` | `str` | No | Virtual warehouse name |
| `role` | `str` | No | Role to use |
| `schema` | `str` | No | Schema (default: `PUBLIC`) |
| `authenticator` | `str` | No | Auth method (`externalbrowser`, `oauth`) |

### SSO Authentication

```python
db = SnowflakeAdapter(ConnectionConfig(
    name="snowflake",
    db_type=DatabaseType.SNOWFLAKE,
    database="MY_DB",
    username="user@company.com",
    options={
        "account": "abc12345.us-east-1",
        "warehouse": "COMPUTE_WH",
        "authenticator": "externalbrowser",
    },
))
```

See [Snowflake Documentation](../databases/snowflake.md) for more details.

---

## Adapter Interface

All adapters implement this interface:

```python
class DatabaseAdapter(ABC):
    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""
        pass

    @abstractmethod
    async def execute(
        self,
        query: str,
        params: tuple = None,
    ) -> List[Dict]:
        """Execute query and return results."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass
```

---

## Using with MetaAgent

```python
# Create adapter
db = MSSQLAdapter(config)
await db.connect()

# Create agent
agent = MetaAgent(llm_client=llm)

# Connect agent to database
stats = await agent.connect(db_executor=db.execute)

print(f"Dialect: {stats['dialect']}")  # Auto-detected!
print(f"Tables: {stats['tables']}")
```

---

## Auto-Detection

The agent automatically detects:

| Feature | MS SQL | PostgreSQL | MySQL | SQLite | Snowflake |
|---------|--------|------------|-------|--------|-----------|
| Row limit | `TOP` | `LIMIT` | `LIMIT` | `LIMIT` | `LIMIT` |
| Current date | `GETDATE()` | `NOW()` | `NOW()` | `datetime('now')` | `CURRENT_TIMESTAMP()` |
| Identifiers | `[brackets]` | `"quotes"` | `` `backticks` `` | `"quotes"` | `"QUOTES"` |
| Boolean | `1/0` | `true/false` | `1/0` | `1/0` | `true/false` |
| Date diff | `DATEDIFF()` | `AGE()` | `DATEDIFF()` | Custom | `DATEDIFF()` |
