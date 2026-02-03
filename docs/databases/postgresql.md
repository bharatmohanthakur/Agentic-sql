# PostgreSQL

Connect to PostgreSQL.

---

## Installation

```bash
pip install -e ".[postgres]"
```

---

## Connection

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
```

---

## Connection String

```python
db = PostgreSQLAdapter(ConnectionConfig(
    name="postgres",
    db_type=DatabaseType.POSTGRESQL,
    connection_string="postgresql://user:pass@localhost:5432/mydb",
))
```

---

## Auto-Detected Features

| Feature | PostgreSQL Syntax |
|---------|-------------------|
| Row limit | `LIMIT 10` |
| Current date | `NOW()` |
| Date diff | `date1 - date2` |
| Identifiers | `"TableName"` |
| Boolean | `true/false` |
