# SQLite

Connect to SQLite databases.

---

## Installation

```bash
pip install -e ".[sqlite]"
```

---

## File Database

```python
from src.database.multi_db import SQLiteAdapter, ConnectionConfig, DatabaseType

db = SQLiteAdapter(ConnectionConfig(
    name="sqlite",
    db_type=DatabaseType.SQLITE,
    database="/path/to/database.db",
))

await db.connect()
```

---

## In-Memory Database

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
        name TEXT
    )
""")
```

---

## Auto-Detected Features

| Feature | SQLite Syntax |
|---------|---------------|
| Row limit | `LIMIT 10` |
| Current date | `datetime('now')` |
| Auto increment | `INTEGER PRIMARY KEY` |
