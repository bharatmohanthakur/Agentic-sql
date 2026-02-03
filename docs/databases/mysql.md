# MySQL

Connect to MySQL / MariaDB.

---

## Installation

```bash
pip install -e ".[mysql]"
```

---

## Connection

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
```

---

## Auto-Detected Features

| Feature | MySQL Syntax |
|---------|--------------|
| Row limit | `LIMIT 10` |
| Current date | `NOW()` |
| Identifiers | `` `TableName` `` |
| Auto increment | `AUTO_INCREMENT` |
