# MS SQL Server

Connect to Microsoft SQL Server.

---

## Installation

```bash
pip install pyodbc
```

Also install ODBC Driver 17 or 18 for SQL Server.

---

## Connection

```python
from src.database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

db = MSSQLAdapter(ConnectionConfig(
    name="mssql",
    db_type=DatabaseType.MSSQL,
    host="server.database.windows.net,1433",
    database="MyDatabase",
    username="user",
    password="password",
))

await db.connect()
```

---

## Azure SQL Database

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

---

## Auto-Detected Features

| Feature | MS SQL Syntax |
|---------|---------------|
| Row limit | `SELECT TOP 10 *` |
| Current date | `GETDATE()` |
| Date diff | `DATEDIFF(day, a, b)` |
| Identifiers | `[TableName]` |
| String concat | `'a' + 'b'` |
