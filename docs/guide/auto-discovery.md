# Auto-Discovery

How the agent automatically discovers your database.

---

## Overview

When you call `agent.connect()`, the agent:

1. **Probes database** to detect SQL dialect
2. **Discovers schema** (tables, columns, types)
3. **Analyzes relationships** between tables
4. **Learns naming conventions**
5. **Generates insights** about the database

```python
stats = await agent.connect(db_executor=db.execute)
# {
#     "dialect": "mssql",
#     "tables": 25,
#     "schema_insights": 19,
# }
```

---

## Dialect Detection

The agent runs probe queries to detect the SQL dialect:

```python
probes = [
    ("SELECT TOP 1 1 AS test", "TOP syntax"),      # MSSQL
    ("SELECT 1 AS test LIMIT 1", "LIMIT syntax"),  # PostgreSQL/MySQL
    ("SELECT GETDATE()", "GETDATE function"),      # MSSQL
    ("SELECT NOW()", "NOW function"),              # PostgreSQL/MySQL
]
```

Based on which queries succeed/fail, the LLM determines the dialect:

| Dialect | TOP | LIMIT | GETDATE | NOW |
|---------|-----|-------|---------|-----|
| MS SQL | :material-check: | :material-close: | :material-check: | :material-close: |
| PostgreSQL | :material-close: | :material-check: | :material-close: | :material-check: |
| MySQL | :material-close: | :material-check: | :material-close: | :material-check: |
| SQLite | :material-close: | :material-check: | :material-close: | :material-close: |

---

## Schema Discovery

The agent queries `INFORMATION_SCHEMA` to discover:

- Tables
- Columns and data types
- Primary keys
- Foreign key relationships

```sql
-- Tables
SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'

-- Columns
SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'users'
```

---

## Schema Analysis

The LLM analyzes the schema and generates insights:

```python
# Example insights generated:
insights = [
    "- Users table has email, name, created_at columns",
    "- Orders joins to Customers via customer_id",
    "- Products table uses decimal for price",
    "- Naming convention: PascalCase for tables",
]
```

These insights are stored and used in future queries.

---

## What Gets Learned

| Category | Example |
|----------|---------|
| Tables | `users`, `orders`, `products` |
| Columns | `id`, `name`, `created_at` |
| Types | `varchar`, `int`, `datetime` |
| Relationships | `orders.customer_id → customers.id` |
| Naming | PascalCase, snake_case |
| Dialect | TOP vs LIMIT, date functions |

---

## Manual Override

You can provide hints:

```python
stats = await agent.connect(
    db_executor=db.execute,
    driver="mssql",  # Hint for dialect
)
```
