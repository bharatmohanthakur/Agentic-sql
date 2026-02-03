# Snowflake

Connect to Snowflake Data Cloud.

---

## Overview

Snowflake adapter provides full support for Snowflake Data Cloud, including:

- Standard username/password authentication
- Key pair authentication
- SSO/OAuth authentication
- Warehouse and role management
- Schema discovery
- Query execution with automatic result formatting

---

## Installation

```bash
pip install agentic-sql[snowflake]

# Or install the connector directly
pip install snowflake-connector-python
```

---

## Configuration

### Basic Connection

```python
from src.database.multi_db import SnowflakeAdapter, ConnectionConfig, DatabaseType

db = SnowflakeAdapter(ConnectionConfig(
    name="snowflake_prod",
    db_type=DatabaseType.SNOWFLAKE,
    database="MY_DATABASE",
    username="my_user",
    password="my_password",
    options={
        "account": "abc12345.us-east-1",  # Account identifier
        "warehouse": "COMPUTE_WH",
        "role": "ANALYST_ROLE",
        "schema": "PUBLIC",
    },
))

await db.connect()
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `database` | `str` | Yes | Database name |
| `username` | `str` | Yes | Snowflake username |
| `password` | `str` | Yes* | Password (*or use key pair) |
| `options.account` | `str` | Yes | Account identifier (e.g., `abc12345.us-east-1`) |
| `options.warehouse` | `str` | No | Virtual warehouse name |
| `options.role` | `str` | No | Role to use |
| `options.schema` | `str` | No | Schema (default: `PUBLIC`) |
| `options.authenticator` | `str` | No | Auth method (`externalbrowser`, `oauth`, etc.) |

---

## Authentication Methods

### Username/Password

```python
db = SnowflakeAdapter(ConnectionConfig(
    name="snowflake",
    db_type=DatabaseType.SNOWFLAKE,
    database="MY_DB",
    username="user",
    password="password",
    options={
        "account": "myaccount.us-east-1",
        "warehouse": "COMPUTE_WH",
    },
))
```

### SSO (Browser-based)

```python
db = SnowflakeAdapter(ConnectionConfig(
    name="snowflake",
    db_type=DatabaseType.SNOWFLAKE,
    database="MY_DB",
    username="user@company.com",
    options={
        "account": "myaccount.us-east-1",
        "warehouse": "COMPUTE_WH",
        "authenticator": "externalbrowser",
    },
))
```

### OAuth

```python
db = SnowflakeAdapter(ConnectionConfig(
    name="snowflake",
    db_type=DatabaseType.SNOWFLAKE,
    database="MY_DB",
    username="user",
    options={
        "account": "myaccount.us-east-1",
        "warehouse": "COMPUTE_WH",
        "authenticator": "oauth",
        "token": "your_oauth_token",
    },
))
```

---

## Account Identifier Format

Snowflake account identifiers vary by region:

| Region | Format | Example |
|--------|--------|---------|
| AWS US West | `account` | `abc12345` |
| AWS US East | `account.us-east-1` | `abc12345.us-east-1` |
| AWS EU | `account.eu-west-1` | `abc12345.eu-west-1` |
| Azure | `account.region.azure` | `abc12345.east-us-2.azure` |
| GCP | `account.region.gcp` | `abc12345.us-central1.gcp` |

---

## Usage Examples

### Basic Query

```python
# Connect
await db.connect()

# Execute query
results = await db.execute("SELECT * FROM customers LIMIT 10")
for row in results:
    print(row)
```

### Schema Discovery

```python
# Get database schema
schema = await db.get_schema()

print(f"Tables: {len(schema['tables'])}")
for table in schema['tables']:
    print(f"  - {table['table_name']}")
```

### Query Execution Plan

```python
# Get execution plan
plan = await db.explain("SELECT * FROM orders WHERE status = 'PENDING'")
print(plan)
```

---

## Using with MetaAgent

```python
from src.database.multi_db import SnowflakeAdapter, ConnectionConfig, DatabaseType
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
from src.intelligence.meta_agent import MetaAgent
import os

# Setup Snowflake
db = SnowflakeAdapter(ConnectionConfig(
    name="snowflake",
    db_type=DatabaseType.SNOWFLAKE,
    database="ANALYTICS",
    username=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    options={
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "warehouse": "ANALYTICS_WH",
        "role": "ANALYST",
    },
))

await db.connect()

# Setup LLM
llm = AzureOpenAIClient(AzureOpenAIConfig(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-4o",
))

# Create agent
agent = MetaAgent(llm_client=llm)

# Connect agent to Snowflake
stats = await agent.connect(db_executor=db.execute)

print(f"Dialect: {stats['dialect']}")  # Auto-detected as Snowflake!
print(f"Tables: {stats['tables']}")

# Query naturally
result = await agent.query("What are the top 10 customers by revenue?")
if result["success"]:
    print(result["data"])
```

---

## Snowflake-Specific SQL

The agent auto-detects Snowflake and uses appropriate syntax:

| Feature | Snowflake Syntax |
|---------|------------------|
| Row limit | `LIMIT n` |
| Current timestamp | `CURRENT_TIMESTAMP()` |
| Date functions | `DATEADD()`, `DATEDIFF()` |
| String concat | `CONCAT()` or `||` |
| Case sensitivity | Identifiers are uppercase by default |
| Sampling | `SAMPLE(n)` |

### Example Queries

```sql
-- Top N rows
SELECT * FROM orders LIMIT 10;

-- Date arithmetic
SELECT * FROM orders
WHERE created_at > DATEADD(day, -30, CURRENT_TIMESTAMP());

-- Sampling
SELECT * FROM large_table SAMPLE (1000 ROWS);

-- Semi-structured data (JSON)
SELECT data:customer.name::STRING as name
FROM events;
```

---

## Environment Variables

```bash
# Snowflake credentials
SNOWFLAKE_ACCOUNT=abc12345.us-east-1
SNOWFLAKE_USER=my_user
SNOWFLAKE_PASSWORD=my_password
SNOWFLAKE_DATABASE=MY_DATABASE
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_ROLE=ANALYST
SNOWFLAKE_SCHEMA=PUBLIC
```

---

## Troubleshooting

### Connection Issues

```python
# Test connection
try:
    await db.connect()
    result = await db.execute("SELECT CURRENT_VERSION()")
    print(f"Connected! Version: {result[0]}")
except Exception as e:
    print(f"Connection failed: {e}")
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Account not found` | Wrong account ID | Check account identifier format |
| `Warehouse not running` | Warehouse suspended | Resume warehouse or set auto-resume |
| `Role not granted` | Missing role permission | Grant role to user |
| `Object does not exist` | Wrong database/schema | Check current database and schema |

---

## Best Practices

1. **Use appropriate warehouse size** - Match warehouse to query complexity
2. **Set query timeout** - Prevent runaway queries
3. **Use roles** - Implement least-privilege access
4. **Cache results** - Use result caching for repeated queries
5. **Monitor costs** - Track warehouse credits usage
