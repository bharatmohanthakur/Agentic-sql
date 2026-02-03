# Memory Types

Different types of memories for different purposes.

---

## Available Types

```python
from src.memory.manager import MemoryType

MemoryType.CONVERSATION     # Chat history
MemoryType.ENTITY_FACT      # Facts about entities
MemoryType.SCHEMA           # Database schema
MemoryType.QUERY_PATTERN    # Successful SQL patterns
MemoryType.ERROR_PATTERN    # Error patterns to avoid
MemoryType.USER_PREFERENCE  # User preferences
MemoryType.SEMANTIC         # General knowledge
```

---

## Usage Examples

### Query Pattern

```python
await memory.add(
    content="SELECT COUNT(*) FROM users WHERE active = true",
    memory_type=MemoryType.QUERY_PATTERN,
    metadata={"question": "active users", "success": True},
)
```

### Schema

```python
await memory.add(
    content="users: id, name, email, created_at",
    memory_type=MemoryType.SCHEMA,
    metadata={"table": "users"},
)
```

### Error Pattern

```python
await memory.add(
    content="LIMIT doesn't work in MSSQL, use TOP",
    memory_type=MemoryType.ERROR_PATTERN,
    metadata={"dialect": "mssql"},
)
```
