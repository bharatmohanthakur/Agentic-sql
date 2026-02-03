---
layout: default
title: Memory System - Agentic SQL
---

# Memory System

How the agent remembers, learns, and improves over time.

---

## Overview

Agentic SQL uses a **hybrid memory architecture** that combines:

- **Graph Store** - Entity relationships
- **Vector Store** - Semantic embeddings
- **SQL Store** - Structured persistence

```
┌─────────────────────────────────────────────────────────────┐
│                      MEMORY MANAGER                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │   GRAPH     │  │   VECTOR    │  │    SQL      │        │
│   │   STORE     │  │   STORE     │  │   STORE     │        │
│   │             │  │             │  │             │        │
│   │ Entities &  │  │ Semantic    │  │ Structured  │        │
│   │ Relations   │  │ Embeddings  │  │ Persistence │        │
│   └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│              ECL PIPELINE (Extract, Cognify, Load)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Memory Types

| Type | Purpose | Example |
|------|---------|---------|
| `CONVERSATION` | Chat history | Previous questions and answers |
| `ENTITY_FACT` | Facts about entities | "User alice@email.com is premium" |
| `SCHEMA` | Database schema | "users table has email column" |
| `QUERY_PATTERN` | Successful SQL | "SELECT COUNT(*) worked for counting" |
| `ERROR_PATTERN` | Errors to avoid | "LIMIT doesn't work in MSSQL" |
| `USER_PREFERENCE` | User settings | "User prefers JSON output" |
| `SEMANTIC` | General knowledge | Domain-specific insights |

---

## MetaKnowledge

The agent's brain stores learned knowledge in `MetaKnowledge`:

```python
@dataclass
class MetaKnowledge:
    # Successful patterns
    successful_solutions: List[LearnedSolution]

    # Failure analysis
    failed_attempts: List[LearnedFailure]

    # Dialect-specific learnings
    dialect_learnings: List[str]

    # Schema insights
    schema_insights: List[str]

    # Fix strategies that worked
    fix_strategies: List[Dict]

    # TRUE LEARNING - Applied corrections
    name_corrections: Dict[str, str]      # {"categorie": "categories"}
    table_relationships: Dict[str, str]   # {"products": "product_items"}
    column_mappings: Dict[str, str]       # {"revenue": "total_amount"}
```

---

## How Memory Flows Through a Query

```
User: "How many orders per customer?"
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. THINK - Analyze question                                  │
│    └─► Check: dialect_learnings, schema_insights             │
│        "COUNT works, TOP not LIMIT for this dialect"         │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. RESEARCH - Find similar problems                          │
│    └─► Search: successful_solutions, failed_attempts         │
│        "Found: 'orders by user' → SELECT user_id, COUNT(*)"  │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. DESIGN - Create custom prompt                             │
│    └─► Apply: name_corrections, table_relationships          │
│        "Don't use 'customer', use 'customer_id'"             │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. EXECUTE - Run query, handle errors                        │
│    └─► Use: fix_strategies from past corrections             │
│        "If conversion error, apply TRY_CAST strategy"        │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. LEARN - Store outcome                                     │
│    ├─► Success: Add to successful_solutions                  │
│    └─► Failure: Add to failed_attempts, extract corrections  │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
      ~/.vanna/meta_agent.json  (Persisted)
```

---

## ECL Pipeline

The **Extract-Cognify-Load** pipeline processes all memories:

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│  EXTRACT   │────▶│  COGNIFY   │────▶│    LOAD    │
│            │     │            │     │            │
│ • Parse    │     │ • Generate │     │ • Graph    │
│   content  │     │   embeddings│    │   Store    │
│ • Extract  │     │ • Find     │     │ • Vector   │
│   metadata │     │   relations │    │   Store    │
│ • Validate │     │ • Enrich   │     │ • SQL      │
│            │     │   context  │     │   Store    │
└────────────┘     └────────────┘     └────────────┘
```

### Extract

```python
memory = await memory.add(
    content="SELECT COUNT(*) FROM users WHERE active = true",
    memory_type=MemoryType.QUERY_PATTERN,
    metadata={"table": "users", "success": True},
)
```

### Cognify

```python
memory = await memory.cognify(memory)
# Generates embeddings
# Extracts entities (users, active)
# Finds relationships
```

### Load

```python
await memory.store(memory)
# Stores in graph (relationships)
# Stores in vector (embeddings)
# Stores in SQL (persistence)
```

### Full Pipeline

```python
memory = await memory.ingest(
    content="...",
    memory_type=MemoryType.SCHEMA,
)
# Runs all three phases automatically
```

---

## Memory Hierarchy

Memories have different scopes and priorities:

```
┌─────────────────────────────────────────────────────────────┐
│                     MEMORY LEVELS                            │
├─────────────────────────────────────────────────────────────┤
│  SESSION   │ Current conversation only (expires)            │
├─────────────────────────────────────────────────────────────┤
│  USER      │ User-specific (preferences, history)           │
├─────────────────────────────────────────────────────────────┤
│  ENTITY    │ Facts about database entities                  │
├─────────────────────────────────────────────────────────────┤
│  GLOBAL    │ Shared across all users                        │
└─────────────────────────────────────────────────────────────┘
```

Higher-priority memories override lower ones during retrieval.

---

## Searching Memories

### Semantic Search

```python
results = await memory.search(
    query="how to count users",
    memory_types=[MemoryType.QUERY_PATTERN],
    limit=5,
)

for r in results:
    print(f"Content: {r.content}")
    print(f"Score: {r.relevance_score}")
```

### Filtered Search

```python
results = await memory.search(
    query="user table structure",
    memory_types=[MemoryType.SCHEMA],
    entity_id="users",  # Filter by entity
    session_id="sess123",  # Filter by session
    limit=10,
)
```

### Graph Traversal

```python
related = await memory.get_related(
    memory_id=some_memory_id,
    depth=2,  # Traverse 2 levels
)
```

---

## Knowledge Persistence

All knowledge is automatically saved to `~/.vanna/meta_agent.json`:

```json
{
  "successful_solutions": [
    {
      "question": "How many users?",
      "sql": "SELECT COUNT(*) FROM users",
      "problem_type": "counting",
      "was_corrected": false
    }
  ],
  "failed_attempts": [
    {
      "question": "Show categories",
      "sql": "SELECT * FROM Categories",
      "error": "Invalid object name 'Categories'",
      "llm_analysis": "Table is named 'Category' not 'Categories'"
    }
  ],
  "dialect_learnings": [
    "DIALECT: mssql - Uses TOP instead of LIMIT"
  ],
  "schema_insights": [
    "- Users table has email, name, created_at columns",
    "- Orders joins to Customers via customer_id"
  ],
  "name_corrections": {
    "categories": "Category",
    "users": "User"
  },
  "table_relationships": {
    "orders": "joins with customers on customer_id"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Self-Healing with Memory

When errors occur, the agent:

1. **Extracts the wrong name** from error message
2. **Searches database** for correct name
3. **Stores correction** for future queries
4. **Retries** with fixed SQL

```python
# Error: "Invalid object name 'Categories'"

# Agent learns:
name_corrections["categories"] = "Category"

# Next time "Show categories" is asked:
# Agent applies correction BEFORE generating SQL
# → "SELECT * FROM Category" (correct!)
```

---

## Using MemoryManager

### Initialize

```python
from src.memory.manager import MemoryManager, MemoryConfig

memory = MemoryManager(MemoryConfig(
    enable_graph=True,
    enable_vector=True,
    enable_sql=True,
    vector_dimensions=1536,
    similarity_threshold=0.7,
))
```

### Add Memories

```python
# Store a query pattern
await memory.add(
    content="SELECT COUNT(*) FROM users WHERE active = true",
    memory_type=MemoryType.QUERY_PATTERN,
    metadata={"question": "How many active users?"},
    priority=MemoryPriority.HIGH,
)

# Store schema knowledge
await memory.add(
    content="users table: id, name, email, active, created_at",
    memory_type=MemoryType.SCHEMA,
    metadata={"table": "users"},
)

# Store error pattern
await memory.add(
    content="LIMIT doesn't work in MSSQL, use TOP instead",
    memory_type=MemoryType.ERROR_PATTERN,
    metadata={"dialect": "mssql"},
)
```

### Search Memories

```python
# Semantic search
results = await memory.search(
    query="count active users",
    memory_types=[MemoryType.QUERY_PATTERN],
    limit=5,
)

# With graph context
results = await memory.search(
    query="user table",
    include_graph_context=True,  # Expand via relationships
)
```

---

## Memory Consolidation

Periodically, memories are consolidated:

```python
consolidated = await memory.consolidate()
# Merges duplicates
# Prunes old, low-relevance memories
# Recalculates relevance scores
```

---

<p align="center">
  <a href="multi-agent">Multi-Agent Workflows →</a>
</p>
