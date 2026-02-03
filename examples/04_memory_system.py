#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 4: MEMORY SYSTEM - Persistent Knowledge Storage
=============================================================================

This tutorial shows how to use the Memory System for:
- Storing successful query patterns
- Remembering user preferences
- Building entity knowledge graphs
- Semantic search across past interactions

The memory system uses a hybrid architecture:
- Graph Store: Entity relationships
- Vector Store: Semantic embeddings
- SQL Store: Structured persistence

Run: python examples/04_memory_system.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


async def main():
    print("=" * 60)
    print("  TUTORIAL 4: MEMORY SYSTEM")
    print("=" * 60)

    # =========================================================================
    # INITIALIZE MEMORY MANAGER
    # =========================================================================
    print("\n[Step 1] Initializing Memory Manager...")

    from memory.manager import (
        MemoryManager,
        MemoryConfig,
        MemoryType,
        MemoryPriority,
        Memory,
    )

    # Configure memory storage paths
    config = MemoryConfig(
        # Where to store vector embeddings
        vector_store_path="./data/vector_store",

        # Where to store graph relationships
        graph_store_path="./data/graph_store",

        # SQLite database for structured data
        sql_store_path="./data/memory.db",

        # Enable different storage backends
        enable_vector_store=True,
        enable_graph_store=True,
        enable_sql_store=True,
    )

    memory = MemoryManager(config)
    print("  ✓ Memory Manager initialized")

    # =========================================================================
    # MEMORY TYPES
    # =========================================================================
    print("\n[Step 2] Understanding Memory Types...")

    print("""
    Available Memory Types:
    ┌──────────────────────┬─────────────────────────────────────┐
    │ Type                 │ Purpose                             │
    ├──────────────────────┼─────────────────────────────────────┤
    │ CONVERSATION         │ Chat history with user              │
    │ ENTITY_FACT          │ Facts about database entities       │
    │ SCHEMA               │ Database schema knowledge           │
    │ QUERY_PATTERN        │ Successful SQL patterns             │
    │ ERROR_PATTERN        │ Errors to avoid                     │
    │ USER_PREFERENCE      │ User-specific preferences           │
    │ SEMANTIC             │ General semantic knowledge          │
    └──────────────────────┴─────────────────────────────────────┘
    """)

    # =========================================================================
    # STORING MEMORIES
    # =========================================================================
    print("[Step 3] Storing Memories...")

    # Store a successful query pattern
    await memory.add(
        content="SELECT COUNT(*) FROM users WHERE active = true",
        memory_type=MemoryType.QUERY_PATTERN,
        metadata={
            "question": "How many active users?",
            "tables": ["users"],
            "success": True,
            "execution_time_ms": 45,
        },
        priority=MemoryPriority.HIGH,
    )
    print("  ✓ Stored query pattern: active users count")

    # Store schema knowledge
    await memory.add(
        content="The 'users' table has columns: id, name, email, active, created_at",
        memory_type=MemoryType.SCHEMA,
        metadata={
            "table": "users",
            "columns": ["id", "name", "email", "active", "created_at"],
        },
    )
    print("  ✓ Stored schema knowledge: users table")

    # Store an entity fact
    await memory.add(
        content="User 'alice@email.com' is a premium customer since 2023",
        memory_type=MemoryType.ENTITY_FACT,
        metadata={
            "entity_type": "user",
            "entity_id": "alice@email.com",
            "fact_type": "subscription",
        },
    )
    print("  ✓ Stored entity fact: Alice's subscription")

    # Store user preference
    await memory.add(
        content="User prefers results in JSON format with max 100 rows",
        memory_type=MemoryType.USER_PREFERENCE,
        metadata={
            "user_id": "user123",
            "preference_type": "output_format",
            "format": "json",
            "max_rows": 100,
        },
    )
    print("  ✓ Stored user preference: output format")

    # Store an error pattern to avoid
    await memory.add(
        content="Error: Cannot use LIMIT in MS SQL Server, use TOP instead",
        memory_type=MemoryType.ERROR_PATTERN,
        metadata={
            "error_type": "syntax",
            "dialect": "mssql",
            "wrong": "LIMIT",
            "correct": "TOP",
        },
    )
    print("  ✓ Stored error pattern: LIMIT vs TOP")

    # =========================================================================
    # SEARCHING MEMORIES
    # =========================================================================
    print("\n[Step 4] Searching Memories...")

    # Semantic search
    results = await memory.search(
        query="how to count users",
        memory_types=[MemoryType.QUERY_PATTERN],
        limit=5,
    )
    print(f"\n  Search: 'how to count users'")
    print(f"  Found {len(results)} results:")
    for r in results:
        print(f"    - {r.content[:50]}...")

    # Search with filters
    results = await memory.search(
        query="users table structure",
        memory_types=[MemoryType.SCHEMA],
        limit=5,
    )
    print(f"\n  Search: 'users table structure'")
    print(f"  Found {len(results)} results:")
    for r in results:
        print(f"    - {r.content[:50]}...")

    # =========================================================================
    # MEMORY HIERARCHY
    # =========================================================================
    print("\n[Step 5] Memory Hierarchy...")

    print("""
    Memory Levels (from most to least specific):
    ┌─────────────────────────────────────────────────────────┐
    │  SESSION   │ Current conversation only (expires)       │
    ├─────────────────────────────────────────────────────────┤
    │  USER      │ User-specific (preferences, history)      │
    ├─────────────────────────────────────────────────────────┤
    │  ENTITY    │ Facts about database entities             │
    ├─────────────────────────────────────────────────────────┤
    │  GLOBAL    │ Shared across all users                   │
    └─────────────────────────────────────────────────────────┘

    During retrieval, higher-priority memories override lower ones.
    """)

    # =========================================================================
    # ECL PIPELINE
    # =========================================================================
    print("[Step 6] ECL Pipeline (Extract, Cognify, Load)...")

    print("""
    When you add a memory, it goes through:

    ┌────────────┐     ┌────────────┐     ┌────────────┐
    │  EXTRACT   │────▶│  COGNIFY   │────▶│    LOAD    │
    │            │     │            │     │            │
    │ • Parse    │     │ • Generate │     │ • Store in │
    │   content  │     │   embeddings│    │   Graph    │
    │ • Extract  │     │ • Find     │     │ • Store in │
    │   metadata │     │   relations │    │   Vector   │
    │ • Validate │     │ • Enrich   │     │ • Store in │
    │            │     │   context  │     │   SQL      │
    └────────────┘     └────────────┘     └────────────┘
    """)

    # =========================================================================
    # USING WITH METAGENT
    # =========================================================================
    print("[Step 7] Integration with MetaAgent...")

    print("""
    The MetaAgent can use MemoryManager for advanced storage:
    - Storing successful SQL patterns (QUERY_PATTERN)
    - Remembering schema insights (SCHEMA)
    - Learning from errors (ERROR_PATTERN)
    - Semantic search for similar past solutions

    Example:
    ```python
    from intelligence.meta_agent import MetaAgent
    from memory.manager import MemoryManager, MemoryConfig
    from memory.stores import SQLiteMemoryStore, ChromaMemoryStore
    from memory.stores.sqlite_store import SQLiteConfig
    from memory.stores.chroma_store import ChromaConfig

    # 1. Setup stores
    sql_store = SQLiteMemoryStore(SQLiteConfig(db_path="./memories.db"))
    vector_store = ChromaMemoryStore(ChromaConfig(path="./chroma"))
    await sql_store.connect()
    await vector_store.connect()

    # 2. Create memory manager with stores
    memory = MemoryManager(
        config=MemoryConfig(enable_sql=True, enable_vector=True),
        sql_store=sql_store,
        vector_store=vector_store,
    )

    # 3. Create agent with memory_manager parameter
    agent = MetaAgent(
        llm_client=llm,
        memory_manager=memory,  # <-- Pass memory manager here!
    )

    # 4. Agent now stores learnings in memory stores
    # and uses semantic search in RESEARCH phase
    await agent.connect(db_executor=db.execute)
    result = await agent.query("Show top customers")
    ```
    """)

    # =========================================================================
    # CLEANUP
    # =========================================================================
    print("\n[Cleanup] Memory is persisted to disk automatically.")
    print("  Vector store: ./data/vector_store/")
    print("  Graph store: ./data/graph_store/")
    print("  SQL store: ./data/memory.db")

    print("\n" + "=" * 60)
    print("  TUTORIAL COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
