#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 4b: MEMORY STORES - Using Different Storage Backends
=============================================================================

This tutorial shows how to configure different memory storage backends:
- SQLite (lightweight, no server)
- ChromaDB (local vector search)
- Qdrant (high-performance vector search)
- OpenSearch (hybrid vector + full-text)
- Neo4j (graph relationships)

Run: python examples/04_memory_stores.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


async def example_sqlite():
    """Example: Using SQLite for lightweight file-based storage"""
    print("\n" + "=" * 60)
    print("  SQLITE MEMORY STORE - No Server Required")
    print("=" * 60)

    from memory.stores import SQLiteMemoryStore
    from memory.stores.sqlite_store import SQLiteConfig
    from memory.manager import Memory, MemoryType, MemoryPriority

    # Configure SQLite store
    config = SQLiteConfig(
        db_path="./data/memories.db"  # File path for database
    )

    store = SQLiteMemoryStore(config)
    await store.connect()
    print("  ✓ Connected to SQLite")

    # Create and store a memory
    memory = Memory(
        type=MemoryType.QUERY_PATTERN,
        content="SELECT COUNT(*) FROM users WHERE active = true",
        priority=MemoryPriority.HIGH,
        metadata={
            "question": "How many active users?",
            "success": True,
        }
    )

    await store.store(memory)
    print("  ✓ Stored memory")

    # Retrieve using full-text search
    results = await store.retrieve(
        query="active users",
        limit=5,
    )
    print(f"  ✓ Retrieved {len(results)} results")

    count = await store.count()
    print(f"  ✓ Total memories: {count}")

    await store.disconnect()
    print("  ✓ Disconnected")


async def example_chromadb():
    """Example: Using ChromaDB for local vector search"""
    print("\n" + "=" * 60)
    print("  CHROMADB MEMORY STORE - Local Vector Search")
    print("=" * 60)

    try:
        from memory.stores import ChromaMemoryStore
        from memory.stores.chroma_store import ChromaConfig
        from memory.manager import Memory, MemoryType, MemoryPriority

        config = ChromaConfig(
            path="./data/chroma",  # Local storage path
            collection_name="sql_memories",
            distance_fn="cosine",  # cosine, l2, ip
        )

        store = ChromaMemoryStore(config)
        await store.connect()
        print("  ✓ Connected to ChromaDB")

        # Store memory (ChromaDB auto-generates embeddings)
        memory = Memory(
            type=MemoryType.SCHEMA,
            content="Users table has columns: id, name, email, created_at",
            metadata={"table": "users"}
        )

        await store.store(memory)
        print("  ✓ Stored memory")

        # Semantic search
        results = await store.retrieve(
            query="user email columns",
            limit=5,
        )
        print(f"  ✓ Retrieved {len(results)} results")

        await store.disconnect()

    except ImportError:
        print("  ⚠ ChromaDB not installed. Run: pip install chromadb")


async def example_qdrant():
    """Example: Using Qdrant for high-performance vector search"""
    print("\n" + "=" * 60)
    print("  QDRANT MEMORY STORE - High-Performance Vector Search")
    print("=" * 60)

    try:
        from memory.stores import QdrantMemoryStore
        from memory.stores.qdrant_store import QdrantConfig
        from memory.manager import Memory, MemoryType, MemoryPriority

        config = QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="sql_memories",
            vector_size=1536,  # Must match your embedding dimension
            distance="Cosine",  # Cosine, Euclid, Dot
        )

        store = QdrantMemoryStore(config)

        print("  ℹ Qdrant requires a running server.")
        print("  Start with: docker run -p 6333:6333 qdrant/qdrant")

        # Uncomment when Qdrant server is running:
        # await store.connect()
        # print("  ✓ Connected to Qdrant")
        #
        # # Qdrant REQUIRES embeddings
        # memory = Memory(
        #     type=MemoryType.QUERY_PATTERN,
        #     content="SELECT * FROM orders WHERE date > '2024-01-01'",
        #     embedding=[0.1] * 1536,  # Your actual embedding
        # )
        #
        # await store.store(memory)
        # results = await store.retrieve(
        #     query="",
        #     query_vector=[0.1] * 1536,  # Search vector
        #     score_threshold=0.7,
        # )
        # print(f"  ✓ Retrieved {len(results)} results")

    except ImportError:
        print("  ⚠ Qdrant not installed. Run: pip install qdrant-client")


async def example_opensearch():
    """Example: Using OpenSearch for hybrid vector + text search"""
    print("\n" + "=" * 60)
    print("  OPENSEARCH MEMORY STORE - Hybrid Search")
    print("=" * 60)

    try:
        from memory.stores import OpenSearchMemoryStore
        from memory.stores.opensearch_store import OpenSearchConfig
        from memory.manager import Memory, MemoryType

        config = OpenSearchConfig(
            hosts=["https://localhost:9200"],
            index_name="sql_memories",
            username="admin",
            password="admin",
            use_ssl=True,
            verify_certs=False,
            vector_dimension=1536,
        )

        store = OpenSearchMemoryStore(config)

        print("  ℹ OpenSearch requires a running cluster.")
        print("  Start with Docker or use AWS OpenSearch Service")

        # Uncomment when OpenSearch is running:
        # await store.connect()
        # print("  ✓ Connected to OpenSearch")
        #
        # # Hybrid search: combine vector + text
        # results = await store.retrieve(
        #     query="order date filtering",  # BM25 text search
        #     query_vector=[0.1] * 1536,     # k-NN vector search
        #     use_hybrid=True,               # Combine both
        #     limit=10,
        # )
        #
        # # Pure text search (no vectors)
        # text_results = await store.search_text(
        #     query="SELECT FROM orders",
        #     limit=10,
        # )

    except ImportError:
        print("  ⚠ OpenSearch not installed. Run: pip install opensearch-py")


async def example_neo4j():
    """Example: Using Neo4j for graph relationships"""
    print("\n" + "=" * 60)
    print("  NEO4J MEMORY STORE - Graph Relationships")
    print("=" * 60)

    try:
        from memory.stores import Neo4jMemoryStore
        from memory.stores.neo4j_store import Neo4jConfig
        from memory.manager import Memory, MemoryType

        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
            database="neo4j",
        )

        store = Neo4jMemoryStore(config)

        print("  ℹ Neo4j requires a running server.")
        print("  Start with: docker run -p 7687:7687 neo4j")

        # Uncomment when Neo4j is running:
        # await store.connect()
        # print("  ✓ Connected to Neo4j")
        #
        # # Store with relationships
        # memory = Memory(
        #     type=MemoryType.SCHEMA,
        #     content="Orders table joins with Customers on customer_id",
        #     metadata={
        #         "relationships": [
        #             {"target_id": "customer-mem-id", "type": "joins"},
        #         ]
        #     }
        # )
        #
        # await store.store(memory)
        #
        # # Graph traversal
        # results = await store.retrieve(
        #     query="starting-memory-id",
        #     depth=2,  # Traverse 2 hops
        # )

    except ImportError:
        print("  ⚠ Neo4j not installed. Run: pip install neo4j")


async def example_hybrid_memory_manager():
    """Example: Using MemoryManager with multiple stores"""
    print("\n" + "=" * 60)
    print("  HYBRID MEMORY MANAGER - Multiple Stores")
    print("=" * 60)

    from memory.manager import MemoryManager, MemoryConfig, MemoryType
    from memory.stores import SQLiteMemoryStore
    from memory.stores.sqlite_store import SQLiteConfig

    # Create stores
    sql_store = SQLiteMemoryStore(SQLiteConfig(db_path="./data/hybrid.db"))
    await sql_store.connect()

    # Optional: Add vector store for semantic search
    # chroma_store = ChromaMemoryStore(ChromaConfig(path="./data/hybrid_chroma"))
    # await chroma_store.connect()

    # Optional: Add graph store for relationships
    # neo4j_store = Neo4jMemoryStore(Neo4jConfig())
    # await neo4j_store.connect()

    # Create memory manager with stores
    memory_manager = MemoryManager(
        config=MemoryConfig(
            enable_sql=True,
            enable_vector=False,  # Set True if using vector store
            enable_graph=False,   # Set True if using graph store
        ),
        sql_store=sql_store,
        # vector_store=chroma_store,
        # graph_store=neo4j_store,
    )

    print("  ✓ Created hybrid MemoryManager")

    # Use ECL pipeline
    memory = await memory_manager.ingest(
        content="SELECT name, email FROM users WHERE status = 'active'",
        memory_type=MemoryType.QUERY_PATTERN,
        metadata={
            "question": "Get active user details",
            "success": True,
        }
    )
    print(f"  ✓ Ingested memory: {memory.id}")

    # Search
    results = await memory_manager.search(
        query="active user",
        memory_type=MemoryType.QUERY_PATTERN,
        limit=5,
    )
    print(f"  ✓ Found {len(results)} results")


async def main():
    print("=" * 60)
    print("  TUTORIAL: MEMORY STORAGE BACKENDS")
    print("=" * 60)

    print("""
    Available Storage Backends:

    ┌────────────────┬──────────────────────────────────────────────┐
    │ Store          │ Best For                                     │
    ├────────────────┼──────────────────────────────────────────────┤
    │ SQLite         │ Simple deployments, no server needed         │
    │ ChromaDB       │ Local semantic search, embedded use          │
    │ Qdrant         │ High-performance production vector search    │
    │ OpenSearch     │ Hybrid search, enterprise scale              │
    │ Neo4j          │ Entity relationships, knowledge graphs       │
    └────────────────┴──────────────────────────────────────────────┘

    Installation:
    pip install agentic-sql[sqlite]      # SQLite
    pip install agentic-sql[vector]      # ChromaDB
    pip install agentic-sql[qdrant]      # Qdrant
    pip install agentic-sql[opensearch]  # OpenSearch
    pip install agentic-sql[graph]       # Neo4j
    pip install agentic-sql[memory-all]  # All backends
    """)

    # Run examples
    await example_sqlite()
    await example_chromadb()
    await example_qdrant()
    await example_opensearch()
    await example_neo4j()
    await example_hybrid_memory_manager()

    print("\n" + "=" * 60)
    print("  TUTORIAL COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
