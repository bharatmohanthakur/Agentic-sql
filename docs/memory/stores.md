# Memory Stores

Agentic SQL provides multiple storage backends for the memory system. Each store has different strengths and use cases.

## Overview

| Store | Type | Use Case | Installation |
|-------|------|----------|--------------|
| SQLite | File-based | Simple deployments, no server needed | `pip install agentic-sql[sqlite]` |
| ChromaDB | Vector | Semantic search, local/embedded | `pip install agentic-sql[vector]` |
| Qdrant | Vector | High-performance vector search | `pip install agentic-sql[qdrant]` |
| OpenSearch | Hybrid | Vector + full-text search | `pip install agentic-sql[opensearch]` |
| Neo4j | Graph | Entity relationships | `pip install agentic-sql[graph]` |

Install all memory backends:
```bash
pip install agentic-sql[memory-all]
```

---

## SQLite Store

Lightweight, file-based storage that requires no external server. Best for simple deployments and development.

### Installation

```bash
pip install agentic-sql[sqlite]
```

### Configuration Options

```python
from memory.stores import SQLiteMemoryStore
from memory.stores.sqlite_store import SQLiteConfig

config = SQLiteConfig(
    # Database file path
    db_path="./memories.db",  # Required: path to SQLite database file
)
```

### Complete Initialization Example

```python
import asyncio
from memory.stores import SQLiteMemoryStore
from memory.stores.sqlite_store import SQLiteConfig
from memory.manager import Memory, MemoryType, MemoryPriority

async def main():
    # 1. Configure
    config = SQLiteConfig(
        db_path="./data/memories.db"
    )

    # 2. Create store and connect
    store = SQLiteMemoryStore(config)
    await store.connect()
    print("Connected to SQLite")

    # 3. Store a memory
    memory = Memory(
        type=MemoryType.QUERY_PATTERN,
        content="SELECT COUNT(*) FROM users WHERE active = true",
        priority=MemoryPriority.HIGH,
        metadata={"question": "How many active users?"}
    )
    await store.store(memory)

    # 4. Retrieve with full-text search
    results = await store.retrieve(
        query="active users",
        memory_type=MemoryType.QUERY_PATTERN,
        limit=10,
        use_fts=True,  # Use FTS5 full-text search
    )
    print(f"Found {len(results)} results")

    # 5. Get by type
    patterns = await store.get_by_type(MemoryType.QUERY_PATTERN, limit=50)

    # 6. Get count
    total = await store.count()
    print(f"Total memories: {total}")

    # 7. Disconnect
    await store.disconnect()

asyncio.run(main())
```

### Features

- No external server required
- FTS5 full-text search support
- WAL mode for concurrent access
- Automatic index creation

---

## ChromaDB Store

Open-source embedding database with automatic embedding support. Good for local/embedded vector search.

### Installation

```bash
pip install agentic-sql[vector]
```

### Configuration Options

```python
from memory.stores import ChromaMemoryStore
from memory.stores.chroma_store import ChromaConfig

config = ChromaConfig(
    # Storage path (for persistent mode)
    path="./chroma_data",              # Local storage directory

    # Collection settings
    collection_name="agentic_sql_memories",  # Collection name

    # Distance function
    distance_fn="cosine",              # Options: "cosine", "l2", "ip"

    # Client-server mode (optional)
    host=None,                         # ChromaDB server host
    port=None,                         # ChromaDB server port
)
```

### Complete Initialization Example

```python
import asyncio
from memory.stores import ChromaMemoryStore
from memory.stores.chroma_store import ChromaConfig
from memory.manager import Memory, MemoryType, MemoryPriority

async def main():
    # 1. Configure - Local persistent mode
    config = ChromaConfig(
        path="./data/chroma",
        collection_name="sql_memories",
        distance_fn="cosine",
    )

    # OR Configure - Client-server mode
    # config = ChromaConfig(
    #     host="localhost",
    #     port=8000,
    #     collection_name="sql_memories",
    # )

    # 2. Create store and connect
    store = ChromaMemoryStore(config)
    await store.connect()
    print("Connected to ChromaDB")

    # 3. Store a memory (ChromaDB auto-generates embeddings if not provided)
    memory = Memory(
        type=MemoryType.SCHEMA,
        content="Users table has columns: id, name, email, created_at",
        metadata={"table": "users"},
        # embedding=[0.1, 0.2, ...],  # Optional: provide your own embedding
    )
    await store.store(memory)

    # 4. Semantic search
    results = await store.retrieve(
        query="user email columns",  # Text query (auto-embedded)
        # query_vector=[0.1, 0.2, ...],  # Or provide vector directly
        memory_type=MemoryType.SCHEMA,
        limit=10,
    )
    print(f"Found {len(results)} results")

    # 5. Get count
    total = await store.count()
    print(f"Total memories: {total}")

    # 6. Disconnect
    await store.disconnect()

asyncio.run(main())
```

### Features

- Simple setup, no server required
- Persistent local storage
- Automatic embedding (optional)
- Metadata filtering

---

## Qdrant Store

High-performance vector database optimized for similarity search with filtering.

### Installation

```bash
pip install agentic-sql[qdrant]
```

### Configuration Options

```python
from memory.stores import QdrantMemoryStore
from memory.stores.qdrant_store import QdrantConfig

config = QdrantConfig(
    # Connection settings
    host="localhost",                  # Qdrant server host
    port=6333,                         # Qdrant HTTP port
    grpc_port=6334,                    # Qdrant gRPC port
    api_key=None,                      # API key (for cloud/auth)
    https=False,                       # Use HTTPS
    url=None,                          # Full URL (alternative to host/port)
    timeout=30,                        # Request timeout in seconds

    # Collection settings
    collection_name="agentic_sql_memories",  # Collection name
    vector_size=1536,                  # Vector dimensions (OpenAI: 1536)
    distance="Cosine",                 # Options: "Cosine", "Euclid", "Dot"
    on_disk=False,                     # Store vectors on disk

    # Multi-tenancy
    namespace=None,                    # Logical namespace for isolation
    tenant_id=None,                    # Tenant identifier
    group_id=None,                     # Group/project identifier

    # Advanced settings
    shard_number=None,                 # Number of shards
    replication_factor=None,           # Replication factor
    write_consistency_factor=None,     # Write consistency
)
```

### Complete Initialization Example

```python
import asyncio
from memory.stores import QdrantMemoryStore
from memory.stores.qdrant_store import QdrantConfig
from memory.manager import Memory, MemoryType, MemoryPriority

async def main():
    # 1. Configure - Local server
    config = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="sql_memories",
        vector_size=1536,
        distance="Cosine",
        # Multi-tenancy
        namespace="production",
        tenant_id="customer_123",
        group_id="project_abc",
    )

    # OR Configure - Qdrant Cloud
    # config = QdrantConfig(
    #     url="https://your-cluster-id.qdrant.io",
    #     api_key="your-api-key",
    #     collection_name="sql_memories",
    #     vector_size=1536,
    # )

    # 2. Create store and connect
    store = QdrantMemoryStore(config)
    await store.connect()
    print("Connected to Qdrant")

    # 3. Store a memory (REQUIRES embedding)
    embedding = [0.1] * 1536  # Your actual embedding from OpenAI/etc
    memory = Memory(
        type=MemoryType.QUERY_PATTERN,
        content="SELECT * FROM orders WHERE date > '2024-01-01'",
        embedding=embedding,  # Required for Qdrant
        metadata={"question": "Recent orders"},
    )
    await store.store(memory)

    # 4. Vector search
    query_embedding = [0.1] * 1536  # Your query embedding
    results = await store.retrieve(
        query="",  # Not used when query_vector provided
        query_vector=query_embedding,
        memory_type=MemoryType.QUERY_PATTERN,
        score_threshold=0.7,  # Minimum similarity
        limit=10,
        # Override multi-tenancy per query (optional)
        # namespace="staging",
        # tenant_id="customer_456",
    )
    print(f"Found {len(results)} results")

    # 5. Get count
    total = await store.count()
    print(f"Total memories: {total}")

    # 6. Delete a memory
    await store.delete(memory.id)

    # 7. Disconnect
    await store.disconnect()

asyncio.run(main())
```

### Docker Setup

```bash
# Start Qdrant server
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### Features

- High-performance vector search
- Multi-tenancy (namespace, tenant_id, group_id)
- Payload filtering
- On-disk storage option
- HNSW indexing
- Cloud or self-hosted

---

## OpenSearch Store

Enterprise-grade search with both vector and full-text capabilities. Best for hybrid search scenarios.

### Installation

```bash
pip install agentic-sql[opensearch]
```

### Configuration Options

```python
from memory.stores import OpenSearchMemoryStore
from memory.stores.opensearch_store import OpenSearchConfig

config = OpenSearchConfig(
    # Connection settings
    hosts=["https://localhost:9200"],  # OpenSearch hosts
    username=None,                     # Username for auth
    password=None,                     # Password for auth
    use_ssl=True,                      # Use SSL/TLS
    verify_certs=False,                # Verify SSL certificates
    ssl_show_warn=False,               # Show SSL warnings
    timeout=30,                        # Request timeout

    # Index settings
    index_name="agentic_sql_memories", # Index name
    index_prefix=None,                 # Prefix (e.g., "prod_" -> "prod_memories")

    # Vector settings
    vector_dimension=1536,             # Vector dimensions
    ef_construction=512,               # HNSW parameter (recall vs speed)
    m=16,                              # HNSW parameter (recall vs memory)

    # Multi-tenancy
    namespace=None,                    # Logical namespace
    tenant_id=None,                    # Tenant identifier
    group_id=None,                     # Group/project identifier

    # Index management
    number_of_shards=1,                # Primary shards
    number_of_replicas=0,              # Replica shards
)
```

### Complete Initialization Example

```python
import asyncio
from memory.stores import OpenSearchMemoryStore
from memory.stores.opensearch_store import OpenSearchConfig
from memory.manager import Memory, MemoryType, MemoryPriority

async def main():
    # 1. Configure
    config = OpenSearchConfig(
        hosts=["https://localhost:9200"],
        username="admin",
        password="admin",
        use_ssl=True,
        verify_certs=False,
        index_name="sql_memories",
        index_prefix="prod_",  # Creates "prod_sql_memories"
        vector_dimension=1536,
        # Multi-tenancy
        namespace="production",
        tenant_id="customer_123",
        group_id="project_abc",
        # Cluster settings
        number_of_shards=3,
        number_of_replicas=1,
    )

    # 2. Create store and connect
    store = OpenSearchMemoryStore(config)
    await store.connect()
    print("Connected to OpenSearch")

    # 3. Store a memory
    embedding = [0.1] * 1536  # Your actual embedding
    memory = Memory(
        type=MemoryType.QUERY_PATTERN,
        content="SELECT name, email FROM users WHERE status = 'active'",
        embedding=embedding,  # Optional for OpenSearch
        metadata={"question": "Active user details"},
    )
    await store.store(memory)

    # 4. Hybrid search (vector + text)
    query_embedding = [0.1] * 1536
    results = await store.retrieve(
        query="active user email",     # BM25 text search
        query_vector=query_embedding,  # k-NN vector search
        use_hybrid=True,               # Combine both
        memory_type=MemoryType.QUERY_PATTERN,
        limit=10,
        # Override multi-tenancy per query (optional)
        # namespace="staging",
        # tenant_id="customer_456",
    )
    print(f"Found {len(results)} results")

    # 5. Pure text search (no vectors)
    text_results = await store.search_text(
        query="SELECT FROM users",
        memory_type=MemoryType.QUERY_PATTERN,
        limit=10,
    )

    # 6. Get count (with multi-tenancy filter)
    total = await store.count()
    tenant_count = await store.count(tenant_id="customer_123")
    print(f"Total: {total}, Tenant: {tenant_count}")

    # 7. Disconnect
    await store.disconnect()

asyncio.run(main())
```

### Docker Setup

```bash
# Start OpenSearch
docker run -p 9200:9200 -p 9600:9600 \
    -e "discovery.type=single-node" \
    -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=YourPassword123!" \
    opensearchproject/opensearch:latest
```

### Features

- k-NN vector similarity search
- BM25 full-text search
- Hybrid search combining both
- Multi-tenancy (namespace, tenant_id, group_id)
- Index prefixing for environments
- Rich filtering capabilities
- Scalable and distributed

---

## Neo4j Store

Graph database for relationship-based memory with traversal capabilities.

### Installation

```bash
pip install agentic-sql[graph]
```

### Configuration Options

```python
from memory.stores import Neo4jMemoryStore
from memory.stores.neo4j_store import Neo4jConfig

config = Neo4jConfig(
    # Connection settings
    uri="bolt://localhost:7687",       # Neo4j URI
    username="neo4j",                  # Username
    password="password",               # Password
    database="neo4j",                  # Database name

    # Multi-tenancy
    namespace=None,                    # Namespace (added as node label)
    tenant_id=None,                    # Tenant identifier (node property)
    group_id=None,                     # Group identifier (node property)

    # Node configuration
    node_label="Memory",               # Base label for nodes
    use_namespace_label=True,          # Add namespace as additional label

    # Connection pool
    max_connection_lifetime=3600,      # Max connection lifetime (seconds)
    max_connection_pool_size=100,      # Max pool size
    connection_timeout=30,             # Connection timeout (seconds)
)
```

### Complete Initialization Example

```python
import asyncio
from memory.stores import Neo4jMemoryStore
from memory.stores.neo4j_store import Neo4jConfig
from memory.manager import Memory, MemoryType, MemoryPriority

async def main():
    # 1. Configure
    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="your-password",
        database="neo4j",
        # Multi-tenancy
        namespace="production",     # Nodes labeled: Memory:NS_production
        tenant_id="customer_123",
        group_id="project_abc",
        # Connection pool
        max_connection_pool_size=50,
        connection_timeout=30,
    )

    # 2. Create store and connect
    store = Neo4jMemoryStore(config)
    await store.connect()
    print("Connected to Neo4j")

    # 3. Store a memory with relationships
    memory = Memory(
        type=MemoryType.SCHEMA,
        content="Orders table joins with Customers on customer_id",
        metadata={
            "relationships": [
                {"target_id": "customer-memory-uuid", "type": "joins_with"},
                {"target_id": "product-memory-uuid", "type": "contains"},
            ]
        },
    )
    await store.store(memory)

    # 4. Graph traversal retrieval
    results = await store.retrieve(
        query="Orders",           # Search by content or ID
        memory_type=MemoryType.SCHEMA,
        depth=2,                  # Traverse up to 2 hops
        limit=10,
        # Override multi-tenancy per query (optional)
        # namespace="staging",
        # tenant_id="customer_456",
    )
    print(f"Found {len(results)} results")

    # 5. Get count (with multi-tenancy filter)
    total = await store.count()
    tenant_count = await store.count(tenant_id="customer_123")
    print(f"Total: {total}, Tenant: {tenant_count}")

    # 6. Delete a memory
    await store.delete(memory.id)

    # 7. Disconnect
    await store.disconnect()

asyncio.run(main())
```

### Docker Setup

```bash
# Start Neo4j
docker run -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your-password \
    -v $(pwd)/neo4j_data:/data \
    neo4j:latest
```

### Features

- Entity relationships
- Multi-tenancy (namespace as labels, tenant/group as properties)
- Graph traversal for context expansion
- Cypher query support
- Automatic index creation
- Relationship types

---

## Using with MetaAgent

Pass memory stores to the MetaAgent for enhanced learning:

```python
import asyncio
from intelligence.meta_agent import MetaAgent
from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
from memory.manager import MemoryManager, MemoryConfig
from memory.stores import SQLiteMemoryStore, QdrantMemoryStore
from memory.stores.sqlite_store import SQLiteConfig
from memory.stores.qdrant_store import QdrantConfig

async def main():
    # 1. Setup LLM
    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key="your-key",
        azure_endpoint="https://your-endpoint.openai.azure.com",
        azure_deployment="gpt-4o",
    ))

    # 2. Setup stores
    sql_store = SQLiteMemoryStore(SQLiteConfig(db_path="./memories.db"))
    vector_store = QdrantMemoryStore(QdrantConfig(
        host="localhost",
        port=6333,
        namespace="production",
    ))

    await sql_store.connect()
    await vector_store.connect()

    # 3. Create MemoryManager
    memory = MemoryManager(
        config=MemoryConfig(enable_sql=True, enable_vector=True),
        sql_store=sql_store,
        vector_store=vector_store,
    )

    # 4. Create agent with memory_manager
    agent = MetaAgent(
        llm_client=llm,
        memory_manager=memory,  # <-- Pass here!
    )

    # 5. Agent now uses memory stores for learning
    # - Stores successful queries in both SQL and vector stores
    # - Uses semantic search to find similar past solutions

asyncio.run(main())
```

---

## Store Selection Guide

### Choose SQLite when:
- Simple deployment without external services
- Development and testing
- Small to medium memory collections
- Full-text search is sufficient

### Choose ChromaDB when:
- Need semantic search without server setup
- Embedded vector database requirement
- Getting started with vector search

### Choose Qdrant when:
- High-performance vector search required
- Large-scale deployments
- Need multi-tenancy support
- Production workloads

### Choose OpenSearch when:
- Need both vector and full-text search
- Enterprise requirements
- Distributed scaling needed
- Complex filtering and aggregations

### Choose Neo4j when:
- Entity relationships are important
- Need graph traversal
- Knowledge graph use cases
- Relationship-based reasoning
