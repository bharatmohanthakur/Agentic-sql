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

### Usage

```python
from memory.stores import SQLiteMemoryStore
from memory.stores.sqlite_store import SQLiteConfig

config = SQLiteConfig(
    db_path="./memories.db"
)

store = SQLiteMemoryStore(config)
await store.connect()

# Store a memory
await store.store(memory)

# Retrieve with full-text search
results = await store.retrieve("user preferences", limit=10)

# Get by type
facts = await store.get_by_type(MemoryType.FACT, limit=50)
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

### Usage

```python
from memory.stores import ChromaMemoryStore
from memory.stores.chroma_store import ChromaConfig

config = ChromaConfig(
    path="./chroma_data",
    collection_name="agentic_sql_memories",
    distance_fn="cosine",  # cosine, l2, ip
)

store = ChromaMemoryStore(config)
await store.connect()

# Store with embedding
memory.embedding = [0.1, 0.2, ...]  # 1536-dim vector
await store.store(memory)

# Semantic search
results = await store.retrieve(
    query="database performance",
    query_vector=embedding,  # Optional pre-computed vector
    limit=10,
)
```

### Client-Server Mode

```python
config = ChromaConfig(
    host="localhost",
    port=8000,
    collection_name="memories",
)
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

### Usage

```python
from memory.stores import QdrantMemoryStore
from memory.stores.qdrant_store import QdrantConfig

config = QdrantConfig(
    host="localhost",
    port=6333,
    collection_name="agentic_sql_memories",
    vector_size=1536,
    distance="Cosine",  # Cosine, Euclid, Dot
    on_disk=False,  # Store vectors on disk for large collections
)

store = QdrantMemoryStore(config)
await store.connect()

# Vector search with filtering
results = await store.retrieve(
    query="",
    query_vector=embedding,
    memory_type=MemoryType.SEMANTIC,
    score_threshold=0.7,
    limit=10,
)
```

### Multi-Tenancy

```python
config = QdrantConfig(
    host="localhost",
    port=6333,
    collection_name="memories",
    # Multi-tenancy options
    namespace="production",      # Logical namespace
    tenant_id="customer_123",    # Tenant isolation
    group_id="project_abc",      # Group/project isolation
)

store = QdrantMemoryStore(config)
await store.connect()

# Search within namespace/tenant (auto-filtered)
results = await store.retrieve(query_vector=embedding, limit=10)

# Override filters per query
results = await store.retrieve(
    query_vector=embedding,
    namespace="staging",
    tenant_id="customer_456",
)
```

### Cloud Deployment

```python
config = QdrantConfig(
    url="https://your-cluster.qdrant.io",  # Full URL for cloud
    api_key="your-api-key",
    collection_name="memories",
)
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

### Usage

```python
from memory.stores import OpenSearchMemoryStore
from memory.stores.opensearch_store import OpenSearchConfig

config = OpenSearchConfig(
    hosts=["https://localhost:9200"],
    index_name="agentic_sql_memories",
    username="admin",
    password="admin",
    vector_dimension=1536,
    use_ssl=True,
    verify_certs=False,
)

store = OpenSearchMemoryStore(config)
await store.connect()

# Hybrid search (vector + text)
results = await store.retrieve(
    query="database optimization tips",
    query_vector=embedding,
    use_hybrid=True,  # Combine k-NN with BM25
    limit=10,
)

# Pure text search
text_results = await store.search_text(
    query="index performance",
    limit=10,
)
```

### Multi-Tenancy

```python
config = OpenSearchConfig(
    hosts=["https://localhost:9200"],
    index_name="memories",
    index_prefix="prod_",         # Index becomes "prod_memories"
    # Multi-tenancy options
    namespace="production",       # Stored as document field
    tenant_id="customer_123",     # Stored as document field
    group_id="project_abc",       # Stored as document field
    # Sharding
    number_of_shards=3,
    number_of_replicas=1,
)

store = OpenSearchMemoryStore(config)
await store.connect()

# Queries auto-filter by namespace/tenant/group
results = await store.retrieve(query="search term", limit=10)

# Override filters per query
results = await store.retrieve(
    query="search term",
    namespace="staging",
    tenant_id="customer_456",
)

# Count within tenant
count = await store.count(tenant_id="customer_123")
```

### HNSW Parameters

```python
config = OpenSearchConfig(
    ef_construction=512,  # Higher = better recall, slower indexing
    m=16,  # Higher = better recall, more memory
)
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

### Usage

```python
from memory.stores import Neo4jMemoryStore
from memory.stores.neo4j_store import Neo4jConfig

config = Neo4jConfig(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password",
    database="neo4j",
)

store = Neo4jMemoryStore(config)
await store.connect()

# Store with relationships
memory.metadata = {
    "relationships": [
        {"target_id": "uuid-1", "type": "relates_to"},
        {"target_id": "uuid-2", "type": "depends_on"},
    ]
}
await store.store(memory)

# Graph traversal retrieval
results = await store.retrieve(
    query="uuid-of-starting-node",
    depth=2,  # Traverse up to 2 hops
    limit=10,
)
```

### Multi-Tenancy

```python
config = Neo4jConfig(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password",
    database="neo4j",
    # Multi-tenancy options
    namespace="production",      # Added as node label (NS_production)
    tenant_id="customer_123",    # Stored as node property
    group_id="project_abc",      # Stored as node property
    node_label="Memory",         # Base label for nodes
    use_namespace_label=True,    # Add namespace as additional label
)

store = Neo4jMemoryStore(config)
await store.connect()

# Nodes are labeled: Memory:NS_production
# Queries auto-filter by namespace/tenant/group
results = await store.retrieve(query="search term", limit=10)

# Override filters per query
results = await store.retrieve(
    query="search term",
    namespace="staging",
    tenant_id="customer_456",
)
```

### Features

- Entity relationships
- Multi-tenancy (namespace as labels, tenant/group as properties)
- Graph traversal for context expansion
- Cypher query support
- Automatic index creation
- Relationship types

---

## Combining Stores

You can use multiple stores together for different purposes:

```python
from memory.manager import MemoryManager

# Use SQLite for quick access, Qdrant for semantic search
sqlite_store = SQLiteMemoryStore(SQLiteConfig())
qdrant_store = QdrantMemoryStore(QdrantConfig())

await sqlite_store.connect()
await qdrant_store.connect()

# Store in both
async def store_memory(memory):
    await sqlite_store.store(memory)
    if memory.embedding:
        await qdrant_store.store(memory)

# Quick lookup from SQLite
facts = await sqlite_store.get_by_type(MemoryType.FACT)

# Semantic search from Qdrant
similar = await qdrant_store.retrieve(
    query="",
    query_vector=embedding,
)
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
- Need payload filtering
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
