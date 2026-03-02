# Ontology-Based Multi-Agent System: Detailed Architecture Plan

## Vision: The Semantic Space Funnel

This system implements a 4-layer semantic space architecture where queries flow
top-down through progressively lower abstraction levels, each enriching the context:

```
 ┌─────────────────────────────────────────────────────────────┐
 │          LAYER 1: ONTOLOGY / FACT-BASED MODEL               │
 │             (HIGHEST Semantic Space)                         │
 │                                                             │
 │  "The home address where <Person> is registered             │
 │   is <Address>"                                             │
 │                                                             │
 │  Tech: OWL/RDF, owlready2, rdflib                          │
 │  Store: Neo4j (TBox schema graph)                           │
 │  Agent: OntologyAgent (concept resolution, fact reasoning)  │
 ├─────────────────────────────────────────────────────────────┤
 │          LAYER 2: KNOWLEDGE GRAPH                           │
 │             (HIGH Semantic Space)                            │
 │                                                             │
 │  [Person] → hasAddress → [Address]                          │
 │                                                             │
 │  Tech: Neo4j 5.x, Cypher, Vector Indexes                   │
 │  Store: Neo4j (ABox instance graph + embeddings)            │
 │  Agent: KnowledgeGraphAgent (Cypher gen, GraphRAG)          │
 ├─────────────────────────────────────────────────────────────┤
 │          LAYER 3: CANONICAL MODEL                           │
 │             (MEDIUM Semantic Space)                          │
 │                                                             │
 │  { "Person": { "Name": {...}, "Address": {...} } }         │
 │                                                             │
 │  Tech: Pydantic, JSON Schema, Schema Registry               │
 │  Store: Snowflake VARIANT + Dynamic Tables                  │
 │  Agent: SchemaAgent (validation, translation)               │
 ├─────────────────────────────────────────────────────────────┤
 │          LAYER 4: DIMENSIONAL MODEL                         │
 │             (LOWEST Semantic Space)                          │
 │                                                             │
 │  Dim_Employee (StreetAddress VARCHAR)                        │
 │                                                             │
 │  Tech: Kimball Star Schema, Snowflake SQL                   │
 │  Store: Snowflake (Facts + Dimensions + Bridges)            │
 │  Agent: DimensionalSQLAgent (star-join SQL, SCD filtering)  │
 └─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component           | Technology                    | Purpose                                    |
|---------------------|-------------------------------|--------------------------------------------|
| Data Warehouse      | **Snowflake**                 | Dimensional model, canonical storage       |
| Graph Database      | **Neo4j 5.x**                | Knowledge graph, ontology schema, lineage  |
| LLM / Reasoning     | **AWS Bedrock (Claude)**      | Agent reasoning, query generation          |
| Embeddings          | **AWS Bedrock (Titan V2)**    | Semantic search across all layers          |
| Orchestration       | **AWS Bedrock Agents**        | SUPERVISOR_ROUTER multi-agent pattern      |
| Ontology Engine     | **owlready2 + rdflib**        | OWL/RDF management, SPARQL, reasoning      |
| Python Framework    | **Pydantic**                  | Canonical models, validation, schemas      |
| Agent Framework     | **Custom (this repo)**        | ReAct + Reflection agents                  |

---

## Folder Structure Created

```
src/
├── ontology/                          # LAYER 1: Ontology (Highest Semantic Space)
│   ├── __init__.py
│   ├── fact_model.py                  # ElementaryFact, ObjectType, RoleType, Constraint
│   ├── ontology_store.py              # Neo4j persistence for ontology definitions
│   ├── ontology_agent.py              # Agent: concept resolution, fact reasoning
│   ├── reasoner.py                    # Forward/backward chaining, constraint checking
│   └── schemas/
│       └── __init__.py                # OWL/RDF schema files
│
├── knowledge_graph/                   # LAYER 2: Knowledge Graph (High Semantic Space)
│   ├── __init__.py
│   ├── graph_model.py                 # GraphNode, GraphEdge, GraphSchema, GraphPath
│   ├── neo4j_knowledge_graph.py       # Neo4j CRUD, Cypher, vector search, bulk import
│   ├── graph_agent.py                 # Agent: Cypher generation, graph queries
│   ├── graph_rag.py                   # GraphRAG: vector search + graph traversal
│   └── embeddings.py                  # Bedrock Titan graph node embeddings
│
├── canonical/                         # LAYER 3: Canonical Model (Medium Semantic Space)
│   ├── __init__.py
│   ├── canonical_model.py             # CanonicalEntity, CanonicalRelationship
│   ├── schema_registry.py             # Schema versioning, compatibility checking
│   ├── schema_agent.py                # Agent: validation, coercion, type inference
│   └── translator.py                  # Cross-layer translation (ontology↔canonical↔dimensional)
│
├── dimensional/                       # LAYER 4: Dimensional Model (Lowest Semantic Space)
│   ├── __init__.py
│   ├── dimensional_model.py           # DimensionTable, FactTable, BridgeTable, StarSchema
│   ├── snowflake_client.py            # Snowflake DDL, query, metadata operations
│   ├── sql_generation_agent.py        # Agent: star-join SQL, SCD filtering
│   └── lineage.py                     # Data lineage: dimensional → ontology tracing
│
└── orchestration/                     # MULTI-AGENT ORCHESTRATOR
    ├── __init__.py
    ├── semantic_router.py             # Routes queries to correct semantic layer
    ├── bedrock_orchestrator.py        # AWS Bedrock SUPERVISOR_ROUTER pattern
    ├── embedding_service.py           # Unified embeddings across all layers
    └── pipeline.py                    # Cross-layer pipeline execution
```

---

## Detailed Layer Plans

### Layer 1: Ontology / Fact-Based Model

**What it does:** Defines the domain through irreducible "elementary facts" using
Object-Role Modeling (ORM). This is the HIGHEST semantic layer - the source of truth
for what domain concepts mean.

**Key Classes:**
- `ElementaryFact`: Irreducible domain fact (e.g., "Person isRegisteredAt Address")
- `ObjectType`: Named entity types (Person, Address) with entity/value classification
- `RoleType`: Roles objects play in facts (registered_person, home_address)
- `Constraint`: Business rules (uniqueness, mandatory, subset, exclusion)
- `OntologyDefinition`: Complete domain ontology container
- `FactReasoner`: Forward/backward chaining inference engine

**Storage:** Neo4j graph (schema as nodes + edges)
```
(Ontology)-[:DEFINES_TYPE]->(ObjectType)
(Ontology)-[:DEFINES_FACT]->(ElementaryFact)
(ElementaryFact)-[:HAS_ROLE]->(RoleType)-[:PLAYED_BY]->(ObjectType)
(Constraint)-[:CONSTRAINS]->(RoleType)
```

**Agent Behavior (OntologyAgent):**
1. THINK: Maps NL question to ontology concepts using LLM
2. ACT: Retrieves relevant facts, expands context through reasoning
3. REFLECT: Validates semantic coverage
4. OUTPUT: Semantic context dict with fact_readings, entity_relationships

**Dependencies:** `owlready2>=0.46`, `rdflib>=7.0`

---

### Layer 2: Knowledge Graph (Neo4j)

**What it does:** Represents domain knowledge as a network of typed nodes and edges.
Bridges ontology facts to queryable graph structures.

**Key Classes:**
- `GraphNode`: Entity with labels, properties, and vector embedding
- `GraphEdge`: Typed relationship with properties
- `GraphSchema`: Schema derived from ontology (labels, relationship types)
- `KnowledgeGraph`: Neo4j operations (CRUD, Cypher, vector search)
- `GraphRAG`: Hybrid retrieval (vector search + graph traversal)
- `GraphEmbeddingService`: Bedrock Titan embedding generation

**Storage:** Neo4j property graph + vector indexes
```cypher
-- Vector index for semantic search
CREATE VECTOR INDEX entity_embedding
FOR (n:Entity) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}

-- Hybrid search: vector + graph traversal
CALL db.index.vector.queryNodes('entity_embedding', 5, $queryVector)
YIELD node, score
MATCH (node)-[r*1..2]->(related)
RETURN node, score, collect(related)
```

**Agent Behavior (KnowledgeGraphAgent):**
1. THINK: Plan graph query from semantic context
2. ACT: Generate Cypher via LLM, execute on Neo4j
3. REFLECT: Validate result coverage
4. OUTPUT: Graph query results with relationship context

**Dependencies:** `neo4j>=5.0.0`, `graphdatascience>=1.6`

---

### Layer 3: Canonical Model

**What it does:** The "lowest common denominator" integration layer. Defines
standardized Pydantic models that all systems agree on.

**Key Classes:**
- `CanonicalEntity`: Base entity with properties, relationships, metadata
- `CanonicalMetadata`: Schema version, source system, lineage tracking
- `SchemaRegistry`: Version management with backward/forward compatibility
- `ModelTranslator`: Converts between ontology ↔ canonical ↔ graph ↔ dimensional

**Storage:** Snowflake VARIANT columns (schema-on-read) + Dynamic Tables
```sql
-- Raw landing: VARIANT for flexibility
CREATE TABLE canonical_raw (id STRING, data VARIANT, ingested_at TIMESTAMP_NTZ);

-- Materialized: Dynamic table for typed access
CREATE DYNAMIC TABLE canonical_persons
  TARGET_LAG = '5 minutes'
  WAREHOUSE = transform_wh
AS SELECT
  data:personId::STRING AS person_id,
  data:name:first::STRING AS first_name,
  data:name:last::STRING AS last_name
FROM canonical_raw;
```

**Agent Behavior (SchemaAgent):**
1. THINK: Identify entity type (heuristic or LLM inference)
2. ACT: Validate against schema registry, coerce if needed
3. REFLECT: Report validation results
4. OUTPUT: Validated canonical entity or error report

**Schema Evolution:** Semantic versioning (MAJOR.MINOR.PATCH), backward
compatibility enforced, version embedded in every payload.

---

### Layer 4: Dimensional Model (Snowflake)

**What it does:** Flattened, denormalized star schemas optimized for analytical
queries. This is where "premature compression" happens.

**Key Classes:**
- `FactTable`: Measures + dimension FKs (transaction, periodic, accumulating)
- `DimensionTable`: Descriptive attributes with SCD support (Types 0-6)
- `BridgeTable`: M:M relationship resolution with weight factors
- `StarSchema`: Complete dimensional model container with DDL generation
- `LineageTracker`: Traces every column back through canonical to ontology

**Storage:** Snowflake star schema
```sql
CREATE TABLE Fact_Sales (
    DateKey INT, CustomerKey INT, ProductKey INT,
    SalesAmount DECIMAL(18,2), Quantity INT
) CLUSTER BY (DateKey, CustomerKey);

CREATE TABLE Dim_Employee (
    EmployeeKey INT, EmployeeID VARCHAR, FullName VARCHAR,
    StreetAddress VARCHAR,  -- Premature compression of Address concept
    effective_date DATE, expiration_date DATE, is_current BOOLEAN
);
```

**Agent Behavior (DimensionalSQLAgent):**
1. THINK: Analyze question + upstream semantic context
2. ACT: Generate star-join SQL with SCD filters, execute on Snowflake
3. REFLECT: Validate results, check data quality
4. OUTPUT: SQL + query results + row count

**Key Features:**
- Understands star-join patterns (fact → dimension, never dimension → dimension)
- Auto-applies SCD Type 2 filters (WHERE is_current = TRUE)
- Uses bridge table weight factors for M:M aggregations
- Snowflake-specific optimizations (QUALIFY, FLATTEN, clustering awareness)

---

### Orchestration Layer (AWS Bedrock)

**What it does:** Coordinates all 4 agents using the SUPERVISOR_ROUTER pattern.

**Routing Logic:**
| Query Type                           | Layers Used              | Pattern          |
|--------------------------------------|--------------------------|------------------|
| "What is a Customer?"                | Ontology only            | Direct route     |
| "How do Orders relate to Products?"  | Ontology → KG            | Sequential       |
| "Total revenue by region last Q"     | Ontology → KG → SQL      | Full pipeline    |
| "SELECT * FROM orders LIMIT 10"      | SQL only                 | Pass-through     |

**AWS Bedrock Configuration:**
- Supervisor: Claude Sonnet 4 (SUPERVISOR_ROUTER mode)
- Specialist agents: Claude Haiku (cost-effective for focused tasks)
- Embeddings: Titan Text Embeddings V2 (1024 dimensions)
- Guardrails: Block destructive SQL, PII detection

**Pipeline Flow:**
```
User Question
    │
    ▼
┌──────────────────┐
│  Semantic Router  │  Classifies query → determines layers
└────────┬─────────┘
         │
    ┌────▼────┐
    │ Simple? │
    └──┬──┬───┘
   Yes │  │ No (cross-layer)
       │  │
  ┌────▼┐ └────────────────────────────────┐
  │Route│                                   │
  │to 1 │     ┌──────────────────┐          │
  │agent│     │  Full Pipeline:   │          │
  └─────┘     │  Ontology Agent   │──context──▶
              │  KG Agent         │──context──▶
              │  Schema Agent     │──context──▶
              │  SQL Agent        │──results──▶
              └──────────────────┘
                       │
                  Final Response
```

---

## Data Flow Example

**User asks:** "Show all employees registered in Amsterdam"

```
STEP 1: Semantic Router
  → Classified as: DIMENSIONAL (primary) + ONTOLOGY + KG (secondary)
  → Cross-layer pipeline activated

STEP 2: Ontology Agent
  Input:  "Show all employees registered in Amsterdam"
  Finds:  FactType "PersonRegisteredAtAddress" (reading: "The home address
          where <Person> is registered is <Address>")
          FactType "AddressInCity" (reading: "<Address> is in <City>")
  Output: { fact_readings: [...], entity_relationships: [(Person, isRegisteredAt, Address),
            (Address, isInCity, City)] }

STEP 3: Knowledge Graph Agent
  Input:  Ontology context + question
  Generates Cypher:
    MATCH (e:Employee)-[:IS_REGISTERED_AT]->(a:Address)-[:IS_IN]->(c:City)
    WHERE c.name = 'Amsterdam'
    RETURN e, a, c LIMIT 100
  Output: { cypher: "...", results: [...], result_count: 42 }

STEP 4: Dimensional SQL Agent
  Input:  Ontology context + KG context + question
  Generates SQL:
    SELECT e.FullName, e.StreetAddress, e.City
    FROM analytics.Dim_Employee e
    WHERE e.City = 'Amsterdam'
      AND e.is_current = TRUE  -- SCD Type 2 filter
    LIMIT 1000
  Output: { sql: "...", data: [...], row_count: 42 }
```

---

## Embedding Strategy

| Layer        | What to Embed                           | Dimensions | Store              |
|-------------|-----------------------------------------|------------|---------------------|
| Ontology    | Concept names + descriptions + facts    | 1024       | Neo4j Vector Index  |
| KG          | Node properties + neighbor context      | 1024       | Neo4j Vector Index  |
| Canonical   | Entity type + properties                | 512        | Snowflake VECTOR    |
| Dimensional | Table name + description + columns      | 512        | Snowflake VECTOR    |

All embeddings generated by **Amazon Titan Text Embeddings V2** via `UnifiedEmbeddingService`.

---

## Phased Implementation Roadmap

### Phase 1: Core Foundation (Current Sprint)
- [x] Create folder structure for all 5 layers
- [x] Implement ontology fact model (ElementaryFact, ObjectType, RoleType)
- [x] Implement ontology store (Neo4j persistence)
- [x] Implement fact reasoner (forward/backward chaining)
- [x] Implement ontology agent (ReAct pattern)
- [x] Implement knowledge graph model and Neo4j operations
- [x] Implement KG agent with Cypher generation
- [x] Implement GraphRAG (hybrid vector + graph retrieval)
- [x] Implement canonical model (Pydantic entities)
- [x] Implement schema registry with versioning
- [x] Implement model translator (cross-layer conversion)
- [x] Implement dimensional model (star schema definitions)
- [x] Implement Snowflake client
- [x] Implement dimensional SQL agent
- [x] Implement lineage tracker
- [x] Implement semantic router
- [x] Implement Bedrock orchestrator (SUPERVISOR_ROUTER)
- [x] Implement unified embedding service
- [x] Implement semantic pipeline

### Phase 2: Integration & Testing
- [ ] Wire all agents into the orchestrator
- [ ] Create sample ontology for a business domain (HR, Sales, etc.)
- [ ] Populate Neo4j with ontology schema + instance data
- [ ] Create Snowflake dimensional model (DDL)
- [ ] End-to-end test: NL question → ontology → KG → SQL → results
- [ ] Add Bedrock Guardrails for SQL safety + PII detection

### Phase 3: Production Hardening
- [ ] Deploy agents as managed Bedrock Agents with Lambda action groups
- [ ] Set up Step Functions for deterministic multi-agent workflows
- [ ] Embed ontology concepts into Neo4j Vector Index
- [ ] Embed table metadata into Snowflake VECTOR columns
- [ ] Configure Bedrock Knowledge Bases for documentation RAG
- [ ] Add comprehensive error handling and retry logic
- [ ] Performance benchmarking and optimization

### Phase 4: Advanced Features
- [ ] Ontology auto-discovery from existing Snowflake schemas
- [ ] LLM-powered lineage parsing from ETL SQL
- [ ] Community detection for entity clustering (GDS algorithms)
- [ ] Time-variant bridge tables for M:M relationship history
- [ ] Multi-tenant ontology support (namespace isolation)
- [ ] Real-time CDC pipeline (Snowflake Streams → Dynamic Tables → Neo4j sync)

---

## Dependencies to Add

```toml
[project.optional-dependencies]
ontology = [
    "owlready2>=0.46",
    "rdflib>=7.0",
]
knowledge-graph = [
    "neo4j>=5.0.0",
    "graphdatascience>=1.6",
]
semantic-platform = [
    "owlready2>=0.46",
    "rdflib>=7.0",
    "neo4j>=5.0.0",
    "graphdatascience>=1.6",
    "boto3>=1.28.0",
    "snowflake-connector-python>=3.0.0",
]
```

---

## Key Design Decisions

1. **Canonical model as the pivot format**: All translations go through
   Layer 3. Ontology ↔ Canonical ↔ Graph/Dimensional. This reduces
   N*(N-1) translations to 2*N.

2. **SUPERVISOR_ROUTER over plain SUPERVISOR**: Routes simple queries
   directly to one agent (cheaper, faster). Falls back to full
   orchestration for complex cross-layer queries.

3. **Haiku for specialists, Sonnet for supervisor**: The supervisor
   needs strong reasoning to decompose queries. Specialists just
   need focused task execution.

4. **Co-located embeddings**: Store embeddings where the data lives
   (Neo4j Vector Index for graph, Snowflake VECTOR for tables).
   Reduces data movement and enables hybrid search.

5. **Lineage as a first-class concern**: Every dimensional column
   traces back through canonical to ontology. The LineageTracker
   stores this in Neo4j for graph traversal.

6. **Schema evolution via registry**: Backward compatibility enforced.
   Schema version embedded in every canonical payload. Dynamic tables
   handle version coexistence naturally.
