# Agentic SQL - Pure LLM Intelligence

A truly intelligent Text-to-SQL system that works on **ANY database** with **ZERO hardcoded rules**. Everything is driven by LLM intelligence.

## Philosophy

> **"No rules, no patterns, no hardcoding - just pure LLM understanding"**

Traditional Text-to-SQL systems rely on:
- Hardcoded regex patterns for SQL dialects
- Pre-defined error handling rules
- Manual training with example queries
- Fixed prompt templates

**This system is different.** The LLM:
- **Discovers** the database schema and dialect automatically
- **Generates** its own test questions to learn
- **Learns** from every success and failure
- **Adapts** to any database without code changes

## Key Features

### 1. Auto-Discovery
When connecting to a new database, the LLM automatically:
- Probes to detect SQL dialect (MSSQL, PostgreSQL, MySQL, SQLite, etc.)
- Discovers table structures and relationships
- Understands data types and constraints
- Learns naming conventions used in the schema

```python
# Just connect - LLM figures out everything else
await agent.connect(db_executor=database.execute)
# Output: "Connected to MS SQL Server: MyDatabase"
# Output: "LLM detected dialect: mssql"
# Output: "LLM discovered 19 schema insights"
```

### 2. Auto-Learning
The system trains itself through an intelligent loop:

```
┌─────────────────────────────────────────────────────────┐
│                    AUTO-LEARN FLOW                       │
├─────────────────────────────────────────────────────────┤
│  1. LLM explores database and understands domain        │
│  2. LLM generates diverse test questions                │
│  3. LLM runs questions and learns from results          │
│  4. LLM identifies weak areas and trains more           │
│  5. Knowledge is stored for future queries              │
└─────────────────────────────────────────────────────────┘
```

```python
# One line to train on any database
results = await agent.auto_learn(intensity="medium")
# Output: "Domain discovered: legislation"
# Output: "Questions generated: 15"
# Output: "Success rate: 86.7%"
# Output: "LEARNED: 'Conflicts' should be 'LegislationConflicts'"
```

### 3. Intelligent Query Processing
Every query goes through the THINK-RESEARCH-DESIGN-EXECUTE-LEARN loop:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  THINK   │───▶│ RESEARCH │───▶│  DESIGN  │───▶│ EXECUTE  │───▶│  LEARN   │
│          │    │          │    │          │    │          │    │          │
│ Classify │    │ Find     │    │ Generate │    │ Run &    │    │ Store    │
│ problem  │    │ similar  │    │ custom   │    │ fix if   │    │ solution │
│ type     │    │ solutions│    │ SQL      │    │ needed   │    │ & errors │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 4. Self-Healing
When errors occur, the LLM:
- Analyzes the error message
- Searches database for correct table/column names
- Learns corrections for future queries
- Retries with fixed SQL

```python
# LLM automatically learns from errors:
# "LEARNED: 'Categories' should be 'Category'"
# "LEARNED: For 'category queries' use table 'Legislations'"
```

### 5. Dynamic Prompts
No fixed prompt templates. Every prompt is generated dynamically based on:
- Problem type classification
- Similar successful solutions
- Previous failure patterns
- Learned dialect specifics
- Schema insights

## Installation

```bash
# Clone the repository
git clone https://github.com/bharatmohanthakur/Agentic-sql.git
cd Agentic-sql

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

```python
import asyncio
from src.intelligence.meta_agent import MetaAgent
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
from src.database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

async def main():
    # 1. Setup LLM
    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key="your-api-key",
        azure_endpoint="your-endpoint",
        azure_deployment="gpt-4o",
    ))

    # 2. Create the intelligent agent
    agent = MetaAgent(llm_client=llm)

    # 3. Connect to any database
    db = MSSQLAdapter(ConnectionConfig(
        name="my_database",
        db_type=DatabaseType.MSSQL,
        host="server.database.windows.net",
        database="MyDatabase",
        username="user",
        password="password",
    ))
    await db.connect()

    # 4. Connect agent (auto-discovers schema & dialect)
    await agent.connect(db_executor=db.execute)

    # 5. (Optional) Auto-train on the database
    await agent.auto_learn(intensity="light")

    # 6. Query naturally
    result = await agent.query("How many orders were placed last month?")

    if result["success"]:
        print(f"SQL: {result['sql']}")
        print(f"Rows: {result['row_count']}")
        print(f"Data: {result['data']}")

asyncio.run(main())
```

## Supported Databases

The system auto-detects and adapts to:

| Database | Status | Auto-Detected Features |
|----------|--------|----------------------|
| MS SQL Server | ✅ | TOP, GETDATE, square brackets |
| PostgreSQL | ✅ | LIMIT, NOW(), double quotes |
| MySQL | ✅ | LIMIT, backticks, NOW() |
| SQLite | ✅ | LIMIT, datetime functions |
| Oracle | ✅ | ROWNUM, SYSDATE |

**No configuration needed** - the LLM probes and learns the dialect automatically.

## Test Results

Comprehensive testing on a legislation database:

| Test Category | Success Rate |
|---------------|--------------|
| Simple Queries | 100% |
| Filtering | 100% |
| Date Queries | 100% |
| Aggregations | 100% |
| Comparisons | 100% |
| Rankings | 100% |
| Complex Joins | 100% |
| Business Questions | 100% |
| **Overall** | **100%** |

### Stress Test Results

| Test Type | Success Rate | Notes |
|-----------|--------------|-------|
| Window Functions | ✅ | ROW_NUMBER, DENSE_RANK |
| Recursive CTEs | ✅ | Hierarchical queries |
| PIVOT Tables | ✅ | Dynamic pivoting |
| Arabic Language | ✅ | Multi-language support |
| SQL Injection | ✅ | Handled safely |
| Typos/Misspellings | ✅ | "Shwo teh legislatoins" works |
| Vague Questions | ✅ | "What's wrong with our data?" |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MetaAgent                                │
│                  (Pure LLM Intelligence)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   CONNECT    │  │  AUTO-LEARN  │  │       QUERY          │  │
│  │              │  │              │  │                      │  │
│  │ • Probe      │  │ • Explore    │  │ • THINK (classify)   │  │
│  │   dialect    │  │   domain     │  │ • RESEARCH (find)    │  │
│  │ • Discover   │  │ • Generate   │  │ • DESIGN (generate)  │  │
│  │   schema     │  │   questions  │  │ • EXECUTE (run)      │  │
│  │ • Learn      │  │ • Test &     │  │ • LEARN (store)      │  │
│  │   insights   │  │   learn      │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                      MetaKnowledge                               │
│  • Successful solutions    • Name corrections                    │
│  • Failed attempts         • Table relationships                 │
│  • Dialect learnings       • Schema insights                     │
│  • Fix strategies          • Problem type patterns               │
└─────────────────────────────────────────────────────────────────┘
```

## How It Learns

### 1. From Successes
Every successful query is stored with:
- The natural language question
- The problem type classification
- The generated SQL
- Whether corrections were needed

### 2. From Failures
When queries fail, the LLM:
- Extracts the problematic name from error
- Searches database for correct name
- Stores the correction for future use
- Analyzes root cause for improvements

### 3. From Auto-Learning
The `auto_learn()` function:
- Generates domain-specific questions
- Tests them against the database
- Identifies weak areas
- Generates targeted questions to improve

## Configuration

### Environment Variables

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com

# Or OpenAI
OPENAI_API_KEY=sk-...

# Or Anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### Auto-Learn Intensities

```python
# Light - Quick training (5 questions)
await agent.auto_learn(intensity="light")

# Medium - Standard training (15 questions)
await agent.auto_learn(intensity="medium")

# Heavy - Comprehensive training (30 questions)
await agent.auto_learn(intensity="heavy")
```

## Knowledge Persistence

All learnings are automatically saved to `~/.vanna/meta_agent.json`:

```json
{
  "successful_solutions": [...],
  "failed_attempts": [...],
  "dialect_learnings": [...],
  "schema_insights": [...],
  "name_corrections": {
    "categories": "Category",
    "conflicts": "LegislationConflicts"
  },
  "table_relationships": {
    "category queries": "Legislations"
  }
}
```

## API Reference

### MetaAgent

```python
class MetaAgent:
    async def connect(db_executor) -> Dict
        """Connect to database, auto-discover schema and dialect"""

    async def query(question: str) -> Dict
        """Process natural language question and return SQL results"""

    async def auto_learn(intensity: str) -> Dict
        """Self-train on the connected database"""

    def get_stats() -> Dict
        """Get current knowledge statistics"""
```

### Query Response

```python
{
    "success": True,
    "sql": "SELECT ...",
    "data": [...],
    "row_count": 42,
    "iterations": 1,
    "problem_type": "aggregation",
    "execution_time_ms": 1234
}
```

## Contributing

Contributions are welcome! The key principle is:

> **No hardcoding** - If you find yourself writing a rule or pattern,
> ask "Can the LLM figure this out?" The answer is usually yes.

## License

MIT

## Credits

Built with pure LLM intelligence using:
- GPT-4o / Azure OpenAI
- Claude (Anthropic)
- Inspired by the vision of truly intelligent AI systems
