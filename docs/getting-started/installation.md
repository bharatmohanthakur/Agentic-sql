# Installation

Get Agentic SQL up and running in minutes.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10, 3.11, or 3.12 |
| pip or uv | Latest |

---

## Installation Methods

=== "uv (Recommended)"

    ```bash
    # Clone repository
    git clone https://github.com/bharatmohanthakur/Agentic-sql.git
    cd Agentic-sql

    # Install with uv
    uv sync
    ```

=== "pip"

    ```bash
    # Clone repository
    git clone https://github.com/bharatmohanthakur/Agentic-sql.git
    cd Agentic-sql

    # Install with pip
    pip install -e .
    ```

=== "pip (from PyPI)"

    ```bash
    # Coming soon
    pip install agentic-sql
    ```

---

## Dependencies

### Core Dependencies

These are installed automatically:

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥2.16.0 | OpenAI/Azure OpenAI client |
| `pydantic` | ≥2.0.0 | Data validation |
| `pyodbc` | ≥5.3.0 | MS SQL Server connectivity |
| `python-dotenv` | ≥1.2.1 | Environment variables |
| `typing-extensions` | ≥4.0.0 | Type hints |

### Optional Dependencies

Install extras based on your needs:

=== "LLM Providers"

    ```bash
    # OpenAI (included in core)
    pip install -e ".[openai]"

    # Anthropic Claude
    pip install -e ".[anthropic]"

    # All LLM providers
    pip install -e ".[all-llms]"
    ```

    | Extra | Packages |
    |-------|----------|
    | `openai` | openai≥1.0.0 |
    | `anthropic` | anthropic≥0.18.0 |
    | `all-llms` | openai, anthropic, google-generativeai |

=== "Databases"

    ```bash
    # PostgreSQL
    pip install -e ".[postgres]"

    # MySQL
    pip install -e ".[mysql]"

    # SQLite
    pip install -e ".[sqlite]"
    ```

    | Extra | Packages |
    |-------|----------|
    | `postgres` | asyncpg≥0.28.0, psycopg2-binary≥2.9.0 |
    | `mysql` | aiomysql≥0.2.0 |
    | `sqlite` | aiosqlite≥0.19.0 |
    | `snowflake` | snowflake-connector-python≥3.0.0 |
    | `bigquery` | google-cloud-bigquery≥3.0.0 |

=== "Memory & Storage"

    ```bash
    # Vector store (ChromaDB)
    pip install -e ".[vector]"

    # Graph store (Neo4j)
    pip install -e ".[graph]"

    # Full memory system
    pip install -e ".[memory]"
    ```

    | Extra | Packages |
    |-------|----------|
    | `vector` | chromadb≥0.4.0, pgvector≥0.2.0 |
    | `graph` | neo4j≥5.0.0 |

=== "API & Visualization"

    ```bash
    # FastAPI server
    pip install -e ".[api]"

    # JWT authentication
    pip install -e ".[auth]"

    # Visualization
    pip install -e ".[viz]"
    ```

    | Extra | Packages |
    |-------|----------|
    | `api` | fastapi≥0.100.0, uvicorn≥0.22.0, sse-starlette≥1.6.0 |
    | `auth` | PyJWT≥2.8.0 |
    | `viz` | plotly≥5.0.0, pandas≥2.0.0 |

=== "Full Installation"

    ```bash
    # Everything
    pip install -e ".[all]"
    ```

---

## System Requirements

### MS SQL Server

If connecting to MS SQL Server, install ODBC Driver:

=== "Windows"

    Download from [Microsoft](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)

=== "macOS"

    ```bash
    brew install microsoft/mssql-release/msodbcsql18
    ```

=== "Ubuntu/Debian"

    ```bash
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
    curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list > /etc/apt/sources.list.d/mssql-release.list
    apt-get update
    ACCEPT_EULA=Y apt-get install -y msodbcsql18
    ```

---

## Verify Installation

```python
# Test import
from src.intelligence.meta_agent import MetaAgent
print("✓ Agentic SQL installed successfully!")
```

---

## Next Steps

- [Quick Start](quickstart.md) - Your first query in 5 minutes
- [Configuration](configuration.md) - Environment setup
