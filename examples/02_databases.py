#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 2: CONNECTING TO DIFFERENT DATABASES
=============================================================================

This tutorial shows how to connect to various database types.
The agent auto-detects the SQL dialect and adapts automatically.

Supported databases:
- MS SQL Server
- PostgreSQL
- MySQL
- SQLite
- (More coming: Oracle, Snowflake, BigQuery)

Run: python examples/02_databases.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# DATABASE CONNECTION EXAMPLES
# =============================================================================

async def connect_mssql():
    """
    Connect to Microsoft SQL Server

    Requirements:
    - pip install pyodbc
    - ODBC Driver 17/18 for SQL Server installed
    """
    from database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

    db = MSSQLAdapter(ConnectionConfig(
        name="mssql_example",
        db_type=DatabaseType.MSSQL,

        # Option 1: Individual parameters
        host="server.database.windows.net,1433",  # Include port after comma
        database="MyDatabase",
        username="myuser",
        password="mypassword",

        # Option 2: Azure AD authentication (uncomment)
        # host="server.database.windows.net",
        # database="MyDatabase",
        # authentication="ActiveDirectoryInteractive",
    ))

    await db.connect()
    print(f"✓ Connected to MSSQL: {db.config.database}")

    # Test query
    result = await db.execute("SELECT TOP 5 name FROM sys.tables")
    print(f"  Tables: {[r['name'] for r in result]}")

    return db


async def connect_postgresql():
    """
    Connect to PostgreSQL

    Requirements:
    - pip install asyncpg psycopg2-binary
    """
    from database.multi_db import PostgreSQLAdapter, ConnectionConfig, DatabaseType

    # Option 1: Individual parameters
    db = PostgreSQLAdapter(ConnectionConfig(
        name="postgres_example",
        db_type=DatabaseType.POSTGRESQL,
        host="localhost",
        port=5432,
        database="mydb",
        username="postgres",
        password="password",
    ))

    # Option 2: Connection string (uncomment)
    # db = PostgreSQLAdapter(ConnectionConfig(
    #     name="postgres_example",
    #     db_type=DatabaseType.POSTGRESQL,
    #     connection_string="postgresql://user:pass@localhost:5432/mydb",
    # ))

    await db.connect()
    print(f"✓ Connected to PostgreSQL: {db.config.database}")

    # Test query
    result = await db.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
        LIMIT 5
    """)
    print(f"  Tables: {[r['tablename'] for r in result]}")

    return db


async def connect_mysql():
    """
    Connect to MySQL / MariaDB

    Requirements:
    - pip install aiomysql
    """
    from database.multi_db import MySQLAdapter, ConnectionConfig, DatabaseType

    db = MySQLAdapter(ConnectionConfig(
        name="mysql_example",
        db_type=DatabaseType.MYSQL,
        host="localhost",
        port=3306,
        database="mydb",
        username="root",
        password="password",
    ))

    await db.connect()
    print(f"✓ Connected to MySQL: {db.config.database}")

    # Test query
    result = await db.execute("SHOW TABLES LIMIT 5")
    print(f"  Tables: {result}")

    return db


async def connect_sqlite():
    """
    Connect to SQLite

    Requirements:
    - pip install aiosqlite
    - No server needed - just a file!
    """
    from database.multi_db import SQLiteAdapter, ConnectionConfig, DatabaseType

    db = SQLiteAdapter(ConnectionConfig(
        name="sqlite_example",
        db_type=DatabaseType.SQLITE,
        database="./my_database.db",  # File path
        # database=":memory:",  # Or in-memory database
    ))

    await db.connect()
    print(f"✓ Connected to SQLite: {db.config.database}")

    # Test query
    result = await db.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        LIMIT 5
    """)
    print(f"  Tables: {[r['name'] for r in result]}")

    return db


# =============================================================================
# USING WITH METAGENT
# =============================================================================

async def use_with_agent(db):
    """
    Once connected, use any database with MetaAgent the same way.
    The agent auto-detects the dialect!
    """
    from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
    from intelligence.meta_agent import MetaAgent

    # Setup LLM
    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4o",
    ))

    # Create agent
    agent = MetaAgent(llm_client=llm)

    # Connect - agent auto-discovers everything!
    stats = await agent.connect(db_executor=db.execute)

    print(f"\n  Agent connected!")
    print(f"  • Detected dialect: {stats['dialect']}")
    print(f"  • Discovered tables: {stats['tables']}")
    print(f"  • Schema insights: {stats['schema_insights']}")

    # Query works the same regardless of database type
    result = await agent.query("How many tables are there?")
    if result["success"]:
        print(f"\n  Query result: {result['data']}")
        print(f"  SQL used: {result['sql']}")

    return agent


# =============================================================================
# MAIN
# =============================================================================

async def main():
    print("=" * 60)
    print("  TUTORIAL 2: DATABASE CONNECTIONS")
    print("=" * 60)

    print("\nThis tutorial shows connection examples for each database.")
    print("Uncomment the database you want to test.\n")

    # Uncomment ONE of these to test:

    # --- MS SQL Server ---
    # db = await connect_mssql()
    # agent = await use_with_agent(db)

    # --- PostgreSQL ---
    # db = await connect_postgresql()
    # agent = await use_with_agent(db)

    # --- MySQL ---
    # db = await connect_mysql()
    # agent = await use_with_agent(db)

    # --- SQLite (easiest to test!) ---
    print("Testing SQLite (no server required)...")

    # Create a test database
    from database.multi_db import SQLiteAdapter, ConnectionConfig, DatabaseType

    db = SQLiteAdapter(ConnectionConfig(
        name="test_db",
        db_type=DatabaseType.SQLITE,
        database=":memory:",  # In-memory for testing
    ))
    await db.connect()

    # Create sample table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute("INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com')")
    await db.execute("INSERT INTO users (name, email) VALUES ('Bob', 'bob@example.com')")
    await db.execute("INSERT INTO users (name, email) VALUES ('Charlie', 'charlie@example.com')")

    print("✓ Created test SQLite database with 'users' table")

    # Now use with agent (if you have LLM configured)
    if os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"):
        agent = await use_with_agent(db)

        # Test some queries
        print("\n--- Testing Queries ---")

        queries = [
            "How many users are there?",
            "Show all user names",
            "Who was the first user created?",
        ]

        for q in queries:
            print(f"\nQ: {q}")
            result = await agent.query(q)
            if result["success"]:
                print(f"A: {result['data']}")
            else:
                print(f"Error: {result['error']}")
    else:
        print("\n⚠ No LLM API key found. Set AZURE_OPENAI_API_KEY or OPENAI_API_KEY")

    print("\n" + "=" * 60)
    print("  TUTORIAL COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
