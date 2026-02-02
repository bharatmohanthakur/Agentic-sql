#!/usr/bin/env python3
"""
Test ONLY schema discovery to see what's happening
"""
import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')


async def test_discovery():
    """Test just the schema discovery part."""
    print("=" * 60)
    print("SCHEMA DISCOVERY DIAGNOSTIC TEST")
    print("=" * 60)

    # Connect to database
    print("\n[1] Connecting to database...")

    from database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

    db = MSSQLAdapter(ConnectionConfig(
        name="legislation_db",
        db_type=DatabaseType.MSSQL,
        host="csptecdbserver.database.windows.net,1433",
        database="TEC.Datalake.PreProduction",
        username="cspsb",
        password="Csp00123@@@#$!@#",
    ))
    await db.connect()
    print("   Database connected!")

    # Test basic query
    print("\n[2] Testing basic query...")
    result = await db.execute("SELECT TOP 1 1 AS test")
    print(f"   Test query result: {result}")

    # Test schema query
    print("\n[3] Discovering tables...")
    start = datetime.utcnow()
    tables = await db.execute("""
        SELECT TOP 20
            t.TABLE_SCHEMA,
            t.TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES t
        WHERE t.TABLE_TYPE = 'BASE TABLE'
        ORDER BY t.TABLE_NAME
    """)
    elapsed = (datetime.utcnow() - start).total_seconds()
    print(f"   Found {len(tables)} tables in {elapsed:.1f}s")

    for t in tables[:10]:
        print(f"   - {t.get('TABLE_SCHEMA', 'dbo')}.{t['TABLE_NAME']}")

    if len(tables) > 10:
        print(f"   ... and {len(tables) - 10} more")

    # Test with SchemaDiscovery
    print("\n[4] Testing SchemaDiscovery class...")
    from intelligence.auto_discovery import SchemaDiscovery, DatabaseDialect

    discovery = SchemaDiscovery(
        db_executor=db.execute,
        llm_client=None,  # Skip LLM for now
        dialect=DatabaseDialect.MSSQL,
    )

    print("   Running discovery (max 5 tables)...")
    start = datetime.utcnow()
    profiles = await discovery.discover(
        include_stats=False,  # Skip stats for speed
        include_samples=False,
        max_tables=5,
    )
    elapsed = (datetime.utcnow() - start).total_seconds()

    print(f"   Discovered {len(profiles)} tables in {elapsed:.1f}s")
    for table_name, profile in profiles.items():
        print(f"   - {table_name}: {len(profile.columns)} columns")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_discovery())
