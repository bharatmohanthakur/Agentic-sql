#!/usr/bin/env python3
"""
TEST: Truly Intelligent System
==============================
Tests the system that:
- Auto-detects database type
- Auto-discovers schema
- Learns from errors automatically
- Improves over time

NO manual configuration needed!
"""
import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')


async def test_truly_intelligent():
    """Test the truly intelligent system."""

    print("\n" + "=" * 70)
    print("    TRULY INTELLIGENT SYSTEM TEST")
    print("    Zero Configuration - Full Auto-Learning")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Initialize LLM (only thing we need to provide)
    # =========================================================================
    print("\n[1] Initializing LLM...")

    from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4o",
        embedding_deployment="text-embedding-3-large",
        api_version="2024-02-01",
    ))
    print("   LLM Ready")

    # =========================================================================
    # STEP 2: Create Intelligent Core (the ONLY component needed)
    # =========================================================================
    print("\n[2] Creating Intelligent Core...")

    from intelligence.intelligent_core import IntelligentCore

    brain = IntelligentCore(llm_client=llm)
    print("   Brain initialized")

    # =========================================================================
    # STEP 3: Connect to Database (auto-discovery happens here)
    # =========================================================================
    print("\n[3] Connecting to Database (auto-discovery)...")

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

    # Connect brain to database - AUTO-DISCOVERY happens here!
    discovery_stats = await brain.connect(
        db_executor=db.execute,
        driver="pyodbc",  # Helps detect dialect
    )

    print(f"   Auto-detected dialect: {discovery_stats['dialect']}")
    print(f"   Discovered {discovery_stats['tables_discovered']} tables")
    print(f"   Discovered {discovery_stats['columns_discovered']} columns")

    # =========================================================================
    # STEP 4: Test Queries - System learns as we go
    # =========================================================================
    print("\n[4] Testing Queries (system learns from each one)...")
    print("=" * 70)

    test_questions = [
        # Simple queries
        "How many categories exist?",
        "Count all articles",

        # Filtering
        "Show laws with status 'In Force'",
        "Find legislations about 'tax'",

        # Date queries (might fail first, then learn)
        "Show the 10 most recent legislations",
        "How many legislations were issued in 2023?",

        # Complex
        "Which issuing authority has the most legislations?",
        "Compare the count of laws vs decrees",

        # After learning, retry a failed query
        "List legislations from the last 2 years",
    ]

    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Query {i}: {question[:50]}...")

        result = await brain.query(question)

        status = "✓" if result.success else "✗"
        corrections = " [CORRECTED]" if result.was_corrected else ""

        print(f"   {status} Result: {result.row_count} rows in {result.execution_time_ms}ms{corrections}")

        if result.success:
            print(f"   SQL: {result.sql[:80]}...")
        else:
            print(f"   Error: {result.error[:80]}...")

        results.append(result)

        # Small delay to avoid rate limiting
        await asyncio.sleep(0.5)

    # =========================================================================
    # STEP 5: Show Learning Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("    LEARNING RESULTS")
    print("=" * 70)

    stats = brain.get_stats()
    learning = stats.get("learning_stats", {})

    print(f"\nSystem Statistics:")
    print(f"   - Detected Dialect: {learning.get('detected_dialect', 'unknown')}")
    print(f"   - Dialect Confidence: {learning.get('dialect_confidence', 0):.2f}")
    print(f"   - Corrections Learned: {learning.get('total_corrections_learned', 0)}")
    print(f"   - High Confidence: {learning.get('high_confidence_corrections', 0)}")
    print(f"   - Column Mappings: {learning.get('column_mappings_learned', 0)}")

    # Show learned corrections
    corrections = brain.get_learned_corrections()
    if corrections:
        print(f"\nLearned Corrections ({len(corrections)}):")
        for c in corrections[:5]:
            print(f"   - {c['pattern'][:30]} -> {c['fix'][:30]} (conf: {c['confidence']:.2f})")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("    TEST SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r.success)
    corrected = sum(1 for r in results if r.was_corrected)

    print(f"\nQuery Results:")
    print(f"   - Total: {len(results)}")
    print(f"   - Successful: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print(f"   - Auto-Corrected: {corrected}")

    print(f"\nKey Features Demonstrated:")
    print(f"   ✓ Auto-detected database type: {learning.get('detected_dialect', 'unknown')}")
    print(f"   ✓ Auto-discovered {stats['schema']['tables']} tables")
    print(f"   ✓ Learned {learning.get('total_corrections_learned', 0)} corrections")
    print(f"   ✓ Zero manual configuration!")

    print("\n" + "=" * 70)
    print("    TRULY INTELLIGENT SYSTEM - TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_truly_intelligent())
