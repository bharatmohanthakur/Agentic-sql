#!/usr/bin/env python3
"""
TEST: Agentic Brain - TRUE LLM Intelligence
============================================
NO REGEX. NO RULES. PURE LLM DECISION MAKING.

The LLM:
- Detects the database type
- Learns from successes and failures
- Fixes errors intelligently
- Improves over time
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')


async def test_agentic_brain():
    """Test the truly agentic LLM-driven brain."""

    print("\n" + "=" * 70)
    print("    AGENTIC BRAIN TEST")
    print("    Pure LLM Intelligence - No Rules, Only Learning")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Initialize LLM
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
    # STEP 2: Create Agentic Brain
    # =========================================================================
    print("\n[2] Creating Agentic Brain...")

    from intelligence.agentic_brain import AgenticBrain

    brain = AgenticBrain(llm_client=llm)
    print("   Brain initialized - LLM is in control")

    # =========================================================================
    # STEP 3: Connect to Database (LLM discovers everything)
    # =========================================================================
    print("\n[3] Connecting to Database (LLM discovers)...")

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

    # LLM discovers dialect and schema
    stats = await brain.connect(
        db_executor=db.execute,
        driver="pyodbc",
    )

    print(f"   LLM detected: {stats['dialect']}")
    print(f"   LLM discovered: {stats['tables_discovered']} tables")
    print(f"   LLM discovered: {stats['columns_discovered']} columns")

    # =========================================================================
    # STEP 4: Test Queries - LLM learns from each one
    # =========================================================================
    print("\n[4] Testing Queries (LLM learns from each)...")
    print("=" * 70)

    test_questions = [
        # Simple queries
        "How many categories exist?",
        "Count all articles",

        # Filtering
        "Show laws with status 'In Force'",
        "Find legislations about 'tax'",

        # Date queries
        "Show the 10 most recent legislations",
        "How many legislations were issued in 2023?",

        # Complex
        "Which issuing authority has the most legislations?",
        "Compare the count of laws vs decrees",

        # After learning
        "List legislations from the last 2 years",
    ]

    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Query {i}: {question[:50]}...")

        result = await brain.query(question)

        status = "SUCCESS" if result["success"] else "FAILED"
        corrected = " [LLM CORRECTED]" if result.get("was_corrected") else ""

        print(f"   {status}: {result['row_count']} rows in {result['execution_time_ms']}ms{corrected}")

        if result["success"]:
            print(f"   SQL: {result['sql'][:80]}...")
        else:
            print(f"   Error: {result.get('error', '')[:80]}...")

        results.append(result)

        # Small delay
        await asyncio.sleep(0.5)

    # =========================================================================
    # STEP 5: Show What LLM Learned
    # =========================================================================
    print("\n" + "=" * 70)
    print("    WHAT THE LLM LEARNED")
    print("=" * 70)

    brain_stats = brain.get_stats()

    print(f"\nBrain Statistics:")
    print(f"   - Dialect Detected: {brain_stats['dialect']}")
    print(f"   - Queries Run: {brain_stats['queries_run']}")
    print(f"   - Success Rate: {brain_stats['success_rate']*100:.1f}%")
    print(f"   - LLM Corrections Made: {brain_stats['corrections_made']}")
    print(f"   - Patterns Learned: {brain_stats['patterns_learned']}")
    print(f"   - Error Fixes Learned: {brain_stats['error_fixes_learned']}")

    # Show learned patterns
    if brain.memory.successful_patterns:
        print(f"\nSuccessful Patterns Learned:")
        for p in brain.memory.successful_patterns[-3:]:
            print(f"   Q: {p['question'][:40]}...")
            print(f"   SQL: {p['sql'][:60]}...")

    if brain.memory.schema_insights:
        print(f"\nSchema Insights:")
        for insight in brain.memory.schema_insights[:3]:
            print(f"   - {insight}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("    TEST SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r["success"])
    corrected = sum(1 for r in results if r.get("was_corrected"))

    print(f"\nResults:")
    print(f"   - Total: {len(results)}")
    print(f"   - Successful: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print(f"   - LLM Auto-Corrected: {corrected}")

    print(f"\nKey Achievements:")
    print(f"   - LLM detected database: {brain_stats['dialect']}")
    print(f"   - LLM discovered schema: {brain_stats['tables_discovered']} tables")
    print(f"   - LLM learned {brain_stats['patterns_learned']} patterns")
    print(f"   - LLM made {brain_stats['corrections_made']} intelligent corrections")
    print(f"   - NO REGEX RULES - Pure LLM Intelligence!")

    print("\n" + "=" * 70)
    print("    AGENTIC BRAIN TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_agentic_brain())
