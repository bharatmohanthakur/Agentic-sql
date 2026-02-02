#!/usr/bin/env python3
"""
STRESS TEST - Push the system until it breaks
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')

# STRESS TEST QUERIES - designed to break the system
STRESS_QUERIES = [
    # LEVEL 1: Complex JOINs
    "Show all articles with their legislation type and issuing authority",
    "List legislations that have both articles AND amendments",

    # LEVEL 2: Multiple aggregations
    "What is the average number of articles per legislation type, ordered by count?",
    "Show the top 3 categories by total article count with percentage of total",

    # LEVEL 3: Subqueries
    "Find legislations that have more articles than the average",
    "Which issuing authorities have legislations in more than 2 categories?",

    # LEVEL 4: Date complexity
    "Show monthly trend of legislations issued in the last 3 years",
    "Which year had the most legislation activity and what types?",

    # LEVEL 5: Ambiguous/Vague
    "What is the most important legislation?",
    "Show me everything about category 5",
    "How active is the system?",

    # LEVEL 6: Non-existent concepts
    "Show all users and their permissions",
    "List the audit log entries",
    "What are the workflow statuses?",

    # LEVEL 7: Very complex
    "Create a report showing: legislation count by type, average articles per type, and the most recent legislation for each type",
    "Compare Q1 vs Q2 legislation counts for 2023 by category",

    # LEVEL 8: Calculations
    "What percentage of legislations have no articles?",
    "Calculate the growth rate of legislations year over year",

    # LEVEL 9: Edge cases
    "Show legislations where the title contains special characters",
    "Find duplicate legislation titles if any exist",
    "List all NULL values in the main legislation table",

    # LEVEL 10: Nonsense/Typos
    "Shwo teh legislatoins form 2023",
    "categroies with most legilsations",
]


async def stress_test():
    print("=" * 70)
    print("    STRESS TEST - BREAKING THE SYSTEM")
    print("=" * 70)

    from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
    from database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4o",
        api_version="2024-02-01",
    ))

    from intelligence.meta_agent import MetaAgent
    agent = MetaAgent(llm_client=llm)

    db = MSSQLAdapter(ConnectionConfig(
        name="legislation_db",
        db_type=DatabaseType.MSSQL,
        host="csptecdbserver.database.windows.net,1433",
        database="TEC.Datalake.PreProduction",
        username="cspsb",
        password="Csp00123@@@#$!@#",
    ))
    await db.connect()
    await agent.connect(db_executor=db.execute)

    results = []
    for i, q in enumerate(STRESS_QUERIES, 1):
        print(f"\n[{i:02d}] {q[:55]}...")

        try:
            result = await agent.query(q)
            status = "✓" if result["success"] else "✗"
            rows = result.get("row_count", 0)
            iters = result.get("iterations", 0)
            print(f"     {status} {rows} rows | {iters} iterations")

            if not result["success"]:
                print(f"     ERROR: {result.get('error', '')[:60]}...")

            results.append({
                "query": q,
                "success": result["success"],
                "rows": rows,
            })
        except Exception as e:
            print(f"     ✗ EXCEPTION: {str(e)[:60]}")
            results.append({"query": q, "success": False, "error": str(e)})

        await asyncio.sleep(0.3)

    # Summary
    print("\n" + "=" * 70)
    print("    STRESS TEST RESULTS")
    print("=" * 70)

    success = sum(1 for r in results if r["success"])
    failed = [r for r in results if not r["success"]]

    print(f"\nTotal: {success}/{len(results)} ({100*success/len(results):.1f}%)")

    if failed:
        print(f"\nFAILED QUERIES ({len(failed)}):")
        for f in failed:
            print(f"  - {f['query'][:50]}...")


if __name__ == "__main__":
    asyncio.run(stress_test())
