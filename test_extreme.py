#!/usr/bin/env python3
"""
EXTREME STRESS TEST - Really break the system
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')

# EXTREME QUERIES - designed to REALLY break the system
EXTREME_QUERIES = [
    # Window functions
    "Show running total of legislations by date with row number",
    "Rank issuing authorities by legislation count using DENSE_RANK",

    # Recursive/CTE
    "Show category hierarchy with parent-child relationships",
    "Find all subcategories of category 1 recursively",

    # Multiple CTEs
    "Using CTEs, show: 1) total articles, 2) total legislations, 3) ratio between them",

    # PIVOT/Dynamic
    "Create a pivot table showing legislation count by type and year",

    # String manipulation
    "Extract the year from legislation titles that contain years",
    "Split the Issuing_Authority into words and count unique words",

    # Math calculations
    "Calculate standard deviation of article counts per legislation",
    "Find the median number of articles per legislation type",

    # Temporal patterns
    "Find legislations issued on the same day of the week as today",
    "Show the busiest month historically for issuing legislations",

    # Self joins
    "Find pairs of legislations issued on the same date",
    "Show legislations that share the same issuing authority and type",

    # NOT EXISTS / Anti-patterns
    "Find issuing authorities that have never issued a Decree",
    "Show categories with no legislations at all",

    # UNION complex
    "Combine a list of all legislation types with a list of all categories",

    # Really vague
    "What's wrong with our data?",
    "Find anomalies",
    "Summarize everything",

    # Multi-language
    "عرض التشريعات باللغة العربية",  # Arabic: Show legislations in Arabic

    # Very long question
    "I need a comprehensive analysis that shows me all the legislations grouped by their type, then for each type show the count, the earliest and latest issue date, the total number of articles, and the most common issuing authority, sorted by total count descending",

    # Impossible requests
    "Predict how many legislations will be issued next year",
    "Show me legislations that don't exist yet",

    # SQL injection style (should handle safely)
    "Show legislations; DROP TABLE Legislations; --",
    "Select * from Legislations where 1=1 OR 1=1",
]


async def extreme_test():
    print("=" * 70)
    print("    EXTREME STRESS TEST - REALLY BREAKING IT")
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
    for i, q in enumerate(EXTREME_QUERIES, 1):
        display_q = q[:50].replace('\n', ' ')
        print(f"\n[{i:02d}] {display_q}...")

        try:
            result = await agent.query(q)
            status = "✓" if result["success"] else "✗"
            rows = result.get("row_count", 0)
            iters = result.get("iterations", 0)
            time_ms = result.get("execution_time_ms", 0)
            print(f"     {status} {rows} rows | {iters} iters | {time_ms}ms")

            if not result["success"]:
                print(f"     ERROR: {result.get('error', '')[:50]}...")
            elif result["success"] and rows > 0:
                print(f"     SQL: {result.get('sql', '')[:60]}...")

            results.append({
                "query": q,
                "success": result["success"],
                "rows": rows,
            })
        except Exception as e:
            print(f"     ✗ EXCEPTION: {str(e)[:50]}")
            results.append({"query": q, "success": False, "error": str(e)})

        await asyncio.sleep(0.3)

    # Summary
    print("\n" + "=" * 70)
    print("    EXTREME TEST RESULTS")
    print("=" * 70)

    success = sum(1 for r in results if r["success"])
    failed = [r for r in results if not r["success"]]

    print(f"\nTotal: {success}/{len(results)} ({100*success/len(results):.1f}%)")

    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for f in failed:
            print(f"  ✗ {f['query'][:45]}...")

    succeeded = [r for r in results if r["success"]]
    if succeeded:
        print(f"\nSUCCEEDED ({len(succeeded)}):")
        for s in succeeded:
            print(f"  ✓ {s['query'][:45]}... ({s['rows']} rows)")


if __name__ == "__main__":
    asyncio.run(extreme_test())
