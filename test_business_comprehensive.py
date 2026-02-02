#!/usr/bin/env python3
"""
COMPREHENSIVE BUSINESS USER TEST
=================================
Tests ALL types of query complexity a business user might need:

1. SIMPLE: Basic counts and selects
2. FILTERING: WHERE conditions
3. DATE: Time-based queries
4. AGGREGATION: GROUP BY, COUNT, SUM
5. COMPARISON: Compare values
6. RANKING: TOP N, ordering
7. COMPLEX: Joins, subqueries
8. AMBIGUOUS: Vague business questions
9. EDGE CASES: Bad data handling

This is the ULTIMATE test of system intelligence.
"""
import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')


# Test categories with business questions
TEST_CATEGORIES = {
    "SIMPLE": [
        "How many categories exist?",
        "Count all articles",
        "How many tables are in this database?",
    ],
    "FILTERING": [
        "Show all legislation types",
        "Find legislations with status 'In Force'",
        "List categories that have a parent",
    ],
    "DATE_QUERIES": [
        "Show the 10 most recent legislations",
        "How many legislations were issued in 2023?",
        "List legislations from the last 2 years",
        "What was the first legislation recorded?",
    ],
    "AGGREGATION": [
        "Which issuing authority has the most legislations?",
        "Count legislations by type",
        "How many articles per legislation on average?",
    ],
    "COMPARISON": [
        "Compare the count of laws vs decrees",
        "Which category has more legislations?",
    ],
    "RANKING": [
        "Top 5 issuing authorities by legislation count",
        "Show the 3 oldest legislations",
    ],
    "COMPLEX": [
        "Show legislations with their category names",
        "Find articles that belong to legislations from 2023",
        "List legislation types with their total article count",
    ],
    "BUSINESS_QUESTIONS": [
        "What types of legal documents do we have?",
        "How is our legislation data organized?",
        "What are the main sources of legislation?",
    ],
}


async def test_comprehensive():
    print("\n" + "=" * 70)
    print("    COMPREHENSIVE BUSINESS USER TEST")
    print("    Testing ALL Query Complexity Types")
    print("=" * 70)

    # Initialize
    print("\n[1] Initializing Meta-Learning Agent...")
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

    # Connect
    print("\n[2] Connecting to Database...")
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

    # Run tests by category
    print("\n[3] Running Comprehensive Tests...")
    print("=" * 70)

    all_results = {}
    category_stats = {}
    query_num = 0

    for category, questions in TEST_CATEGORIES.items():
        print(f"\n{'='*70}")
        print(f"  CATEGORY: {category}")
        print(f"{'='*70}")

        category_results = []

        for question in questions:
            query_num += 1
            print(f"\n[{query_num}] {question[:55]}...")

            try:
                result = await agent.query(question)

                status = "✓" if result["success"] else "✗"
                rows = result.get("row_count", 0)
                time_ms = result.get("execution_time_ms", 0)
                problem_type = result.get("problem_type", "unknown")

                print(f"    {status} {rows} rows | {time_ms}ms | Type: {problem_type}")

                if result["success"] and result.get("sql"):
                    print(f"    SQL: {result['sql'][:60]}...")

                category_results.append({
                    "question": question,
                    "success": result["success"],
                    "rows": rows,
                    "time_ms": time_ms,
                })

            except Exception as e:
                print(f"    ✗ ERROR: {str(e)[:50]}")
                category_results.append({
                    "question": question,
                    "success": False,
                    "error": str(e),
                })

            await asyncio.sleep(0.3)

        # Category stats
        success_count = sum(1 for r in category_results if r["success"])
        category_stats[category] = {
            "total": len(category_results),
            "success": success_count,
            "rate": success_count / len(category_results) if category_results else 0,
        }
        all_results[category] = category_results

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("    COMPREHENSIVE TEST RESULTS")
    print("=" * 70)

    print("\n┌─────────────────────┬───────┬─────────┬──────────┐")
    print("│ Category            │ Total │ Success │ Rate     │")
    print("├─────────────────────┼───────┼─────────┼──────────┤")

    total_queries = 0
    total_success = 0

    for category, stats in category_stats.items():
        rate_pct = stats["rate"] * 100
        status = "✓" if stats["rate"] >= 0.8 else "○" if stats["rate"] >= 0.5 else "✗"
        print(f"│ {category:<19} │ {stats['total']:>5} │ {stats['success']:>7} │ {rate_pct:>6.1f}% {status} │")
        total_queries += stats["total"]
        total_success += stats["success"]

    print("├─────────────────────┼───────┼─────────┼──────────┤")
    overall_rate = (total_success / total_queries * 100) if total_queries > 0 else 0
    print(f"│ {'TOTAL':<19} │ {total_queries:>5} │ {total_success:>7} │ {overall_rate:>6.1f}%   │")
    print("└─────────────────────┴───────┴─────────┴──────────┘")

    # Agent stats
    agent_stats = agent.get_stats()
    print(f"\nMeta-Learning Statistics:")
    print(f"   - Problem Types Learned: {agent_stats['problem_types_learned']}")
    print(f"   - Actions Learned: {agent_stats['actions_learned']}")
    print(f"   - Solutions Stored: {agent_stats['solutions_stored']}")

    # Failed queries
    failed = []
    for category, results in all_results.items():
        for r in results:
            if not r["success"]:
                failed.append(f"[{category}] {r['question'][:40]}")

    if failed:
        print(f"\nFailed Queries ({len(failed)}):")
        for f in failed[:5]:
            print(f"   - {f}")

    print("\n" + "=" * 70)
    print(f"    OVERALL SUCCESS: {overall_rate:.1f}%")
    print("=" * 70)

    return overall_rate


if __name__ == "__main__":
    asyncio.run(test_comprehensive())
