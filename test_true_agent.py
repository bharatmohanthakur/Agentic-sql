#!/usr/bin/env python3
"""
TEST: TRUE AGENTIC SYSTEM
=========================
Tests the full agentic patterns:
- Chain of Thought
- Tree of Thoughts
- ReAct (Act-Observe-Iterate)
- Self-Reflection
- Planning
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')


async def test_true_agent():
    print("\n" + "=" * 70)
    print("    TRUE AGENTIC SYSTEM TEST")
    print("    Chain of Thought | Tree of Thoughts | ReAct")
    print("=" * 70)

    # Initialize
    print("\n[1] Initializing LLM...")
    from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4o",
        api_version="2024-02-01",
    ))

    # Create agent
    print("\n[2] Creating True Agent...")
    from intelligence.true_agent import TrueAgent
    agent = TrueAgent(llm_client=llm)

    # Connect
    print("\n[3] Connecting to Database...")
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

    stats = await agent.connect(db_executor=db.execute, driver="pyodbc")
    print(f"   Dialect: {stats['dialect']}")
    print(f"   Tables: {stats['tables']}")

    # Test queries
    print("\n[4] Testing with Full Agentic Reasoning...")
    print("=" * 70)

    questions = [
        "How many categories exist?",
        "Show the 5 most recent legislations",
        "Which issuing authority has the most legislations?",
        "How many legislations were issued in 2023?",
        "List legislations from the last 2 years",
    ]

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n--- Query {i}: {q[:50]}...")

        result = await agent.query(q)

        status = "SUCCESS" if result["success"] else "FAILED"
        print(f"   {status}: {result['row_count']} rows")
        print(f"   Reasoning Steps: {result.get('reasoning_steps', 0)}")
        print(f"   SQL Options Explored: {result.get('sql_options_explored', 0)}")
        print(f"   ReAct Iterations: {result.get('iterations', 0)}")
        print(f"   Time: {result.get('execution_time_ms', 0)}ms")

        if result["success"]:
            print(f"   SQL: {result['sql'][:70]}...")

        results.append(result)
        await asyncio.sleep(0.5)

    # Summary
    print("\n" + "=" * 70)
    print("    TEST SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r["success"])
    total_thoughts = sum(r.get("reasoning_steps", 0) for r in results)
    total_options = sum(r.get("sql_options_explored", 0) for r in results)

    print(f"\nResults: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print(f"Total Reasoning Steps: {total_thoughts}")
    print(f"Total SQL Options Explored: {total_options}")

    print(f"\nAgentic Patterns Used:")
    print(f"   - Chain of Thought: Step-by-step reasoning")
    print(f"   - Tree of Thoughts: Multiple SQL options generated")
    print(f"   - ReAct: Act-Observe-Iterate loop")
    print(f"   - Self-Reflection: Evaluated and selected best option")
    print(f"   - Planning: Created plan before execution")

    print("\n" + "=" * 70)
    print("    TRUE AGENTIC TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_true_agent())
