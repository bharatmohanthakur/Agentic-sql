#!/usr/bin/env python3
"""
TEST: META-LEARNING AGENT
=========================
Tests true meta-learning:
- THINK: Problem analysis
- RESEARCH: Find what worked
- DESIGN: Custom approach
- EXECUTE: Dynamic actions
- LEARN: Update meta-knowledge
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')


async def test_meta_agent():
    print("\n" + "=" * 70)
    print("    META-LEARNING AGENT TEST")
    print("    THINK → RESEARCH → DESIGN → EXECUTE → LEARN")
    print("=" * 70)

    # Initialize
    print("\n[1] Initializing...")
    from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
    from database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4o",
        api_version="2024-02-01",
    ))

    # Create meta-agent
    print("\n[2] Creating Meta-Learning Agent...")
    from intelligence.meta_agent import MetaAgent
    agent = MetaAgent(llm_client=llm)

    # Connect
    print("\n[3] Connecting...")
    db = MSSQLAdapter(ConnectionConfig(
        name="legislation_db",
        db_type=DatabaseType.MSSQL,
        host="csptecdbserver.database.windows.net,1433",
        database="TEC.Datalake.PreProduction",
        username="cspsb",
        password="Csp00123@@@#$!@#",
    ))
    await db.connect()
    stats = await agent.connect(db_executor=db.execute)
    print(f"   Tables: {stats['tables']}")

    # Test queries
    print("\n[4] Testing Meta-Learning...")
    print("=" * 70)

    questions = [
        "How many categories exist?",  # count type
        "Show the 5 most recent legislations",  # filter + date type
        "Which issuing authority has the most legislations?",  # aggregate type
        "How many legislations were issued in 2023?",  # date type
        "List legislations from the last 2 years",  # date type
    ]

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n--- Query {i}: {q[:50]}...")

        result = await agent.query(q)

        status = "SUCCESS" if result["success"] else "FAILED"
        print(f"   {status}: {result['row_count']} rows in {result['execution_time_ms']}ms")
        print(f"   Problem Type: {result.get('problem_type', 'unknown')}")
        print(f"   Steps: {result.get('steps_taken', 0)}")
        print(f"   Iterations: {result.get('iterations', 1)}")

        if result["success"]:
            print(f"   SQL: {result['sql'][:60]}...")

        results.append(result)
        await asyncio.sleep(0.5)

    # Summary
    print("\n" + "=" * 70)
    print("    META-LEARNING SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r["success"])
    stats = agent.get_stats()

    print(f"\nResults: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print(f"\nMeta-Knowledge Acquired:")
    print(f"   - Problem Types: {stats['problem_types_learned']}")
    print(f"   - Actions Learned: {stats['actions_learned']}")
    print(f"   - Solutions Stored: {stats['solutions_stored']}")

    print(f"\nWhat Makes This Different:")
    print(f"   - NO fixed prompts - prompts DESIGNED per problem")
    print(f"   - NO fixed actions - actions SELECTED dynamically")
    print(f"   - LEARNS from each query")
    print(f"   - RESEARCHES what worked before")
    print(f"   - DESIGNS custom approach")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(test_meta_agent())
