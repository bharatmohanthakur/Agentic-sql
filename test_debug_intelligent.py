#!/usr/bin/env python3
"""
Debug: Test IntelligentCore to see schema context and SQL generation
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')


async def test_debug():
    """Debug IntelligentCore."""
    print("=" * 60)
    print("INTELLIGENT CORE DEBUG")
    print("=" * 60)

    # Initialize LLM
    print("\n[1] Initializing LLM...")
    from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4o",
        embedding_deployment="text-embedding-3-large",
        api_version="2024-02-01",
    ))

    # Create brain
    print("\n[2] Creating IntelligentCore...")
    from intelligence.intelligent_core import IntelligentCore

    brain = IntelligentCore(llm_client=llm)

    # Connect to database
    print("\n[3] Connecting to database...")
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

    stats = await brain.connect(
        db_executor=db.execute,
        driver="pyodbc",
    )
    print(f"   Discovered {stats['tables_discovered']} tables, {stats['columns_discovered']} columns")

    # Show the schema context
    print("\n[4] Schema Context being sent to LLM:")
    print("-" * 60)
    print(brain._schema_context)
    print("-" * 60)
    print(f"Schema context length: {len(brain._schema_context)} chars")

    # Test one query
    print("\n[5] Testing single query...")
    question = "How many categories exist?"
    print(f"Question: {question}")

    # Test SQL generation manually
    prompt = f"""{brain._schema_context}

QUESTION: {question}

Write a valid MSSQL query to answer this question.
IMPORTANT:
- Use ONLY tables and columns from the schema above
- Use dbo.TableName format (e.g., dbo.Legislations)
- Use TOP N not LIMIT
- Return ONLY the raw SQL query, nothing else
- NO markdown, NO explanations, NO comments

SQL:"""

    print(f"\n[6] Full prompt length: {len(prompt)} chars")
    print("\n[7] Generating SQL...")

    response = await llm.generate(prompt=prompt, max_tokens=200)
    print(f"Raw response:\n---\n{response}\n---")

    # Now test with brain.query - with debugging
    print("\n[8] Testing brain.query()...")

    # First, test what apply_learned_corrections does
    test_sql = "SELECT COUNT(*) FROM dbo.Category;"
    print(f"Original SQL: {test_sql}")

    corrected1 = brain.adaptive_learning.apply_learned_corrections(test_sql)
    print(f"After apply_learned_corrections: {corrected1}")

    corrected2 = brain.adaptive_learning.apply_column_mappings(corrected1)
    print(f"After apply_column_mappings: {corrected2}")

    # Show all column mappings
    print(f"\nNumber of column mappings: {len(brain.adaptive_learning._column_mappings)}")
    for key, mapping in list(brain.adaptive_learning._column_mappings.items())[:10]:
        print(f"  {key}: {mapping.user_term} -> {mapping.actual_column} (conf: {mapping.confidence})")

    # Show all learned corrections
    print(f"\nNumber of learned corrections: {len(brain.adaptive_learning._corrections)}")
    for corr in brain.adaptive_learning._corrections[:5]:
        print(f"  pattern: {corr.original_sql_pattern} -> {corr.correction_pattern}")

    # Now actually query
    result = await brain.query(question)
    print(f"\nResult: success={result.success}, sql={result.sql}")
    if result.error:
        print(f"Error: {result.error}")

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")


if __name__ == "__main__":
    asyncio.run(test_debug())
