#!/usr/bin/env python3
"""
Debug: Test SQL generation to see what's being produced
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')


async def test_sql_gen():
    """Test SQL generation directly."""
    print("=" * 60)
    print("SQL GENERATION DEBUG TEST")
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
    print("   LLM Ready")

    # Simple schema context
    schema_context = """=== MSSQL DATABASE SCHEMA ===

SYNTAX RULES (MUST FOLLOW):
1. Use dbo.TableName format: FROM dbo.Legislations
2. Use TOP N not LIMIT: SELECT TOP 10 * FROM dbo.Table
3. Use GETDATE() not NOW()
4. Use YEAR(col) for date extraction

WORKING EXAMPLE QUERIES:
-- Count records: SELECT COUNT(*) FROM dbo.Category
-- Get top 10: SELECT TOP 10 * FROM dbo.Legislations ORDER BY Created_at DESC
-- Filter: SELECT * FROM dbo.Legislations WHERE Status = 'In Force'

TABLES AND COLUMNS:
dbo.Category: Category_Id, Category_Name_En, Category_Name_Ar
dbo.Legislations: Legislation_Id, Title_En, Title_Ar, Status, Created_at, Category_Id, Legislation_Type_Id
dbo.Articles: Article_Id, Article_Number, Text_En, Text_Ar, Legislation_Id
dbo.IssuingAuthorities: Authority_Id, Authority_Name_En, Authority_Name_Ar
dbo.LegislationType: Legislation_Type_Id, Type_Name_En, Type_Name_Ar"""

    # Test questions
    questions = [
        "How many categories exist?",
        "Count all articles",
        "Show the 5 most recent legislations",
    ]

    print("\n[2] Testing SQL generation...")
    print("-" * 60)

    for q in questions:
        print(f"\nQuestion: {q}")

        prompt = f"""{schema_context}

QUESTION: {q}

Write a valid MSSQL query to answer this question.
IMPORTANT:
- Use ONLY tables and columns from the schema above
- Use dbo.TableName format (e.g., dbo.Legislations)
- Use TOP N not LIMIT
- Return ONLY the raw SQL query, nothing else
- NO markdown, NO explanations, NO comments

SQL:"""

        response = await llm.generate(prompt=prompt, max_tokens=200)
        print(f"Generated SQL:\n---\n{response}\n---")

        # Clean up
        sql = response.strip()
        if "```" in sql:
            import re
            match = re.search(r"```(?:sql)?\s*(.*?)\s*```", sql, re.DOTALL)
            if match:
                sql = match.group(1).strip()

        print(f"Cleaned SQL: {sql}")

    print("\n" + "=" * 60)
    print("DEBUG TEST COMPLETE")


if __name__ == "__main__":
    asyncio.run(test_sql_gen())
