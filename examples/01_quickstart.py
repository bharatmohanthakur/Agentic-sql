#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 1: QUICKSTART - Your First Query in 5 Minutes
=============================================================================

This tutorial shows the simplest way to get started with Agentic SQL.
By the end, you'll have a working text-to-SQL system.

Prerequisites:
- Python 3.10+
- An LLM API key (OpenAI, Azure, or Anthropic)
- A database to connect to

Run: python examples/01_quickstart.py
"""

import asyncio
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


async def main():
    """
    Step-by-step quickstart tutorial.
    """
    print("=" * 60)
    print("  AGENTIC SQL - QUICKSTART TUTORIAL")
    print("=" * 60)

    # =========================================================================
    # STEP 1: Setup your LLM Client
    # =========================================================================
    print("\n[Step 1] Setting up LLM client...")

    # Option A: Azure OpenAI (recommended for enterprise)
    from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4o",  # or your deployment name
        api_version="2024-02-01",
    ))

    # Option B: OpenAI (uncomment to use)
    # from llm.openai_client import OpenAIClient, OpenAIConfig
    # llm = OpenAIClient(OpenAIConfig(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     model="gpt-4",
    # ))

    # Option C: Anthropic Claude (uncomment to use)
    # from llm.anthropic_client import AnthropicClient, AnthropicConfig
    # llm = AnthropicClient(AnthropicConfig(
    #     api_key=os.getenv("ANTHROPIC_API_KEY"),
    #     model="claude-3-opus-20240229",
    # ))

    # Option D: AWS Bedrock (uncomment to use)
    # from llm.bedrock_client import BedrockClient, BedrockConfig
    # llm = BedrockClient(BedrockConfig(
    #     region_name="us-east-1",
    #     model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    #     # Uses IAM role by default, or set explicit credentials:
    #     # aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    #     # aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    # ))

    print("  ✓ LLM client ready")

    # =========================================================================
    # STEP 2: Create the MetaAgent
    # =========================================================================
    print("\n[Step 2] Creating MetaAgent...")

    from intelligence.meta_agent import MetaAgent

    agent = MetaAgent(llm_client=llm)
    print("  ✓ MetaAgent created")

    # =========================================================================
    # STEP 3: Connect to your Database
    # =========================================================================
    print("\n[Step 3] Connecting to database...")

    from database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

    # Configure your database connection
    db = MSSQLAdapter(ConnectionConfig(
        name="my_database",
        db_type=DatabaseType.MSSQL,
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "MyDatabase"),
        username=os.getenv("DB_USER", "sa"),
        password=os.getenv("DB_PASSWORD", "password"),
    ))

    await db.connect()
    print(f"  ✓ Connected to database: {db.config.database}")

    # =========================================================================
    # STEP 4: Connect Agent to Database (Auto-Discovery)
    # =========================================================================
    print("\n[Step 4] Connecting agent (auto-discovery)...")

    stats = await agent.connect(db_executor=db.execute)
    print(f"  ✓ Discovered {stats['tables']} tables")
    print(f"  ✓ Detected dialect: {stats['dialect']}")
    print(f"  ✓ Schema insights: {stats['schema_insights']}")

    # =========================================================================
    # STEP 5: Ask Questions!
    # =========================================================================
    print("\n[Step 5] Asking questions...")
    print("-" * 60)

    # Example questions - modify these for your database
    questions = [
        "How many tables are in the database?",
        "Show me the first 5 records from any table",
        "What are the column names in the main table?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n  Q{i}: {question}")

        result = await agent.query(question)

        if result["success"]:
            print(f"  ✓ Success! ({result['row_count']} rows)")
            print(f"  SQL: {result['sql'][:80]}...")
            if result["data"]:
                print(f"  Sample: {result['data'][0]}")
        else:
            print(f"  ✗ Failed: {result['error'][:50]}...")

    # =========================================================================
    # STEP 6: Check What the Agent Learned
    # =========================================================================
    print("\n" + "-" * 60)
    print("[Step 6] Agent Statistics:")

    stats = agent.get_stats()
    print(f"  • Dialect: {stats['dialect']}")
    print(f"  • Tables discovered: {stats['tables']}")
    print(f"  • Solutions stored: {stats['solutions_stored']}")
    print(f"  • Schema insights: {stats['schema_insights']}")

    print("\n" + "=" * 60)
    print("  QUICKSTART COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  • Run 02_databases.py to learn about different databases")
    print("  • Run 03_auto_learning.py to train the agent automatically")
    print("  • Run 04_memory_system.py to add persistent memory")


if __name__ == "__main__":
    asyncio.run(main())
