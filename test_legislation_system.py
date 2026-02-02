#!/usr/bin/env python3
"""
Test the Intelligent Text-to-SQL System with UAE Legislation Database
Acting as a business user asking questions about legislation data
"""
import asyncio
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment from unified-legislation-agent-langgraph
from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')


class IntelligentLegislationSystem:
    """
    Wrapper for testing the Intelligent System with UAE Legislation Data
    """

    def __init__(self):
        self.system = None
        self.llm_client = None

    async def initialize(self):
        """Initialize the system with Azure OpenAI and MSSQL"""
        logger.info("=" * 60)
        logger.info("INTELLIGENT LEGISLATION QUERY SYSTEM")
        logger.info("=" * 60)

        # Import components
        from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
        from intelligence.intelligent_system import (
            IntelligentSystem,
            IntelligentSystemConfig,
        )
        from database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

        # Configure Azure OpenAI
        logger.info("\n[1] Configuring Azure OpenAI LLM...")
        llm_config = AzureOpenAIConfig(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
            embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            model="gpt-4o",
            max_tokens=2000,
            temperature=0.1,
        )
        self.llm_client = AzureOpenAIClient(llm_config)
        logger.info(f"   LLM: Azure OpenAI ({llm_config.azure_deployment})")
        logger.info(f"   Endpoint: {llm_config.azure_endpoint}")

        # Test LLM connection
        logger.info("   Testing LLM connection...")
        test_response = await self.llm_client.generate(
            prompt="Say 'Connection successful' in exactly 2 words.",
            max_tokens=10
        )
        logger.info(f"   LLM Test: {test_response.strip()}")

        # Configure system
        logger.info("\n[2] Configuring Intelligent System...")
        system_config = IntelligentSystemConfig(
            auto_discover_on_connect=False,  # We'll manually set schema
            enable_auto_learning=True,
            enable_user_learning=True,
            enable_business_learning=True,
            enable_research_agent=False,  # Disable for testing
            enable_self_healing=True,
            enable_deep_reasoning=True,
            max_healing_attempts=3,
        )

        # Embedding function
        def embedding_fn(text: str) -> List[float]:
            return self.llm_client.generate_embedding(text)

        # Create system
        self.system = IntelligentSystem(
            config=system_config,
            llm_client=self.llm_client,
            embedding_fn=embedding_fn,
        )

        # Connect to MSSQL
        logger.info("\n[3] Connecting to MS SQL Server (Legislation Database)...")
        db_config = ConnectionConfig(
            name="legislation_db",
            db_type=DatabaseType.MSSQL,
            host="csptecdbserver.database.windows.net,1433",
            database="TEC.Datalake.PreProduction",
            username="cspsb",
            password="Csp00123@@@#$!@#",
        )

        adapter = MSSQLAdapter(db_config)
        await adapter.connect()
        self.system._databases["legislation_db"] = adapter
        logger.info("   Connected successfully!")

        # Discover schema manually
        logger.info("\n[4] Discovering database schema...")
        schema = await adapter.get_schema()
        self.system._schema_profiles["legislation_db"] = {}

        # Find legislation views
        legislation_views = [
            t for t in schema.get("tables", [])
            if "legislation" in t.get("table_name", "").lower()
            or "vw_" in t.get("table_name", "").lower()
        ]

        logger.info(f"   Found {len(schema.get('tables', []))} tables/views")
        logger.info("   Key legislation views:")
        for view in legislation_views[:5]:
            logger.info(f"     - {view.get('table_name')}")

        # Set up schema context for the system
        schema_context = self._build_legislation_schema()
        self.schema_context = schema_context

        # Add business knowledge
        await self._add_business_knowledge()

        logger.info("\n[5] System Ready!")
        logger.info("=" * 60)

        return self

    def _build_legislation_schema(self) -> str:
        """Build schema context for legislation database"""
        return """
DATABASE SCHEMA - UAE Legislation Database

CRITICAL: Use table names directly WITHOUT any database prefix. Just use dbo.TableName format.

=== dbo.VW_LegislationInfo (Main Legislation View) ===
Primary view for all legislation information.
EXACT Columns (use these EXACTLY):
- Article_Id (int): Unique article identifier
- Legislation_Id (bigint): Legislation identifier
- Date_Of_Issuance (datetime): When legislation was issued
- Legislation_Title (nvarchar): Full title of legislation
- Article_Title (nvarchar): Title of specific article
- Article_Number (int): Article number
- Article_Body (nvarchar): Full text of article
- Category_Name (nvarchar): Category in English
- Category_Name_Arabic (nvarchar): Category in Arabic
- Sub_Category_Name (nvarchar): Sub-category in English
- Sub_Category_Name_Arabic (nvarchar): Sub-category in Arabic
- Authority_Name_En (nvarchar): Issuing authority in English
- Authority_Name_Ar (nvarchar): Issuing authority in Arabic
- Official_Gazette_Number (nvarchar): Gazette publication number
- SourceName_En (nvarchar): Source in English
- SourceName_Ar (nvarchar): Source in Arabic
- Status_Name_En (nvarchar): Status in English (In Force, Repealed, etc.)
- Status_Name_Ar (nvarchar): Status in Arabic
- Status_Name_Arabic (nvarchar): Status in Arabic (alternate)
- Language_Name_En (nvarchar): Language (Arabic, English)
- Legislation_Type_Name_En (nvarchar): Type in English (Law, Decree, Resolution)
- Legislation_Type_Name_Ar (nvarchar): Type in Arabic
- Source_File_Name (nvarchar): File name of source document
- Appendix_Content (nvarchar): Appendix content

=== IMPORTANT RULES ===
1. NEVER include database name in queries - use dbo.VW_LegislationInfo only
2. Use dbo.VW_LegislationInfo for ALL queries
3. Status values in Status_Name_En: 'In Force', 'Repealed', 'Amended'
4. Use TOP instead of LIMIT for result limiting (e.g., SELECT TOP 10)
5. Use DISTINCT for unique results
6. Format dates as: FORMAT(Date_Of_Issuance, 'dd-MMM-yyyy')
7. For text search: LOWER(column) LIKE LOWER(N'%text%')
8. AVOID selecting very long columns like Article_Body, Appendix_Content unless specifically asked
9. For legislation type queries, use Legislation_Type_Name_En
10. Only use columns that exist - refer to the exact list above
"""

    async def _add_business_knowledge(self):
        """Add legislation-specific business knowledge"""
        logger.info("\n[4b] Adding business knowledge...")

        # Add terminology
        if self.system.business_learning:
            self.system.business_learning.add_terminology(
                term="in force",
                definition="Legislation that is currently active and applicable",
                synonyms=["active", "current", "valid", "effective"]
            )
            self.system.business_learning.add_terminology(
                term="repealed",
                definition="Legislation that has been cancelled/revoked",
                synonyms=["cancelled", "revoked", "abolished"]
            )
            self.system.business_learning.add_terminology(
                term="amended",
                definition="Legislation that has been modified",
                synonyms=["modified", "changed", "updated"]
            )

        logger.info("   Added legislation terminology")

    async def ask(self, question: str, user_id: str = "business_user_1") -> Dict[str, Any]:
        """
        Ask a natural language question about legislation
        Returns formatted result
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"QUESTION: {question}")
        logger.info(f"{'='*60}")

        start_time = datetime.now()

        # Generate SQL using deep reasoning
        sql = await self._generate_sql(question)

        if not sql:
            return {
                "success": False,
                "question": question,
                "error": "Failed to generate SQL"
            }

        logger.info(f"\nGENERATED SQL:\n{sql}")

        # Execute query
        try:
            adapter = self.system._databases.get("legislation_db")
            if not adapter:
                return {"success": False, "error": "Database not connected"}

            results = await adapter.execute(sql)

            elapsed = (datetime.now() - start_time).total_seconds()

            # Format output
            logger.info(f"\nRESULTS ({len(results)} rows in {elapsed:.2f}s):")
            logger.info("-" * 60)

            if results:
                # Show first few results
                for i, row in enumerate(results[:5]):
                    logger.info(f"Row {i+1}:")
                    for key, value in row.items():
                        # Truncate long values
                        str_val = str(value)
                        if len(str_val) > 100:
                            str_val = str_val[:100] + "..."
                        logger.info(f"  {key}: {str_val}")
                    logger.info("")

                if len(results) > 5:
                    logger.info(f"... and {len(results) - 5} more rows")

            # Learn from success
            if self.system.user_learning:
                await self.system.user_learning.learn_from_interaction(
                    user_id=user_id,
                    question=question,
                    generated_sql=sql,
                    executed_sql=sql,
                    success=True,
                    result_data=results[:10],  # Sample for learning
                )

            return {
                "success": True,
                "question": question,
                "sql": sql,
                "data": results,
                "row_count": len(results),
                "time_seconds": elapsed,
            }

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {
                "success": False,
                "question": question,
                "sql": sql,
                "error": str(e)
            }

    async def _generate_sql(self, question: str) -> str:
        """Generate SQL from natural language question"""

        prompt = f"""You are an expert MS SQL query generator for UAE Legislation database.

{self.schema_context}

USER QUESTION: {question}

Generate a precise MS SQL query following these rules:
1. Use TOP instead of LIMIT
2. Use DISTINCT when appropriate
3. For text search, use: LOWER(column) LIKE LOWER(N'%value%')
4. Format dates as: FORMAT(date_col, 'dd-MMM-yyyy')
5. Use VW_LegislationInfo for general queries
6. Only output the SQL query, no explanation

SQL Query:"""

        response = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=500,
        )

        # Extract SQL from response
        sql = response.strip()

        # Clean up common issues
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]

        return sql.strip()

    async def interactive_session(self):
        """Run interactive question-answer session"""
        print("\n" + "=" * 60)
        print("INTERACTIVE LEGISLATION QUERY SESSION")
        print("=" * 60)
        print("Ask questions about UAE legislation in natural language.")
        print("Type 'quit' to exit, 'stats' for system statistics.")
        print("=" * 60 + "\n")

        while True:
            try:
                question = input("\nYour question: ").strip()

                if not question:
                    continue

                if question.lower() == 'quit':
                    print("Goodbye!")
                    break

                if question.lower() == 'stats':
                    stats = self.system.get_stats()
                    print("\nSystem Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                result = await self.ask(question)

                if not result.get("success"):
                    print(f"\nError: {result.get('error')}")

            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


async def run_business_tests():
    """Run sample business queries to test the system"""

    # Initialize system
    system = IntelligentLegislationSystem()
    await system.initialize()

    # Sample business questions
    business_questions = [
        # Basic queries
        "How many legislations are currently in force?",
        "List the top 10 most recent legislations",
        "What are the different types of legislation in the database?",

        # Category queries
        "Show me legislations related to sports",
        "How many legislations are there by category?",

        # Status queries
        "List legislations that have been repealed",
        "Show amended legislations from 2024",

        # Authority queries
        "Which authority has issued the most legislations?",

        # Arabic search
        "Find legislation about Dubai Sports Council",
    ]

    print("\n" + "=" * 60)
    print("RUNNING BUSINESS QUERY TESTS")
    print("=" * 60)

    results_summary = []

    for i, question in enumerate(business_questions, 1):
        print(f"\n[Test {i}/{len(business_questions)}]")
        result = await system.ask(question)

        results_summary.append({
            "question": question,
            "success": result.get("success"),
            "rows": result.get("row_count", 0),
            "time": result.get("time_seconds", 0),
        })

        # Small delay between queries
        await asyncio.sleep(1)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results_summary if r["success"])
    print(f"Total Tests: {len(results_summary)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results_summary) - successful}")

    print("\nDetailed Results:")
    for i, r in enumerate(results_summary, 1):
        status = "OK" if r["success"] else "FAIL"
        print(f"  {i}. [{status}] {r['question'][:50]}... ({r['rows']} rows, {r['time']:.2f}s)")

    # Get system stats
    print("\nSystem Statistics:")
    stats = system.system.get_stats()
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")

    return system


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Intelligent Legislation Query System")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run interactive session")
    parser.add_argument("--test", "-t", action="store_true",
                       help="Run business tests")
    parser.add_argument("--question", "-q", type=str,
                       help="Ask a single question")

    args = parser.parse_args()

    if args.question:
        system = IntelligentLegislationSystem()
        await system.initialize()
        await system.ask(args.question)

    elif args.interactive:
        system = IntelligentLegislationSystem()
        await system.initialize()
        await system.interactive_session()

    elif args.test:
        await run_business_tests()

    else:
        # Default: run business tests
        await run_business_tests()


if __name__ == "__main__":
    asyncio.run(main())
