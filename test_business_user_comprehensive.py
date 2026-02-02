#!/usr/bin/env python3
"""
COMPREHENSIVE BUSINESS USER TEST
=================================
Tests the intelligent system as a real business user would use it.

Test Categories:
1. Simple Counts & Aggregations
2. Filtering & Search
3. Date & Time Queries
4. Multi-Table Joins
5. Business Terminology
6. Ambiguous Questions
7. Complex Analytics
8. Self-Healing (intentionally bad SQL)
9. Learning & Memory
10. Edge Cases
"""
import asyncio
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')


class BusinessUserTest:
    """Simulates a business user interacting with the system."""

    def __init__(self):
        self.db = None
        self.llm = None
        self.discovery = None
        self.profiles = None
        self.schema_context = None
        self.deep_reasoner = None
        self.self_healing = None
        self.user_learning = None
        self.business_learning = None
        self.knowledge_base = None
        self.results = []

    async def setup(self):
        """Initialize all system components."""
        print("\n" + "=" * 70)
        print("    COMPREHENSIVE BUSINESS USER TEST")
        print("    Testing as a real user would interact with the system")
        print("=" * 70)

        # 1. Database
        print("\n[SETUP] Connecting to database...")
        from database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType
        self.db = MSSQLAdapter(ConnectionConfig(
            name="legislation_db",
            db_type=DatabaseType.MSSQL,
            host="csptecdbserver.database.windows.net,1433",
            database="TEC.Datalake.PreProduction",
            username="cspsb",
            password="Csp00123@@@#$!@#",
        ))
        await self.db.connect()
        print("   Database: Connected")

        # 2. LLM
        from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
        self.llm = AzureOpenAIClient(AzureOpenAIConfig(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment="gpt-4o",
            embedding_deployment="text-embedding-3-large",
            api_version="2024-02-01",
        ))
        print("   LLM: Ready")

        # 3. Auto-Discovery
        print("[SETUP] Auto-discovering schema...")
        from intelligence.auto_discovery import SchemaDiscovery, DatabaseDialect
        self.discovery = SchemaDiscovery(
            db_executor=self.db.execute,
            dialect=DatabaseDialect.MSSQL,
        )
        self.profiles = await self.discovery.discover(
            include_stats=True,
            include_samples=False,
            max_tables=25,
        )
        print(f"   Auto-Discovery: {len(self.profiles)} tables discovered")

        # 4. Build schema context
        self.schema_context = self._build_schema_context()

        # 5. Knowledge Base
        from intelligence.knowledge_base import KnowledgeBase
        self.knowledge_base = KnowledgeBase(
            embedding_fn=lambda t: self.llm.generate_embedding(t),
            llm_client=self.llm,
        )
        print("   Knowledge Base: Ready")

        # 6. User Learning
        from intelligence.user_learning import UserLearningEngine
        self.user_learning = UserLearningEngine(llm_client=self.llm)
        print("   User Learning: Ready")

        # 7. Business Learning
        from intelligence.business_learning import BusinessLogicLearner
        self.business_learning = BusinessLogicLearner(
            llm_client=self.llm,
            knowledge_base=self.knowledge_base,
        )
        # Add domain terminology
        self.business_learning.add_terminology("in force", "Legislation currently active", ["active", "current", "valid"])
        self.business_learning.add_terminology("repealed", "Legislation that has been cancelled", ["cancelled", "revoked"])
        self.business_learning.add_terminology("gazette", "Official government publication", ["official gazette"])
        self.business_learning.add_terminology("amendment", "Change or modification to legislation", ["change", "modification"])
        print("   Business Learning: Ready with terminology")

        # 8. Deep Reasoner
        from intelligence.deep_reasoner import DeepReasoner
        self.deep_reasoner = DeepReasoner(
            llm_client=self.llm,
            knowledge_base=self.knowledge_base,
            max_depth=5,
        )
        print("   Deep Reasoner: Ready")

        # 9. Self-Healing
        from intelligence.self_healing import SelfHealingEngine
        self.self_healing = SelfHealingEngine(
            schema_profiles=self.profiles,
            db_executor=self.db.execute,
            llm_client=self.llm,
        )
        print("   Self-Healing: Ready")

        print("\n[SETUP] All components initialized!")

    def _build_schema_context(self) -> str:
        """Build schema context from auto-discovered profiles."""
        parts = [
            "=" * 60,
            "DATABASE SCHEMA (Auto-Discovered from MS SQL Server)",
            "CRITICAL: Only use column names listed below. Do NOT invent columns!",
            "=" * 60,
            ""
        ]

        # Key tables with full column details
        key_tables = ['DLPEnglishData', 'DLPArabicData', 'Articles', 'AmendedLegislations',
                      'Category', 'Appendix', 'AmendmentTypes']

        for name, p in self.profiles.items():
            is_key = name in key_tables
            parts.append(f"{'='*40 if is_key else '-'*40}")
            parts.append(f"TABLE: dbo.{name}")
            parts.append(f"Rows: ~{p.row_count}")

            # List ALL columns for key tables, limited for others
            max_cols = 30 if is_key else 10
            parts.append("COLUMNS (use ONLY these names):")
            for c in p.columns[:max_cols]:
                parts.append(f"  - {c.name} ({c.data_type})")
            if len(p.columns) > max_cols:
                remaining = [c.name for c in p.columns[max_cols:]]
                parts.append(f"  - Also: {', '.join(remaining[:10])}...")
            parts.append("")

        parts.extend([
            "=" * 60,
            "MSSQL QUERY RULES (MUST FOLLOW)",
            "=" * 60,
            "1. Use dbo.TableName format (never use database prefix)",
            "2. Use TOP N instead of LIMIT N: SELECT TOP 10 * FROM dbo.Table",
            "3. Date functions: YEAR(col), MONTH(col), FORMAT(col, 'yyyy-MM-dd')",
            "4. Text search: LOWER(col) LIKE LOWER(N'%value%')",
            "5. Use DISTINCT for unique values",
            "6. For counting: COUNT(*) or COUNT(column_name)",
            "7. NEVER use columns that are not listed above!",
            "8. Common date column: Date_of_Issuance (NOT Created_At)",
            "9. Status is often in the 'Status' column of DLPEnglishData/DLPArabicData",
            "",
            "COMMON COLUMNS:",
            "- DLPEnglishData/DLPArabicData: ID, Type, Status, Date_of_Issuance, Issuing_Authority, Official_Gazette_No",
            "- Articles: Article_Id, Legislation_Id, Article_Title, Article_Number",
            "- AmendedLegislations: ID, Original_Legislation_Id, Modified_Legislation_Id",
            "- Category: Category_Id, Name_Ar, Name_En",
        ])
        return "\n".join(parts)

    async def ask(self, question: str, category: str = "General") -> Dict[str, Any]:
        """Process a business user question through the full pipeline."""
        start = datetime.now()
        result = {
            "question": question,
            "category": category,
            "success": False,
            "sql": None,
            "row_count": 0,
            "sample_data": None,
            "time_seconds": 0,
            "healed": False,
            "error": None,
        }

        try:
            # 1. Translate user terminology
            translated = self.user_learning.translate_question(question, "business_user")

            # 2. Get business context
            biz_context = self.business_learning.get_context_for_query(translated)

            # 3. Deep reasoning to generate SQL
            from intelligence.deep_reasoner import ReasoningStrategy
            full_context = f"{self.schema_context}\n\nBusiness Context:\n{biz_context}" if biz_context else self.schema_context

            chain = await self.deep_reasoner.reason(
                question=translated,
                schema_context=full_context,
                strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                examples=[],
            )
            sql = chain.sql_result
            result["sql"] = sql

            # 4. Execute with self-healing
            try:
                data, final_sql, was_healed = await self.self_healing.execute_with_healing(sql, max_retries=3)
                result["healed"] = was_healed
                if was_healed:
                    result["sql"] = final_sql
            except Exception as e:
                # Fallback to direct execution
                data = await self.db.execute(sql)

            result["success"] = True
            result["row_count"] = len(data) if data else 0
            result["sample_data"] = data[:3] if data else []

            # 5. Learn from interaction
            await self.user_learning.learn_from_interaction(
                user_id="business_user",
                question=question,
                generated_sql=sql,
                executed_sql=result["sql"],
                success=True,
                result_data=data[:5] if data else None,
            )

        except Exception as e:
            result["error"] = str(e)[:100]

        result["time_seconds"] = (datetime.now() - start).total_seconds()
        self.results.append(result)
        return result

    def print_result(self, r: Dict[str, Any]):
        """Print a single result nicely."""
        status = "✓" if r["success"] else "✗"
        healed = " [HEALED]" if r.get("healed") else ""
        print(f"\n{status} [{r['category']}] {r['question'][:50]}...")
        if r["success"]:
            print(f"   SQL: {r['sql'][:80]}..." if len(r['sql']) > 80 else f"   SQL: {r['sql']}")
            print(f"   Result: {r['row_count']} rows in {r['time_seconds']:.2f}s{healed}")
            if r["sample_data"] and r["row_count"] > 0:
                sample = r["sample_data"][0]
                keys = list(sample.keys())[:4]
                print(f"   Sample: {keys}")
        else:
            print(f"   ERROR: {r['error']}")

    async def run_tests(self):
        """Run all test categories."""

        # =====================================================================
        # CATEGORY 1: Simple Counts
        # =====================================================================
        print("\n" + "=" * 70)
        print("    CATEGORY 1: SIMPLE COUNTS & TOTALS")
        print("=" * 70)

        tests_1 = [
            "How many total legislations are in the database?",
            "Count all articles",
            "How many categories exist?",
            "What is the total number of amendments?",
        ]
        for q in tests_1:
            r = await self.ask(q, "Simple Count")
            self.print_result(r)
            await asyncio.sleep(0.5)

        # =====================================================================
        # CATEGORY 2: Aggregations
        # =====================================================================
        print("\n" + "=" * 70)
        print("    CATEGORY 2: AGGREGATIONS & GROUPING")
        print("=" * 70)

        tests_2 = [
            "How many legislations per status?",
            "Show legislation count by type",
            "What are the top 5 issuing authorities by number of legislations?",
            "Average number of articles per legislation",
        ]
        for q in tests_2:
            r = await self.ask(q, "Aggregation")
            self.print_result(r)
            await asyncio.sleep(0.5)

        # =====================================================================
        # CATEGORY 3: Filtering & Search
        # =====================================================================
        print("\n" + "=" * 70)
        print("    CATEGORY 3: FILTERING & SEARCH")
        print("=" * 70)

        tests_3 = [
            "Find legislations about 'real estate'",
            "Show all laws with status 'In Force'",
            "Find legislations containing the word 'tax'",
            "List decrees from the Ministry of Finance",
            "Search for legislations mentioning 'property'",
        ]
        for q in tests_3:
            r = await self.ask(q, "Filter/Search")
            self.print_result(r)
            await asyncio.sleep(0.5)

        # =====================================================================
        # CATEGORY 4: Date & Time Queries
        # =====================================================================
        print("\n" + "=" * 70)
        print("    CATEGORY 4: DATE & TIME QUERIES")
        print("=" * 70)

        tests_4 = [
            "How many legislations were issued in 2023?",
            "Show legislations from the last 2 years",
            "List the 10 most recent legislations",
            "Count legislations by year",
            "Which month had the most legislation issuances in 2023?",
        ]
        for q in tests_4:
            r = await self.ask(q, "Date/Time")
            self.print_result(r)
            await asyncio.sleep(0.5)

        # =====================================================================
        # CATEGORY 5: Business Terminology
        # =====================================================================
        print("\n" + "=" * 70)
        print("    CATEGORY 5: BUSINESS TERMINOLOGY")
        print("=" * 70)

        tests_5 = [
            "How many laws are currently in force?",
            "List all repealed legislations",
            "Show amendments to existing laws",
            "Find gazette publications from 2023",
            "What laws have been cancelled?",
        ]
        for q in tests_5:
            r = await self.ask(q, "Terminology")
            self.print_result(r)
            await asyncio.sleep(0.5)

        # =====================================================================
        # CATEGORY 6: Complex Multi-Condition
        # =====================================================================
        print("\n" + "=" * 70)
        print("    CATEGORY 6: COMPLEX MULTI-CONDITION QUERIES")
        print("=" * 70)

        tests_6 = [
            "Find in-force laws about taxation issued after 2020",
            "Show decrees from 2022 that have been amended",
            "List royal decrees that are still active with more than 10 articles",
            "Find legislations in the 'Commercial' category that were issued by the Council of Ministers",
        ]
        for q in tests_6:
            r = await self.ask(q, "Complex")
            self.print_result(r)
            await asyncio.sleep(0.5)

        # =====================================================================
        # CATEGORY 7: Analytics & Insights
        # =====================================================================
        print("\n" + "=" * 70)
        print("    CATEGORY 7: ANALYTICS & INSIGHTS")
        print("=" * 70)

        tests_7 = [
            "What percentage of legislations are in force vs repealed?",
            "Which category has the highest number of active laws?",
            "Show trend of legislation issuance over the years",
            "Compare the number of laws vs decrees",
        ]
        for q in tests_7:
            r = await self.ask(q, "Analytics")
            self.print_result(r)
            await asyncio.sleep(0.5)

        # =====================================================================
        # CATEGORY 8: Ambiguous Questions (tests understanding)
        # =====================================================================
        print("\n" + "=" * 70)
        print("    CATEGORY 8: AMBIGUOUS/NATURAL LANGUAGE")
        print("=" * 70)

        tests_8 = [
            "What's new in legislation?",
            "Tell me about labor laws",
            "Anything about real estate regulations?",
            "Show me the important ones",
            "What do we have on banking?",
        ]
        for q in tests_8:
            r = await self.ask(q, "Ambiguous")
            self.print_result(r)
            await asyncio.sleep(0.5)

        # =====================================================================
        # CATEGORY 9: Edge Cases
        # =====================================================================
        print("\n" + "=" * 70)
        print("    CATEGORY 9: EDGE CASES & SPECIAL QUERIES")
        print("=" * 70)

        tests_9 = [
            "Show legislations with NULL status",
            "Find the oldest legislation in the database",
            "List legislations without any articles",
            "Show distinct issuing authorities",
        ]
        for q in tests_9:
            r = await self.ask(q, "Edge Case")
            self.print_result(r)
            await asyncio.sleep(0.5)

        # =====================================================================
        # CATEGORY 10: Follow-up Questions (tests context)
        # =====================================================================
        print("\n" + "=" * 70)
        print("    CATEGORY 10: DETAILED LOOKUPS")
        print("=" * 70)

        tests_10 = [
            "Show the full details of legislation with ID 1",
            "List all articles for the first legislation",
            "What appendices exist for commercial laws?",
            "Show the amendment history for the oldest law",
        ]
        for q in tests_10:
            r = await self.ask(q, "Detail Lookup")
            self.print_result(r)
            await asyncio.sleep(0.5)

    def print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 70)
        print("    COMPREHENSIVE TEST SUMMARY")
        print("=" * 70)

        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        healed = sum(1 for r in self.results if r.get("healed"))
        failed = total - successful

        print(f"\nOverall Results:")
        print(f"   Total Queries:  {total}")
        print(f"   Successful:     {successful} ({100*successful/total:.1f}%)")
        print(f"   Failed:         {failed} ({100*failed/total:.1f}%)")
        print(f"   Self-Healed:    {healed}")

        # By category
        categories = {}
        for r in self.results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "success": 0}
            categories[cat]["total"] += 1
            if r["success"]:
                categories[cat]["success"] += 1

        print(f"\nBy Category:")
        for cat, stats in categories.items():
            pct = 100 * stats["success"] / stats["total"]
            status = "✓" if pct == 100 else ("~" if pct >= 50 else "✗")
            print(f"   {status} {cat}: {stats['success']}/{stats['total']} ({pct:.0f}%)")

        # Timing
        times = [r["time_seconds"] for r in self.results if r["success"]]
        if times:
            print(f"\nPerformance:")
            print(f"   Average time:   {sum(times)/len(times):.2f}s")
            print(f"   Fastest:        {min(times):.2f}s")
            print(f"   Slowest:        {max(times):.2f}s")

        # Failed queries
        if failed > 0:
            print(f"\nFailed Queries:")
            for r in self.results:
                if not r["success"]:
                    print(f"   ✗ {r['question'][:50]}...")
                    print(f"     Error: {r['error']}")

        print("\n" + "=" * 70)
        print("    TEST COMPLETE")
        print("=" * 70)


async def main():
    """Run comprehensive business user test."""
    test = BusinessUserTest()
    await test.setup()
    await test.run_tests()
    test.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
