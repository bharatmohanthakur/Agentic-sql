#!/usr/bin/env python3
"""
END-TO-END Test of the Complete Intelligent Text-to-SQL System

Tests:
1. Auto-Discovery - System discovers schema automatically
2. Auto-Learning - System learns from interactions
3. Self-Healing - System corrects SQL errors
4. Deep Reasoning - System uses advanced reasoning
5. User Learning - System learns user preferences
6. Business Learning - System learns business rules
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

# Load environment
from dotenv import load_dotenv
load_dotenv('/home/admincsp/unified-legislation-agent-langgraph/.env')


async def run_e2e_test():
    """Run complete end-to-end test of the intelligent system"""

    print("\n" + "=" * 70)
    print("    END-TO-END INTELLIGENT SYSTEM TEST")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Initialize LLM Client
    # =========================================================================
    print("\n[STEP 1] Initializing Azure OpenAI LLM Client...")

    from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

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
    llm_client = AzureOpenAIClient(llm_config)

    # Test LLM
    test_response = await llm_client.generate(prompt="Say 'OK' if working", max_tokens=5)
    print(f"   LLM Status: {test_response.strip()}")

    # =========================================================================
    # STEP 2: Initialize Database Adapter
    # =========================================================================
    print("\n[STEP 2] Connecting to MS SQL Server Database...")

    from database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

    db_config = ConnectionConfig(
        name="legislation_db",
        db_type=DatabaseType.MSSQL,
        host="csptecdbserver.database.windows.net,1433",
        database="TEC.Datalake.PreProduction",
        username="cspsb",
        password="Csp00123@@@#$!@#",
    )

    db_adapter = MSSQLAdapter(db_config)
    await db_adapter.connect()
    print("   Database: Connected!")

    # =========================================================================
    # STEP 3: Auto-Discovery
    # =========================================================================
    print("\n[STEP 3] AUTO-DISCOVERY - Discovering Schema Automatically...")

    from intelligence.auto_discovery import SchemaDiscovery, DatabaseDialect

    discovery = SchemaDiscovery(
        db_executor=db_adapter.execute,
        llm_client=llm_client,
        dialect=DatabaseDialect.MSSQL,
    )

    # Discover schema
    profiles = await discovery.discover(
        include_stats=True,
        include_samples=False,  # Skip samples to speed up
        max_tables=30,
    )

    print(f"   Discovered {len(profiles)} tables/views:")
    for name, profile in list(profiles.items())[:10]:
        print(f"     - {name}: {len(profile.columns)} columns, ~{profile.row_count} rows")

    if len(profiles) > 10:
        print(f"     ... and {len(profiles) - 10} more")

    # =========================================================================
    # STEP 4: Initialize Knowledge Base
    # =========================================================================
    print("\n[STEP 4] Initializing Knowledge Base with discovered schema...")

    from intelligence.knowledge_base import KnowledgeBase

    def embedding_fn(text: str) -> List[float]:
        return llm_client.generate_embedding(text)

    knowledge_base = KnowledgeBase(
        embedding_fn=embedding_fn,
        llm_client=llm_client,
    )

    # Ingest discovered schema
    await knowledge_base.ingest_schema(profiles)
    print(f"   Knowledge Base: {knowledge_base.get_stats()}")

    # =========================================================================
    # STEP 5: Initialize User Learning
    # =========================================================================
    print("\n[STEP 5] Initializing User Learning Engine...")

    from intelligence.user_learning import UserLearningEngine

    user_learning = UserLearningEngine(llm_client=llm_client)
    print("   User Learning: Ready")

    # =========================================================================
    # STEP 6: Initialize Business Learning
    # =========================================================================
    print("\n[STEP 6] Initializing Business Learning Engine...")

    from intelligence.business_learning import BusinessLogicLearner

    business_learning = BusinessLogicLearner(
        llm_client=llm_client,
        knowledge_base=knowledge_base,
    )

    # Add legislation-specific terminology
    business_learning.add_terminology(
        term="in force",
        definition="Legislation that is currently active and applicable",
        synonyms=["active", "current", "valid", "effective"]
    )
    business_learning.add_terminology(
        term="repealed",
        definition="Legislation that has been cancelled or revoked",
        synonyms=["cancelled", "revoked", "abolished"]
    )
    print("   Business Learning: Ready with terminology")

    # =========================================================================
    # STEP 7: Initialize Deep Reasoner
    # =========================================================================
    print("\n[STEP 7] Initializing Deep Reasoning Engine...")

    from intelligence.deep_reasoner import DeepReasoner

    deep_reasoner = DeepReasoner(
        llm_client=llm_client,
        knowledge_base=knowledge_base,
        max_depth=5,
    )
    print("   Deep Reasoner: Ready")

    # =========================================================================
    # STEP 8: Initialize Self-Healing Engine
    # =========================================================================
    print("\n[STEP 8] Initializing Self-Healing Engine...")

    from intelligence.self_healing import SelfHealingEngine

    self_healing = SelfHealingEngine(
        schema_profiles=profiles,
        db_executor=db_adapter.execute,
        llm_client=llm_client,
    )
    print("   Self-Healing: Ready")

    # =========================================================================
    # STEP 9: Build Schema Context from Auto-Discovery
    # =========================================================================
    print("\n[STEP 9] Building Schema Context from Discovery...")

    def build_auto_schema_context(profiles: Dict[str, Any]) -> str:
        """Build schema context from auto-discovered profiles"""
        context_parts = [
            "DATABASE SCHEMA (Auto-Discovered)",
            "CRITICAL: Use dbo.TableName format. Never include database name prefix.",
            ""
        ]

        for table_name, profile in profiles.items():
            # Table header
            context_parts.append(f"=== dbo.{table_name} ({profile.table_type.value}) ===")

            if profile.description:
                context_parts.append(f"Description: {profile.description}")

            context_parts.append(f"Row Count: ~{profile.row_count}")

            # Columns
            col_descs = []
            for col in profile.columns[:20]:  # Limit columns
                col_type = col.semantic_type.value if hasattr(col, 'semantic_type') else 'text'
                col_descs.append(f"  - {col.name} ({col.data_type}): {col_type}")

            context_parts.append("Columns:")
            context_parts.extend(col_descs)

            if len(profile.columns) > 20:
                context_parts.append(f"  ... and {len(profile.columns) - 20} more columns")

            context_parts.append("")

        # Add rules
        context_parts.extend([
            "=== QUERY RULES ===",
            "1. Use dbo.TableName format (no database prefix)",
            "2. Use TOP instead of LIMIT (e.g., SELECT TOP 10)",
            "3. Use DISTINCT for unique results",
            "4. Format dates: FORMAT(date_col, 'dd-MMM-yyyy')",
            "5. Text search: LOWER(col) LIKE LOWER(N'%value%')",
        ])

        return "\n".join(context_parts)

    schema_context = build_auto_schema_context(profiles)
    print(f"   Schema Context: {len(schema_context)} characters")

    # =========================================================================
    # STEP 10: Test Query Generation with Full Pipeline
    # =========================================================================
    print("\n[STEP 10] TESTING FULL INTELLIGENT QUERY PIPELINE")
    print("=" * 70)

    async def intelligent_query(question: str, user_id: str = "test_user") -> Dict[str, Any]:
        """Process query through the full intelligent pipeline"""

        print(f"\n--- Question: {question}")
        start = datetime.now()

        # Step 1: Translate user terminology
        translated = user_learning.translate_question(question, user_id)
        if translated != question:
            print(f"   [User Learning] Translated: {translated}")

        # Step 2: Get business context
        business_context = business_learning.get_context_for_query(translated)
        if business_context:
            print(f"   [Business Learning] Context applied")

        # Step 3: Search knowledge base
        relevant_knowledge = await knowledge_base.search(translated, limit=5)
        print(f"   [Knowledge Base] Found {len(relevant_knowledge)} relevant items")

        # Step 4: Deep reasoning to generate SQL
        from intelligence.deep_reasoner import ReasoningStrategy

        full_context = f"{schema_context}\n\nBusiness Context:\n{business_context}" if business_context else schema_context

        reasoning_chain = await deep_reasoner.reason(
            question=translated,
            schema_context=full_context,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            examples=[],
        )

        sql = reasoning_chain.sql_result
        print(f"   [Deep Reasoner] Generated SQL ({reasoning_chain.total_confidence:.2f} confidence)")
        print(f"   SQL: {sql[:200]}..." if len(sql) > 200 else f"   SQL: {sql}")

        # Step 5: Execute with self-healing
        try:
            data, final_sql, was_healed = await self_healing.execute_with_healing(
                sql, max_retries=3
            )
            if was_healed:
                print(f"   [Self-Healing] SQL was corrected!")
                print(f"   Fixed SQL: {final_sql[:200]}..." if len(final_sql) > 200 else f"   Fixed SQL: {final_sql}")
        except Exception as e:
            print(f"   [Error] {e}")
            # Try direct execution as fallback
            try:
                data = await db_adapter.execute(sql)
                final_sql = sql
                was_healed = False
            except Exception as e2:
                return {"success": False, "error": str(e2), "sql": sql}

        elapsed = (datetime.now() - start).total_seconds()

        # Step 6: Learn from interaction
        await user_learning.learn_from_interaction(
            user_id=user_id,
            question=question,
            generated_sql=sql,
            executed_sql=final_sql,
            success=True,
            result_data=data[:5] if data else None,
        )
        print(f"   [Learning] Learned from this interaction")

        # Results
        row_count = len(data) if data else 0
        print(f"   RESULT: {row_count} rows in {elapsed:.2f}s")

        if data and row_count > 0:
            print(f"   Sample: {list(data[0].keys())[:5]}")

        return {
            "success": True,
            "question": question,
            "sql": final_sql,
            "row_count": row_count,
            "time_seconds": elapsed,
            "was_healed": was_healed,
            "data": data[:3] if data else [],
        }

    # Run test queries
    test_questions = [
        "How many legislations are currently in force?",
        "What are the different types of legislation?",
        "Which authority has issued the most legislations?",
        "Show me the top 5 categories with most legislations",
        "Find legislations about real estate",
        "List laws issued in 2024 that are in force",
    ]

    results = []
    for question in test_questions:
        try:
            result = await intelligent_query(question)
            results.append(result)
        except Exception as e:
            print(f"   FAILED: {e}")
            results.append({"success": False, "question": question, "error": str(e)})

        await asyncio.sleep(1)  # Rate limiting

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("    TEST SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r.get("success"))
    healed = sum(1 for r in results if r.get("was_healed"))

    print(f"\nTotal Queries: {len(results)}")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Self-Healed: {healed}")

    print("\nDetailed Results:")
    for r in results:
        status = "OK" if r.get("success") else "FAIL"
        q = r.get("question", "?")[:45]
        rows = r.get("row_count", 0)
        time = r.get("time_seconds", 0)
        healed = " [HEALED]" if r.get("was_healed") else ""
        print(f"  [{status}] {q}... ({rows} rows, {time:.2f}s){healed}")

    print("\nSystem Statistics:")
    print(f"  - Tables Discovered: {len(profiles)}")
    print(f"  - Knowledge Items: {knowledge_base.get_stats()}")
    print(f"  - User Profiles: {len(user_learning._profiles)}")
    print(f"  - Business Rules: {business_learning.get_stats()}")

    print("\n" + "=" * 70)
    print("    END-TO-END TEST COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    asyncio.run(run_e2e_test())
