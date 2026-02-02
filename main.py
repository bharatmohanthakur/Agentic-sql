"""
Intelligent Text-to-SQL System - Main Entry Point

A best-in-class agentic system that:
- AUTO-DISCOVERS database schemas when connected
- AUTO-TRAINS from successful queries and corrections
- SELF-HEALS by automatically fixing errors
- DEEP REASONS using Chain-of-Thought and Tree-of-Thought
- SCALES to 100+ tables across multiple databases
- LEARNS continuously from all interactions

Usage:
    python main.py              # Run demo
    python main.py serve        # Run API server
    python main.py discover     # Discover schema only
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def create_intelligent_system():
    """
    Create the full intelligent system with:
    - Auto-discovery
    - Auto-training
    - Self-healing
    - Deep reasoning
    - Knowledge integration
    """
    from src.intelligence.intelligent_system import (
        IntelligentSystem,
        IntelligentSystemConfig,
    )
    from src.intelligence.deep_reasoner import ReasoningStrategy
    from src.llm.base import LLMConfig, LLMProvider
    from src.llm.openai_client import OpenAIClient

    # 1. Create LLM Client
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=os.getenv("LLM_MODEL", "gpt-4"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
        max_tokens=4096,
        enable_caching=True,
        enable_cost_tracking=True,
    )
    llm = OpenAIClient(llm_config)
    logger.info(f"LLM client initialized: {llm_config.model}")

    # 2. Create Intelligent System config
    system_config = IntelligentSystemConfig(
        auto_discover_on_connect=True,
        enable_auto_learning=True,
        enable_self_healing=True,
        enable_deep_reasoning=True,
        default_reasoning_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        max_reasoning_depth=10,
        max_healing_attempts=3,
    )

    # 3. Create Intelligent System
    system = IntelligentSystem(
        config=system_config,
        llm_client=llm,
    )
    logger.info("Intelligent System created")

    return system, llm


async def create_demo_system():
    """Create a demo system with all components (legacy mode)"""
    from src.core.base import UserContext
    from src.core.registry import ToolRegistry
    from src.core.orchestrator import AgentOrchestrator, PipelineBuilder
    from src.agents.sql_agent import SQLAgent, SQLAgentConfig
    from src.memory.manager import MemoryManager, MemoryConfig
    from src.llm.base import LLMConfig, LLMProvider
    from src.llm.openai_client import OpenAIClient
    from src.tools.database import GetSchemaTool, ExecuteSQLTool

    # 1. Create Memory Manager with hybrid storage
    memory_config = MemoryConfig(
        enable_graph=True,
        enable_vector=True,
        enable_sql=True,
        similarity_threshold=0.7,
    )
    memory = MemoryManager(config=memory_config)
    logger.info("Memory manager initialized")

    # 2. Create LLM Client
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=os.getenv("LLM_MODEL", "gpt-4"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
        max_tokens=4096,
        enable_caching=True,
        enable_cost_tracking=True,
    )
    llm = OpenAIClient(llm_config)
    logger.info(f"LLM client initialized: {llm_config.model}")

    # 3. Create Tool Registry
    registry = ToolRegistry()

    # Mock database connection for demo
    class MockDB:
        async def execute(self, sql: str, params: Optional[Dict] = None):
            # Demo: return mock data based on query
            sql_lower = sql.lower()

            if "customer" in sql_lower:
                return [
                    {"customer_id": 1, "name": "Acme Corp", "revenue": 150000},
                    {"customer_id": 2, "name": "TechStart", "revenue": 230000},
                    {"customer_id": 3, "name": "Global Inc", "revenue": 180000},
                    {"customer_id": 4, "name": "Local Shop", "revenue": 45000},
                    {"customer_id": 5, "name": "Big Enterprise", "revenue": 520000},
                ]
            elif "product" in sql_lower:
                return [
                    {"id": 1, "name": "Product A", "sales": 1500, "category": "Electronics"},
                    {"id": 2, "name": "Product B", "sales": 2300, "category": "Software"},
                    {"id": 3, "name": "Product C", "sales": 1800, "category": "Electronics"},
                ]
            elif "order" in sql_lower:
                return [
                    {"order_id": 101, "customer": "Acme Corp", "amount": 5000, "date": "2024-01-15"},
                    {"order_id": 102, "customer": "TechStart", "amount": 7500, "date": "2024-01-16"},
                    {"order_id": 103, "customer": "Global Inc", "amount": 3200, "date": "2024-01-17"},
                ]
            else:
                return [
                    {"id": 1, "name": "Product A", "sales": 1500},
                    {"id": 2, "name": "Product B", "sales": 2300},
                    {"id": 3, "name": "Product C", "sales": 1800},
                ]

        async def fetch_schema(self):
            return {
                "tables": [
                    {"name": "products", "description": "Product catalog with pricing"},
                    {"name": "orders", "description": "Customer orders and transactions"},
                    {"name": "customers", "description": "Customer information and contacts"},
                    {"name": "order_items", "description": "Individual items in orders"},
                    {"name": "categories", "description": "Product categories"},
                    {"name": "inventory", "description": "Stock levels per product"},
                    {"name": "suppliers", "description": "Product suppliers"},
                    {"name": "employees", "description": "Company employees"},
                    {"name": "regions", "description": "Geographic regions"},
                    {"name": "payments", "description": "Payment transactions"},
                ],
                "columns": {
                    "products": [
                        {"name": "id", "type": "INTEGER", "description": "Primary key"},
                        {"name": "name", "type": "VARCHAR(255)", "description": "Product name"},
                        {"name": "price", "type": "DECIMAL(10,2)", "description": "Unit price"},
                        {"name": "category_id", "type": "INTEGER", "description": "FK to categories"},
                        {"name": "supplier_id", "type": "INTEGER", "description": "FK to suppliers"},
                    ],
                    "orders": [
                        {"name": "id", "type": "INTEGER", "description": "Primary key"},
                        {"name": "customer_id", "type": "INTEGER", "description": "FK to customers"},
                        {"name": "order_date", "type": "DATE", "description": "Date of order"},
                        {"name": "total_amount", "type": "DECIMAL(10,2)", "description": "Order total"},
                        {"name": "status", "type": "VARCHAR(50)", "description": "Order status"},
                    ],
                    "customers": [
                        {"name": "id", "type": "INTEGER", "description": "Primary key"},
                        {"name": "name", "type": "VARCHAR(255)", "description": "Customer name"},
                        {"name": "email", "type": "VARCHAR(255)", "description": "Email address"},
                        {"name": "region_id", "type": "INTEGER", "description": "FK to regions"},
                        {"name": "created_at", "type": "TIMESTAMP", "description": "Registration date"},
                    ],
                },
                "relationships": [
                    {"from_table": "orders", "from_column": "customer_id",
                     "to_table": "customers", "to_column": "id"},
                    {"from_table": "products", "from_column": "category_id",
                     "to_table": "categories", "to_column": "id"},
                    {"from_table": "order_items", "from_column": "order_id",
                     "to_table": "orders", "to_column": "id"},
                    {"from_table": "order_items", "from_column": "product_id",
                     "to_table": "products", "to_column": "id"},
                ],
            }

        async def explain(self, sql: str):
            return "Seq Scan on products (cost=0.00..1.10 rows=10 width=64)"

    db = MockDB()

    # Register tools
    registry.register(GetSchemaTool(db))
    registry.register(ExecuteSQLTool(db))
    logger.info(f"Registered {len(registry._tools)} tools")

    # 4. Create SQL Agent
    sql_config = SQLAgentConfig(
        name="sql_agent",
        max_iterations=5,
        enable_reflection=True,
        enable_planning=True,
        enable_memory=True,
        block_destructive_queries=True,
        require_row_level_security=True,
    )

    sql_agent = SQLAgent(
        config=sql_config,
        tool_registry=registry,
        memory=memory,
        llm_client=llm,
        db_executor=db.execute,
    )
    logger.info("SQL Agent initialized")

    # 5. Create Orchestrator for multi-agent workflows
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent("sql", sql_agent)
    logger.info("Orchestrator initialized")

    return {
        "memory": memory,
        "llm": llm,
        "registry": registry,
        "sql_agent": sql_agent,
        "orchestrator": orchestrator,
        "db": db,
    }


async def run_intelligent_demo():
    """
    Run the intelligent system demo

    Demonstrates:
    - Auto-discovery of schema
    - Deep reasoning for queries
    - Self-healing on errors
    - Learning from interactions
    """
    logger.info("\n" + "="*70)
    logger.info("INTELLIGENT TEXT-TO-SQL SYSTEM - DEMO")
    logger.info("="*70)

    system, llm = await create_intelligent_system()

    # Connect to demo database
    logger.info("\n[1] CONNECTING & AUTO-DISCOVERING...")

    # For demo, we'll simulate the connection
    # In production: await system.connect([{"type": "postgresql", "host": "...", ...}])

    # Simulate discovered schema
    from src.intelligence.auto_discovery import TableProfile, ColumnProfile, TableType, ColumnType

    # Create mock profiles
    system._schema_profiles["demo"] = {
        "customers": TableProfile(
            name="customers",
            table_type=TableType.ENTITY,
            row_count=10000,
            description="Customer information including contact details and purchase history",
            columns=[
                ColumnProfile(name="id", data_type="INTEGER", semantic_type=ColumnType.PRIMARY_KEY),
                ColumnProfile(name="name", data_type="VARCHAR", semantic_type=ColumnType.NAME),
                ColumnProfile(name="email", data_type="VARCHAR", semantic_type=ColumnType.EMAIL),
                ColumnProfile(name="revenue", data_type="DECIMAL", semantic_type=ColumnType.AMOUNT),
                ColumnProfile(name="region", data_type="VARCHAR", semantic_type=ColumnType.CATEGORY),
            ],
        ),
        "orders": TableProfile(
            name="orders",
            table_type=TableType.TRANSACTION,
            row_count=50000,
            description="Customer orders with amounts and dates",
            columns=[
                ColumnProfile(name="id", data_type="INTEGER", semantic_type=ColumnType.PRIMARY_KEY),
                ColumnProfile(name="customer_id", data_type="INTEGER", semantic_type=ColumnType.FOREIGN_KEY),
                ColumnProfile(name="amount", data_type="DECIMAL", semantic_type=ColumnType.AMOUNT),
                ColumnProfile(name="order_date", data_type="DATE", semantic_type=ColumnType.DATE),
            ],
        ),
        "products": TableProfile(
            name="products",
            table_type=TableType.ENTITY,
            row_count=500,
            description="Product catalog with pricing",
            columns=[
                ColumnProfile(name="id", data_type="INTEGER", semantic_type=ColumnType.PRIMARY_KEY),
                ColumnProfile(name="name", data_type="VARCHAR", semantic_type=ColumnType.NAME),
                ColumnProfile(name="price", data_type="DECIMAL", semantic_type=ColumnType.PRICE),
                ColumnProfile(name="category", data_type="VARCHAR", semantic_type=ColumnType.CATEGORY),
            ],
        ),
    }

    # Ingest schema into knowledge base
    await system.knowledge_base.ingest_schema(system._schema_profiles["demo"])

    logger.info(f"   Discovered {len(system._schema_profiles['demo'])} tables")
    logger.info(f"   Knowledge items: {system.knowledge_base.get_stats()['total_items']}")

    # Demo queries
    demo_queries = [
        {
            "question": "Who are the top 5 customers by revenue?",
            "expected_complexity": "simple",
        },
        {
            "question": "Show me the monthly order trends for the past year",
            "expected_complexity": "moderate",
        },
        {
            "question": "Which products have declining sales compared to last quarter and why might that be?",
            "expected_complexity": "complex",
        },
    ]

    logger.info("\n[2] PROCESSING QUERIES WITH DEEP REASONING...")

    for i, demo in enumerate(demo_queries, 1):
        question = demo["question"]
        logger.info(f"\n{'─'*60}")
        logger.info(f"Query {i}: {question}")
        logger.info(f"Expected complexity: {demo['expected_complexity']}")
        logger.info("─"*60)

        # Mock LLM for demo (would use real LLM in production)
        class MockLLMForDemo:
            async def generate(self, prompt: str = "", **kwargs):
                if "top 5 customers" in prompt.lower():
                    return """
                    STEP 1: Need to find customers and their revenue
                    STEP 2: The customers table has revenue column
                    STEP 3: Order by revenue descending
                    STEP 4: Limit to 5

                    ```sql
                    SELECT name, revenue
                    FROM customers
                    ORDER BY revenue DESC
                    LIMIT 5
                    ```

                    CONFIDENCE: 0.95
                    """
                elif "monthly" in prompt.lower():
                    return """
                    STEP 1: Need order data grouped by month
                    STEP 2: Use orders table with order_date
                    STEP 3: Extract month and sum amounts

                    ```sql
                    SELECT
                        DATE_TRUNC('month', order_date) as month,
                        SUM(amount) as total_orders,
                        COUNT(*) as order_count
                    FROM orders
                    WHERE order_date >= NOW() - INTERVAL '1 year'
                    GROUP BY DATE_TRUNC('month', order_date)
                    ORDER BY month
                    ```

                    CONFIDENCE: 0.90
                    """
                elif "declining" in prompt.lower():
                    return """
                    STEP 1: This is a complex query requiring comparison
                    STEP 2: Need current quarter sales and previous quarter
                    STEP 3: Calculate the difference

                    ```sql
                    WITH current_quarter AS (
                        SELECT product_id, SUM(amount) as current_sales
                        FROM orders
                        WHERE order_date >= DATE_TRUNC('quarter', NOW())
                        GROUP BY product_id
                    ),
                    previous_quarter AS (
                        SELECT product_id, SUM(amount) as prev_sales
                        FROM orders
                        WHERE order_date >= DATE_TRUNC('quarter', NOW() - INTERVAL '3 months')
                          AND order_date < DATE_TRUNC('quarter', NOW())
                        GROUP BY product_id
                    )
                    SELECT
                        p.name,
                        p.category,
                        COALESCE(cq.current_sales, 0) as current_sales,
                        COALESCE(pq.prev_sales, 0) as previous_sales,
                        COALESCE(pq.prev_sales, 0) - COALESCE(cq.current_sales, 0) as decline
                    FROM products p
                    LEFT JOIN current_quarter cq ON p.id = cq.product_id
                    LEFT JOIN previous_quarter pq ON p.id = pq.product_id
                    WHERE COALESCE(cq.current_sales, 0) < COALESCE(pq.prev_sales, 0)
                    ORDER BY decline DESC
                    ```

                    CONFIDENCE: 0.85
                    """
                return "SELECT 1"

        system.llm = MockLLMForDemo()
        system.deep_reasoner.llm = MockLLMForDemo()

        # Assess complexity
        complexity = await system._assess_complexity(question)
        logger.info(f"   Intelligence level: {complexity.value}")

        # Get strategy
        strategy = system._select_strategy(complexity)
        logger.info(f"   Reasoning strategy: {strategy.value}")

        # Build schema context
        schema_context = system._build_schema_context(question)
        logger.info(f"   Schema context: {len(schema_context)} chars")

        # Perform reasoning
        chain = await system.deep_reasoner.reason(
            question=question,
            schema_context=schema_context,
            strategy=strategy,
        )

        logger.info(f"   Reasoning steps: {len(chain.steps)}")
        logger.info(f"   Confidence: {chain.total_confidence:.2f}")

        if chain.sql_result:
            sql_preview = chain.sql_result[:100].replace('\n', ' ')
            logger.info(f"   Generated SQL: {sql_preview}...")

    # Show system stats
    logger.info("\n[3] SYSTEM STATISTICS")
    logger.info("─"*60)

    stats = {
        "Tables discovered": sum(len(p) for p in system._schema_profiles.values()),
        "Knowledge items": system.knowledge_base.get_stats()["total_items"],
        "Training examples": len(system.training_pipeline._examples),
    }

    for key, value in stats.items():
        logger.info(f"   {key}: {value}")

    logger.info("\n" + "="*70)
    logger.info("DEMO COMPLETE - System is ready for production use")
    logger.info("="*70)


async def run_demo():
    """Run a demo query (legacy mode)"""
    from src.core.base import UserContext

    system = await create_demo_system()
    sql_agent = system["sql_agent"]

    # Create user context
    user = UserContext(
        user_id="demo_user",
        roles=["analyst"],
        permissions={
            "allowed_schemas": ["public"],
            "sql_filters": {},
        },
    )

    # Example queries
    queries = [
        "What are the top 3 products by sales?",
        "Show me total orders per customer",
        "Which customers have the highest revenue?",
    ]

    for question in queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Question: {question}")
        logger.info("="*60)

        result = await sql_agent.execute(question, user_context=user)

        if result.success:
            data = result.data
            logger.info(f"SQL: {data.get('sql', 'N/A')}")
            logger.info(f"Rows: {data.get('row_count', 0)}")
            logger.info(f"Data: {data.get('data', [])[:3]}")
            if data.get('explanation'):
                logger.info(f"Explanation: {data['explanation']}")
        else:
            logger.error(f"Error: {result.error}")


async def run_api_server():
    """Run the FastAPI server"""
    from src.api.server import create_app, APIConfig, run_server
    from src.api.auth import JWTUserResolver

    system = await create_demo_system()

    # Create user resolver
    user_resolver = JWTUserResolver(
        secret_key=os.getenv("JWT_SECRET", "demo-secret-key"),
    )

    # Create app
    config = APIConfig(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("DEBUG", "false").lower() == "true",
    )

    app = create_app(
        config=config,
        sql_agent=system["sql_agent"],
        user_resolver=user_resolver.resolve,
    )

    logger.info(f"Starting API server on {config.host}:{config.port}")
    run_server(app, config)


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "serve":
            # Run API server
            asyncio.run(run_api_server())

        elif command == "intelligent" or command == "smart":
            # Run intelligent system demo
            asyncio.run(run_intelligent_demo())

        elif command == "legacy":
            # Run legacy demo
            asyncio.run(run_demo())

        elif command == "help":
            print("""
Intelligent Text-to-SQL System

Usage:
    python main.py              Run intelligent demo (default)
    python main.py intelligent  Run intelligent system demo
    python main.py serve        Start API server
    python main.py legacy       Run legacy demo mode
    python main.py help         Show this help

Features:
    - AUTO-DISCOVERY: Automatically discovers database schemas
    - AUTO-TRAINING: Learns from successful queries
    - SELF-HEALING: Fixes errors automatically
    - DEEP REASONING: Chain-of-Thought and Tree-of-Thought
    - MULTI-DATABASE: Supports 100+ tables across databases
            """)
        else:
            print(f"Unknown command: {command}")
            print("Use 'python main.py help' for usage")
    else:
        # Default: Run intelligent demo
        asyncio.run(run_intelligent_demo())


if __name__ == "__main__":
    main()
