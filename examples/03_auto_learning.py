#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 3: AUTO-LEARNING - Train the Agent Automatically
=============================================================================

This tutorial shows how to use the auto_learn() feature to train
the agent on your database automatically.

The agent will:
1. Explore your database to understand the domain
2. Generate test questions relevant to your data
3. Run those questions and learn from results
4. Identify weak areas and improve

Run: python examples/03_auto_learning.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()


async def main():
    print("=" * 60)
    print("  TUTORIAL 3: AUTO-LEARNING")
    print("=" * 60)

    # =========================================================================
    # SETUP
    # =========================================================================
    from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
    from intelligence.meta_agent import MetaAgent
    from database.multi_db import SQLiteAdapter, ConnectionConfig, DatabaseType

    # Create LLM client
    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4o",
    ))

    # Create a sample database for this tutorial
    print("\n[Setup] Creating sample e-commerce database...")

    db = SQLiteAdapter(ConnectionConfig(
        name="ecommerce_db",
        db_type=DatabaseType.SQLITE,
        database=":memory:",
    ))
    await db.connect()

    # Create sample tables
    await db.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            country TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    await db.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price DECIMAL(10,2),
            stock INTEGER DEFAULT 0
        )
    """)

    await db.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            total_amount DECIMAL(10,2),
            status TEXT DEFAULT 'pending',
            order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    # Insert sample data
    await db.execute("INSERT INTO customers (name, email, country) VALUES ('Alice', 'alice@email.com', 'USA')")
    await db.execute("INSERT INTO customers (name, email, country) VALUES ('Bob', 'bob@email.com', 'UK')")
    await db.execute("INSERT INTO customers (name, email, country) VALUES ('Charlie', 'charlie@email.com', 'USA')")
    await db.execute("INSERT INTO customers (name, email, country) VALUES ('Diana', 'diana@email.com', 'Canada')")

    await db.execute("INSERT INTO products (name, category, price, stock) VALUES ('Laptop', 'Electronics', 999.99, 50)")
    await db.execute("INSERT INTO products (name, category, price, stock) VALUES ('Phone', 'Electronics', 699.99, 100)")
    await db.execute("INSERT INTO products (name, category, price, stock) VALUES ('Headphones', 'Electronics', 199.99, 200)")
    await db.execute("INSERT INTO products (name, category, price, stock) VALUES ('Book', 'Books', 29.99, 500)")

    await db.execute("INSERT INTO orders (customer_id, product_id, quantity, total_amount, status) VALUES (1, 1, 1, 999.99, 'completed')")
    await db.execute("INSERT INTO orders (customer_id, product_id, quantity, total_amount, status) VALUES (1, 3, 2, 399.98, 'completed')")
    await db.execute("INSERT INTO orders (customer_id, product_id, quantity, total_amount, status) VALUES (2, 2, 1, 699.99, 'shipped')")
    await db.execute("INSERT INTO orders (customer_id, product_id, quantity, total_amount, status) VALUES (3, 4, 3, 89.97, 'pending')")
    await db.execute("INSERT INTO orders (customer_id, product_id, quantity, total_amount, status) VALUES (4, 1, 1, 999.99, 'completed')")

    print("  ✓ Created tables: customers, products, orders")
    print("  ✓ Inserted sample data")

    # =========================================================================
    # CONNECT AGENT
    # =========================================================================
    print("\n[Step 1] Connecting MetaAgent...")

    agent = MetaAgent(llm_client=llm)
    stats = await agent.connect(db_executor=db.execute)

    print(f"  ✓ Discovered {stats['tables']} tables")
    print(f"  ✓ Dialect: {stats['dialect']}")

    # =========================================================================
    # AUTO-LEARN: LIGHT INTENSITY
    # =========================================================================
    print("\n[Step 2] Running AUTO-LEARN (light intensity)...")
    print("  This will generate 5 questions and test them.\n")

    results = await agent.auto_learn(intensity="light")

    print("\n  AUTO-LEARN RESULTS:")
    print(f"  • Domain discovered: {results['domain']}")
    print(f"  • Questions generated: {results['questions_generated']}")
    print(f"  • Questions tested: {results['questions_tested']}")
    print(f"  • Successes: {results['successes']}")
    print(f"  • Failures: {results['failures']}")
    print(f"  • Success rate: {results['success_rate']*100:.1f}%")

    # =========================================================================
    # CHECK WHAT WAS LEARNED
    # =========================================================================
    print("\n[Step 3] Checking learned knowledge...")

    stats = agent.get_stats()
    print(f"  • Solutions stored: {stats['solutions_stored']}")
    print(f"  • Failures analyzed: {stats['failures_analyzed']}")
    print(f"  • Schema insights: {stats['schema_insights']}")

    # Check name corrections learned
    if agent.knowledge.name_corrections:
        print(f"  • Name corrections learned: {agent.knowledge.name_corrections}")

    # Check table relationships learned
    if agent.knowledge.table_relationships:
        print(f"  • Table relationships: {agent.knowledge.table_relationships}")

    # =========================================================================
    # TEST WITH NEW QUESTIONS
    # =========================================================================
    print("\n[Step 4] Testing with new questions (using learned knowledge)...")

    test_questions = [
        "What is the total revenue from completed orders?",
        "Which customer spent the most money?",
        "What products are low in stock?",
        "How many orders per country?",
    ]

    for q in test_questions:
        print(f"\n  Q: {q}")
        result = await agent.query(q)
        if result["success"]:
            print(f"  ✓ Success ({result['row_count']} rows, {result['iterations']} iterations)")
            print(f"  SQL: {result['sql'][:60]}...")
            if result["data"]:
                print(f"  Result: {result['data'][:2]}...")
        else:
            print(f"  ✗ Failed: {result['error'][:50]}...")

    # =========================================================================
    # AUTO-LEARN: MEDIUM INTENSITY (Optional)
    # =========================================================================
    print("\n" + "-" * 60)
    print("[Optional] For more comprehensive training, use medium or heavy:")
    print("""
    # Medium: 15 questions
    results = await agent.auto_learn(intensity="medium")

    # Heavy: 30 questions
    results = await agent.auto_learn(intensity="heavy")
    """)

    # =========================================================================
    # KNOWLEDGE PERSISTENCE
    # =========================================================================
    print("\n[Note] Knowledge Persistence:")
    print("  All learned knowledge is automatically saved to:")
    print(f"  ~/.vanna/meta_agent.json")
    print("\n  Next time you start the agent, it loads this knowledge!")

    print("\n" + "=" * 60)
    print("  TUTORIAL COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
