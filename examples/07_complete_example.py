#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 7: COMPLETE END-TO-END EXAMPLE
=============================================================================

This is a comprehensive example that demonstrates the full power of
Agentic SQL by building a complete analytics assistant.

Features demonstrated:
- Database connection and auto-discovery
- Auto-learning for the domain
- Interactive query session
- Memory persistence
- Error handling and self-healing

Run: python examples/07_complete_example.py
"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()


async def create_sample_database():
    """Create a realistic sample database for the demo."""
    from database.multi_db import SQLiteAdapter, ConnectionConfig, DatabaseType

    db = SQLiteAdapter(ConnectionConfig(
        name="analytics_demo",
        db_type=DatabaseType.SQLITE,
        database="./demo_analytics.db",
    ))
    await db.connect()

    # Create schema
    await db.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            segment TEXT CHECK(segment IN ('enterprise', 'mid-market', 'smb')),
            country TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            lifetime_value DECIMAL(10,2) DEFAULT 0
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT,
            price DECIMAL(10,2) NOT NULL,
            cost DECIMAL(10,2),
            stock_level INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT 1
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER REFERENCES customers(id),
            order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT CHECK(status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled')),
            total_amount DECIMAL(10,2),
            discount_percent DECIMAL(5,2) DEFAULT 0,
            shipping_country TEXT
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER REFERENCES orders(id),
            product_id INTEGER REFERENCES products(id),
            quantity INTEGER NOT NULL,
            unit_price DECIMAL(10,2) NOT NULL,
            total DECIMAL(10,2)
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS support_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER REFERENCES customers(id),
            subject TEXT NOT NULL,
            priority TEXT CHECK(priority IN ('low', 'medium', 'high', 'critical')),
            status TEXT CHECK(status IN ('open', 'in_progress', 'resolved', 'closed')),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            resolved_at DATETIME
        )
    """)

    # Clear existing data
    await db.execute("DELETE FROM order_items")
    await db.execute("DELETE FROM orders")
    await db.execute("DELETE FROM support_tickets")
    await db.execute("DELETE FROM products")
    await db.execute("DELETE FROM customers")

    # Insert sample data
    customers = [
        ("Acme Corp", "contact@acme.com", "enterprise", "USA", 125000.00),
        ("TechStart Inc", "hello@techstart.io", "smb", "USA", 15000.00),
        ("Global Traders", "info@globaltraders.com", "mid-market", "UK", 45000.00),
        ("Local Shop", "owner@localshop.com", "smb", "Canada", 8000.00),
        ("MegaCorp", "sales@megacorp.com", "enterprise", "Germany", 250000.00),
        ("StartupXYZ", "team@startupxyz.com", "smb", "USA", 12000.00),
        ("Euro Systems", "contact@eurosystems.eu", "mid-market", "France", 67000.00),
        ("Asia Pacific Ltd", "info@asiapacific.com", "enterprise", "Japan", 180000.00),
    ]

    for name, email, segment, country, ltv in customers:
        await db.execute(
            "INSERT INTO customers (name, email, segment, country, lifetime_value) VALUES (?, ?, ?, ?, ?)",
            (name, email, segment, country, ltv)
        )

    products = [
        ("Enterprise Suite", "Software", 9999.00, 2000.00, 100),
        ("Pro License", "Software", 499.00, 100.00, 500),
        ("Basic Plan", "Software", 99.00, 20.00, 1000),
        ("Consulting Hours", "Services", 250.00, 150.00, 200),
        ("Training Package", "Services", 1500.00, 500.00, 50),
        ("Hardware Bundle", "Hardware", 2999.00, 1800.00, 75),
        ("Support Contract", "Services", 5000.00, 1000.00, 100),
    ]

    for name, category, price, cost, stock in products:
        await db.execute(
            "INSERT INTO products (name, category, price, cost, stock_level) VALUES (?, ?, ?, ?, ?)",
            (name, category, price, cost, stock)
        )

    # Sample orders
    orders = [
        (1, "2024-01-15", "delivered", 10499.00, 5, "USA"),
        (1, "2024-02-20", "delivered", 5000.00, 0, "USA"),
        (2, "2024-01-20", "delivered", 598.00, 10, "USA"),
        (3, "2024-02-01", "shipped", 3499.00, 0, "UK"),
        (4, "2024-02-15", "delivered", 99.00, 0, "Canada"),
        (5, "2024-01-10", "delivered", 24997.00, 15, "Germany"),
        (5, "2024-03-01", "processing", 7500.00, 0, "Germany"),
        (6, "2024-02-28", "pending", 499.00, 0, "USA"),
        (7, "2024-01-25", "delivered", 4499.00, 5, "France"),
        (8, "2024-02-10", "delivered", 15998.00, 10, "Japan"),
    ]

    for cust_id, date, status, total, discount, country in orders:
        await db.execute(
            "INSERT INTO orders (customer_id, order_date, status, total_amount, discount_percent, shipping_country) VALUES (?, ?, ?, ?, ?, ?)",
            (cust_id, date, status, total, discount, country)
        )

    # Sample tickets
    tickets = [
        (1, "Integration issue", "high", "resolved", "2024-01-20", "2024-01-21"),
        (2, "Billing question", "low", "closed", "2024-02-01", "2024-02-01"),
        (3, "Feature request", "medium", "open", "2024-02-15", None),
        (5, "Urgent: System down", "critical", "resolved", "2024-01-15", "2024-01-15"),
        (8, "Training request", "low", "in_progress", "2024-02-20", None),
    ]

    for cust_id, subject, priority, status, created, resolved in tickets:
        await db.execute(
            "INSERT INTO support_tickets (customer_id, subject, priority, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?)",
            (cust_id, subject, priority, status, created, resolved)
        )

    return db


async def main():
    print("=" * 70)
    print("  AGENTIC SQL - COMPLETE ANALYTICS ASSISTANT DEMO")
    print("=" * 70)

    # =========================================================================
    # STEP 1: CREATE DATABASE
    # =========================================================================
    print("\n[1/5] Creating sample analytics database...")

    db = await create_sample_database()

    print("  ✓ Created tables: customers, products, orders, order_items, support_tickets")
    print("  ✓ Inserted sample data: 8 customers, 7 products, 10 orders, 5 tickets")

    # =========================================================================
    # STEP 2: SETUP LLM
    # =========================================================================
    print("\n[2/5] Setting up LLM client...")

    from llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("  ⚠ AZURE_OPENAI_API_KEY not set!")
        print("  Set environment variables to run this demo:")
        print("    export AZURE_OPENAI_API_KEY=your-key")
        print("    export AZURE_OPENAI_ENDPOINT=your-endpoint")
        return

    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version="2024-02-01",
    ))
    print("  ✓ LLM client ready")

    # =========================================================================
    # STEP 3: CREATE AGENT
    # =========================================================================
    print("\n[3/5] Creating MetaAgent and connecting to database...")

    from intelligence.meta_agent import MetaAgent

    agent = MetaAgent(llm_client=llm)
    stats = await agent.connect(db_executor=db.execute)

    print(f"  ✓ Connected to database")
    print(f"  ✓ Dialect detected: {stats['dialect']}")
    print(f"  ✓ Tables discovered: {stats['tables']}")
    print(f"  ✓ Schema insights: {stats['schema_insights']}")

    # =========================================================================
    # STEP 4: AUTO-LEARN
    # =========================================================================
    print("\n[4/5] Running auto-learning...")

    learn_results = await agent.auto_learn(intensity="light")

    print(f"  ✓ Domain identified: {learn_results['domain']}")
    print(f"  ✓ Questions tested: {learn_results['questions_tested']}")
    print(f"  ✓ Success rate: {learn_results['success_rate']*100:.0f}%")

    # =========================================================================
    # STEP 5: INTERACTIVE SESSION
    # =========================================================================
    print("\n[5/5] Interactive Query Session")
    print("=" * 70)
    print("  Ask questions about your business data!")
    print("  Type 'quit' to exit, 'stats' to see agent statistics")
    print("=" * 70)

    # Demo questions to show capabilities
    demo_questions = [
        "What is the total revenue from all orders?",
        "Which customer segment generates the most revenue?",
        "Show the top 3 customers by lifetime value",
        "What is our average order value by country?",
        "How many support tickets are still open?",
        "Which products have the highest profit margin?",
        "Show monthly revenue trends",
        "What percentage of orders have been delivered?",
    ]

    print("\n  Example questions you can ask:")
    for i, q in enumerate(demo_questions, 1):
        print(f"    {i}. {q}")

    print("\n" + "-" * 70)

    while True:
        try:
            # Get user input
            question = input("\n📊 Your question: ").strip()

            if not question:
                continue

            if question.lower() == 'quit':
                break

            if question.lower() == 'stats':
                stats = agent.get_stats()
                print("\n  Agent Statistics:")
                print(f"    • Dialect: {stats['dialect']}")
                print(f"    • Tables: {stats['tables']}")
                print(f"    • Solutions learned: {stats['solutions_stored']}")
                print(f"    • Failures analyzed: {stats['failures_analyzed']}")
                continue

            # Try demo question by number
            if question.isdigit() and 1 <= int(question) <= len(demo_questions):
                question = demo_questions[int(question) - 1]
                print(f"  → {question}")

            # Execute query
            print("  ⏳ Thinking...", end="", flush=True)
            start_time = datetime.now()

            result = await agent.query(question)

            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\r  ⏱ Completed in {elapsed:.1f}s")

            if result["success"]:
                print(f"\n  ✅ SQL Generated:")
                print(f"     {result['sql']}")

                print(f"\n  📊 Results ({result['row_count']} rows):")
                if result["data"]:
                    # Format as table
                    if result["row_count"] <= 10:
                        for row in result["data"]:
                            print(f"     {row}")
                    else:
                        for row in result["data"][:5]:
                            print(f"     {row}")
                        print(f"     ... and {result['row_count'] - 5} more rows")
                else:
                    print("     (No data returned)")

                if result["iterations"] > 1:
                    print(f"\n  🔄 Self-corrected after {result['iterations']} attempts")

            else:
                print(f"\n  ❌ Failed: {result['error']}")
                print("     Try rephrasing your question.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n  ⚠ Error: {e}")

    # =========================================================================
    # CLEANUP
    # =========================================================================
    print("\n" + "=" * 70)
    print("  SESSION SUMMARY")
    print("=" * 70)

    final_stats = agent.get_stats()
    print(f"  • Queries executed: {final_stats['solutions_stored']}")
    print(f"  • Knowledge persisted to: ~/.vanna/meta_agent.json")
    print(f"  • Demo database: ./demo_analytics.db")

    print("\n  Thank you for trying Agentic SQL!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
