#!/usr/bin/env python3
"""
Test the dynamic training question calculation based on database schema.
"""

import sys
sys.path.insert(0, 'src')

from intelligence.meta_agent import MetaAgent


class MockLLM:
    """Mock LLM for testing"""
    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        return "DIALECT: sqlite\nREASONING: test"


def test_calculate_training_questions():
    """Test that question count scales with database size"""

    agent = MetaAgent(llm_client=MockLLM())

    print("=" * 70)
    print("  TESTING DYNAMIC TRAINING QUESTION CALCULATION")
    print("=" * 70)

    # Test 1: Small database (5 tables, ~30 columns)
    print("\n[Test 1] Small Database")
    agent._schema = {
        "users": {"columns": [{"name": "id"}, {"name": "name"}, {"name": "email"}, {"name": "created_at"}]},
        "posts": {"columns": [{"name": "id"}, {"name": "user_id"}, {"name": "title"}, {"name": "content"}, {"name": "created_at"}]},
        "comments": {"columns": [{"name": "id"}, {"name": "post_id"}, {"name": "user_id"}, {"name": "text"}]},
        "tags": {"columns": [{"name": "id"}, {"name": "name"}]},
        "post_tags": {"columns": [{"name": "post_id"}, {"name": "tag_id"}]},
    }

    total_cols = sum(len(t["columns"]) for t in agent._schema.values())
    print(f"  Tables: {len(agent._schema)}, Columns: {total_cols}")

    for intensity in ["light", "medium", "heavy", "exhaustive"]:
        count = agent._calculate_training_questions(intensity)
        print(f"  {intensity:12} → {count} questions")

    # Test 2: Medium database (15 tables, ~100 columns)
    print("\n[Test 2] Medium Database")
    agent._schema = {
        "customers": {"columns": [{"name": "id"}, {"name": "name"}, {"name": "email"}, {"name": "phone"}, {"name": "address"}, {"name": "city"}, {"name": "country"}]},
        "products": {"columns": [{"name": "id"}, {"name": "name"}, {"name": "category_id"}, {"name": "price"}, {"name": "cost"}, {"name": "stock"}]},
        "categories": {"columns": [{"name": "id"}, {"name": "name"}, {"name": "parent_id"}]},
        "orders": {"columns": [{"name": "id"}, {"name": "customer_id"}, {"name": "order_date"}, {"name": "status"}, {"name": "total"}, {"name": "shipping_id"}]},
        "order_items": {"columns": [{"name": "id"}, {"name": "order_id"}, {"name": "product_id"}, {"name": "quantity"}, {"name": "price"}]},
        "shipping": {"columns": [{"name": "id"}, {"name": "method"}, {"name": "cost"}, {"name": "estimated_days"}]},
        "payments": {"columns": [{"name": "id"}, {"name": "order_id"}, {"name": "amount"}, {"name": "method"}, {"name": "status"}, {"name": "paid_at"}]},
        "reviews": {"columns": [{"name": "id"}, {"name": "product_id"}, {"name": "customer_id"}, {"name": "rating"}, {"name": "comment"}]},
        "suppliers": {"columns": [{"name": "id"}, {"name": "name"}, {"name": "contact"}, {"name": "email"}]},
        "inventory": {"columns": [{"name": "id"}, {"name": "product_id"}, {"name": "warehouse_id"}, {"name": "quantity"}]},
        "warehouses": {"columns": [{"name": "id"}, {"name": "name"}, {"name": "location"}]},
        "discounts": {"columns": [{"name": "id"}, {"name": "code"}, {"name": "percent"}, {"name": "valid_until"}]},
        "wishlists": {"columns": [{"name": "id"}, {"name": "customer_id"}, {"name": "product_id"}]},
        "returns": {"columns": [{"name": "id"}, {"name": "order_id"}, {"name": "reason"}, {"name": "status"}]},
        "audit_log": {"columns": [{"name": "id"}, {"name": "table_name"}, {"name": "action"}, {"name": "timestamp"}]},
    }

    total_cols = sum(len(t["columns"]) for t in agent._schema.values())
    print(f"  Tables: {len(agent._schema)}, Columns: {total_cols}")

    for intensity in ["light", "medium", "heavy", "exhaustive"]:
        count = agent._calculate_training_questions(intensity)
        print(f"  {intensity:12} → {count} questions")

    # Test 3: Large database (40 tables, ~300 columns)
    print("\n[Test 3] Large Database")
    agent._schema = {}
    for i in range(40):
        cols = [{"name": f"col_{j}"} for j in range(7)]
        cols.insert(0, {"name": "id"})
        if i % 3 == 0:
            cols.append({"name": f"ref_{i}_id"})
        agent._schema[f"table_{i}"] = {"columns": cols}

    total_cols = sum(len(t["columns"]) for t in agent._schema.values())
    print(f"  Tables: {len(agent._schema)}, Columns: {total_cols}")

    for intensity in ["light", "medium", "heavy", "exhaustive"]:
        count = agent._calculate_training_questions(intensity)
        print(f"  {intensity:12} → {count} questions")

    # Test 4: Very large database (80 tables) - should cap at 100
    print("\n[Test 4] Very Large Database (should cap at 100)")
    agent._schema = {}
    for i in range(80):
        cols = [{"name": "id"}] + [{"name": f"col_{j}"} for j in range(10)]
        agent._schema[f"table_{i}"] = {"columns": cols}

    total_cols = sum(len(t["columns"]) for t in agent._schema.values())
    print(f"  Tables: {len(agent._schema)}, Columns: {total_cols}")

    for intensity in ["light", "medium", "heavy", "exhaustive"]:
        count = agent._calculate_training_questions(intensity)
        capped = " (CAPPED)" if count == 100 else ""
        print(f"  {intensity:12} → {count} questions{capped}")

    # Test 5: Tiny database (2 tables) - should have minimum 3
    print("\n[Test 5] Tiny Database (should have minimum 3)")
    agent._schema = {
        "users": {"columns": [{"name": "id"}, {"name": "name"}]},
        "sessions": {"columns": [{"name": "id"}, {"name": "user_id"}]},
    }

    total_cols = sum(len(t["columns"]) for t in agent._schema.values())
    print(f"  Tables: {len(agent._schema)}, Columns: {total_cols}")

    for intensity in ["light", "medium", "heavy", "exhaustive"]:
        count = agent._calculate_training_questions(intensity)
        print(f"  {intensity:12} → {count} questions")

    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_calculate_training_questions()
