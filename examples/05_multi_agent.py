#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 5: MULTI-AGENT WORKFLOWS
=============================================================================

This tutorial shows how to create complex workflows using multiple agents:
- SQLAgent: Generates and executes SQL
- AnalystAgent: Analyzes results and provides insights
- ValidatorAgent: Validates queries for security

Agents can be:
- Chained sequentially (A → B → C)
- Run in parallel (A + B → C)
- Conditional execution (if A then B else C)

Run: python examples/05_multi_agent.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


async def main():
    print("=" * 60)
    print("  TUTORIAL 5: MULTI-AGENT WORKFLOWS")
    print("=" * 60)

    # =========================================================================
    # UNDERSTANDING THE AGENTS
    # =========================================================================
    print("\n[Overview] Available Agents")

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                      AGENT TYPES                             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
    │  │  SQL Agent   │   │   Analyst    │   │  Validator   │    │
    │  │              │   │    Agent     │   │    Agent     │    │
    │  │ • ReAct loop │   │              │   │              │    │
    │  │ • SQL gen    │   │ • Insights   │   │ • Security   │    │
    │  │ • Execution  │   │ • Summaries  │   │ • Validation │    │
    │  │ • Reflection │   │ • Trends     │   │ • RLS check  │    │
    │  └──────────────┘   └──────────────┘   └──────────────┘    │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    """)

    # =========================================================================
    # SETUP AGENTS
    # =========================================================================
    print("[Step 1] Setting up agents...")

    from core.orchestrator import AgentOrchestrator, Workflow, AgentTask
    from core.base import UserContext, AgentConfig

    # Create orchestrator
    orchestrator = AgentOrchestrator()

    # For this demo, we'll create mock agents
    # In real usage, you'd import the actual agent classes

    print("""
    # Real usage:
    from agents.sql_agent import SQLAgent, SQLAgentConfig
    from agents.analyst_agent import AnalystAgent, AnalystAgentConfig
    from agents.validator_agent import ValidatorAgent, ValidatorAgentConfig

    # Create SQL Agent
    sql_agent = SQLAgent(
        config=SQLAgentConfig(
            name="sql_agent",
            max_sql_retries=3,
            enable_query_validation=True,
        ),
        llm_client=llm,
        db_executor=db.execute,
    )

    # Create Analyst Agent
    analyst_agent = AnalystAgent(
        config=AnalystAgentConfig(
            name="analyst_agent",
            enable_trend_analysis=True,
        ),
        llm_client=llm,
    )

    # Create Validator Agent
    validator_agent = ValidatorAgent(
        config=ValidatorAgentConfig(
            name="validator_agent",
            block_destructive_queries=True,
        ),
    )

    # Register with orchestrator
    orchestrator.register_agent("sql", sql_agent)
    orchestrator.register_agent("analyst", analyst_agent)
    orchestrator.register_agent("validator", validator_agent)
    """)

    # =========================================================================
    # WORKFLOW 1: SEQUENTIAL PIPELINE
    # =========================================================================
    print("\n[Workflow 1] Sequential Pipeline")

    print("""
    A simple chain: Validate → SQL → Analyze

    ┌────────────┐     ┌────────────┐     ┌────────────┐
    │  Validator │────▶│    SQL     │────▶│  Analyst   │
    │            │     │            │     │            │
    │ Check      │     │ Generate   │     │ Analyze    │
    │ security   │     │ & execute  │     │ results    │
    └────────────┘     └────────────┘     └────────────┘

    Code:
    ```python
    from core.orchestrator import PipelineBuilder

    result = await (
        PipelineBuilder(orchestrator)
        .create("analysis_pipeline")
        .add("validator")    # Step 1
        .add("sql")          # Step 2 (after validator)
        .add("analyst")      # Step 3 (after sql)
        .run(
            user_context=user,
            initial_input="Show sales trends for Q4"
        )
    )
    ```
    """)

    # =========================================================================
    # WORKFLOW 2: PARALLEL EXECUTION
    # =========================================================================
    print("[Workflow 2] Parallel Execution")

    print("""
    Run multiple queries in parallel, then combine:

         ┌────────────┐
         │   SQL 1    │──┐
         │ (sales)    │  │
         └────────────┘  │     ┌────────────┐
                         ├────▶│  Analyst   │
         ┌────────────┐  │     │ (combine)  │
         │   SQL 2    │──┘     └────────────┘
         │ (orders)   │
         └────────────┘

    Code:
    ```python
    workflow = Workflow(
        name="parallel_analysis",
        tasks=[
            AgentTask(
                id="sales_query",
                agent_name="sql",
                input_data="Get total sales by month",
            ),
            AgentTask(
                id="orders_query",
                agent_name="sql",
                input_data="Get order count by month",
            ),
            AgentTask(
                id="analysis",
                agent_name="analyst",
                dependencies=["sales_query", "orders_query"],
                input_data="Compare sales vs orders trends",
            ),
        ],
    )

    result = await orchestrator.execute_workflow(workflow, user_context)
    ```
    """)

    # =========================================================================
    # WORKFLOW 3: CONDITIONAL EXECUTION
    # =========================================================================
    print("[Workflow 3] Conditional Execution")

    print("""
    Execute different paths based on conditions:

                        ┌────────────┐
                   ┌───▶│  Analyst   │  (if data found)
    ┌────────────┐ │    │ (insights) │
    │    SQL     │─┤    └────────────┘
    │  (query)   │ │
    └────────────┘ │    ┌────────────┐
                   └───▶│  Handler   │  (if no data)
                        │ (no data)  │
                        └────────────┘

    Code:
    ```python
    async def conditional_workflow(question: str, user: UserContext):
        # Step 1: Run SQL
        sql_result = await sql_agent.execute(question, user)

        # Step 2: Check result
        if sql_result.data and len(sql_result.data) > 0:
            # Has data - run analysis
            return await analyst_agent.execute(
                f"Analyze: {sql_result.data}",
                user,
            )
        else:
            # No data - return helpful message
            return {
                "success": True,
                "message": "No data found for your query",
                "suggestions": ["Try different date range", "Check filters"],
            }
    ```
    """)

    # =========================================================================
    # WORKFLOW 4: ERROR HANDLING & RETRIES
    # =========================================================================
    print("[Workflow 4] Error Handling & Retries")

    print("""
    Built-in retry logic and error handling:

    ┌────────────┐     ┌────────────┐
    │    SQL     │────▶│  Retry?    │──┐
    │  (fails)   │     │            │  │ (max 3x)
    └────────────┘     └────────────┘  │
                             │         │
                             ▼         │
                       ┌────────────┐  │
                       │    SQL     │◀─┘
                       │  (retry)   │
                       └────────────┘

    Code:
    ```python
    task = AgentTask(
        agent_name="sql",
        input_data="Complex query here",
        max_retries=3,          # Retry up to 3 times
        timeout_seconds=60.0,   # Timeout per attempt
    )
    ```

    The orchestrator automatically:
    - Catches exceptions
    - Applies exponential backoff
    - Tracks retry attempts
    - Fails gracefully after max retries
    """)

    # =========================================================================
    # WORKFLOW 5: REAL-TIME PROGRESS
    # =========================================================================
    print("[Workflow 5] Real-Time Progress Tracking")

    print("""
    Track workflow progress in real-time:

    ```python
    async def run_with_progress():
        workflow = create_workflow()

        # Subscribe to progress updates
        async for event in orchestrator.execute_with_progress(workflow, user):
            if event.type == "task_started":
                print(f"Started: {event.task_id}")
            elif event.type == "task_completed":
                print(f"Completed: {event.task_id} in {event.duration}ms")
            elif event.type == "task_failed":
                print(f"Failed: {event.task_id} - {event.error}")
            elif event.type == "workflow_completed":
                print(f"Workflow done! Results: {event.results}")
    ```
    """)

    # =========================================================================
    # FULL EXAMPLE
    # =========================================================================
    print("\n[Full Example] Complete Multi-Agent Setup")

    print("""
    ```python
    import asyncio
    from core.orchestrator import AgentOrchestrator, PipelineBuilder
    from core.base import UserContext
    from agents.sql_agent import SQLAgent, SQLAgentConfig
    from agents.analyst_agent import AnalystAgent

    async def main():
        # Setup
        llm = create_llm_client()
        db = await connect_database()

        # Create agents
        sql_agent = SQLAgent(
            config=SQLAgentConfig(name="sql"),
            llm_client=llm,
            db_executor=db.execute,
        )

        analyst_agent = AnalystAgent(
            config={"name": "analyst"},
            llm_client=llm,
        )

        # Create orchestrator
        orchestrator = AgentOrchestrator()
        orchestrator.register_agent("sql", sql_agent)
        orchestrator.register_agent("analyst", analyst_agent)

        # Define user
        user = UserContext(
            user_id="user123",
            roles=["analyst"],
        )

        # Run pipeline
        result = await (
            PipelineBuilder(orchestrator)
            .create("sales_analysis")
            .add("sql", input="Get monthly sales for 2024")
            .add("analyst", input="Identify trends and anomalies")
            .run(user_context=user)
        )

        print(f"Analysis: {result.final_output}")

    asyncio.run(main())
    ```
    """)

    print("\n" + "=" * 60)
    print("  TUTORIAL COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
