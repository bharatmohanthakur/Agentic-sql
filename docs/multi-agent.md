---
layout: default
title: Multi-Agent Workflows - Agentic SQL
---

# Multi-Agent Workflows

Build complex workflows with multiple specialized agents.

---

## Agent Types

| Agent | Purpose | Tools |
|-------|---------|-------|
| `SQLAgent` | Converts natural language to SQL | ReAct loop, SQL generation, execution |
| `AnalystAgent` | Analyzes results and provides insights | Trend analysis, summaries |
| `ValidatorAgent` | Validates queries for security | SQL injection detection, RLS |

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENT TYPES                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │  SQL Agent   │   │   Analyst    │   │  Validator   │    │
│  │              │   │    Agent     │   │    Agent     │    │
│  │ • ReAct loop │   │ • Insights   │   │ • Security   │    │
│  │ • SQL gen    │   │ • Summaries  │   │ • Validation │    │
│  │ • Execution  │   │ • Trends     │   │ • RLS check  │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Setting Up Agents

```python
from src.core.orchestrator import AgentOrchestrator
from src.agents.sql_agent import SQLAgent, SQLAgentConfig
from src.agents.analyst_agent import AnalystAgent, AnalystAgentConfig
from src.agents.validator_agent import ValidatorAgent, ValidatorAgentConfig

# Create orchestrator
orchestrator = AgentOrchestrator()

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
```

---

## Workflow Patterns

### 1. Sequential Pipeline

Agents execute one after another:

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│  Validator │────▶│    SQL     │────▶│  Analyst   │
│            │     │            │     │            │
│ Check      │     │ Generate   │     │ Analyze    │
│ security   │     │ & execute  │     │ results    │
└────────────┘     └────────────┘     └────────────┘
```

```python
from src.core.orchestrator import PipelineBuilder

result = await (
    PipelineBuilder(orchestrator)
    .create("analysis_pipeline")
    .add("validator")    # Step 1
    .add("sql")          # Step 2
    .add("analyst")      # Step 3
    .run(
        user_context=user,
        initial_input="Show sales trends for Q4"
    )
)
```

### 2. Parallel Execution

Independent tasks run simultaneously:

```
     ┌────────────┐
     │   SQL 1    │──┐
     │ (sales)    │  │
     └────────────┘  │     ┌────────────┐
                     ├────▶│  Analyst   │
     ┌────────────┐  │     │ (combine)  │
     │   SQL 2    │──┘     └────────────┘
     │ (orders)   │
     └────────────┘
```

```python
from src.core.orchestrator import Workflow, AgentTask

workflow = Workflow(
    name="parallel_analysis",
    tasks=[
        # These two run in parallel
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
        # This runs after both complete
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

### 3. Conditional Execution

Different paths based on results:

```
                    ┌────────────┐
               ┌───▶│  Analyst   │  (if data found)
┌────────────┐ │    │ (insights) │
│    SQL     │─┤    └────────────┘
│  (query)   │ │
└────────────┘ │    ┌────────────┐
               └───▶│  Handler   │  (if no data)
                    │ (no data)  │
                    └────────────┘
```

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

---

## Error Handling & Retries

Built-in retry logic:

```
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
```

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

---

## Progress Tracking

Track workflow progress in real-time:

```python
async def run_with_progress():
    workflow = create_workflow()

    async for event in orchestrator.execute_with_progress(workflow, user):
        if event.type == "task_started":
            print(f"Started: {event.task_id}")
        elif event.type == "task_completed":
            print(f"Completed: {event.task_id} in {event.duration}ms")
        elif event.type == "task_failed":
            print(f"Failed: {event.task_id} - {event.error}")
        elif event.type == "workflow_completed":
            print(f"Done! Results: {event.results}")
```

---

## Complete Example

```python
import asyncio
from src.core.orchestrator import AgentOrchestrator, PipelineBuilder
from src.core.base import UserContext
from src.agents.sql_agent import SQLAgent, SQLAgentConfig
from src.agents.analyst_agent import AnalystAgent

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

---

## Agent Communication

Agents can pass data between each other:

```python
workflow = Workflow(
    name="data_pipeline",
    tasks=[
        AgentTask(
            id="fetch",
            agent_name="sql",
            input_data="Get raw data",
            output_key="raw_data",  # Store output with this key
        ),
        AgentTask(
            id="process",
            agent_name="analyst",
            dependencies=["fetch"],
            input_template="Analyze this data: {raw_data}",  # Use previous output
        ),
    ],
)
```

---

## Custom Agents

Create your own agents:

```python
from src.core.base import BaseAgent, AgentConfig, AgentResult

class MyCustomAgent(BaseAgent):
    def __init__(self, config: AgentConfig, llm_client):
        super().__init__(config)
        self.llm = llm_client

    async def execute(self, input_data: str, user: UserContext) -> AgentResult:
        # Your custom logic here
        response = await self.llm.generate(prompt=input_data)

        return AgentResult(
            success=True,
            output=response,
            metadata={"custom": "data"},
        )

# Register custom agent
orchestrator.register_agent("custom", MyCustomAgent(config, llm))
```

---

<p align="center">
  <a href="examples">Examples →</a>
</p>
