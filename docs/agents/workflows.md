# Workflows

Build complex multi-agent workflows.

---

## Pipeline Builder

```python
from src.core.orchestrator import PipelineBuilder

result = await (
    PipelineBuilder(orchestrator)
    .create("analysis")
    .add("validator")
    .add("sql")
    .add("analyst")
    .run(user_context=user, initial_input="Show sales")
)
```

---

## Parallel Execution

```python
from src.core.orchestrator import Workflow, AgentTask

workflow = Workflow(
    name="parallel",
    tasks=[
        AgentTask(id="q1", agent_name="sql", input_data="Get sales"),
        AgentTask(id="q2", agent_name="sql", input_data="Get orders"),
        AgentTask(
            id="analyze",
            agent_name="analyst",
            dependencies=["q1", "q2"],
        ),
    ],
)
```

---

## Error Handling

```python
task = AgentTask(
    agent_name="sql",
    input_data="Complex query",
    max_retries=3,
    timeout_seconds=60.0,
)
```
