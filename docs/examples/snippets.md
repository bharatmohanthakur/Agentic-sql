# Code Snippets

Copy-paste ready code examples.

---

## Basic Query

```python
import asyncio
from src.intelligence.meta_agent import MetaAgent
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

async def main():
    llm = AzureOpenAIClient(AzureOpenAIConfig(
        api_key="your-key",
        azure_endpoint="https://your-endpoint.openai.azure.com",
        azure_deployment="gpt-4o",
    ))

    agent = MetaAgent(llm_client=llm)
    await agent.connect(db_executor=db.execute)

    result = await agent.query("How many users?")
    print(result["data"])

asyncio.run(main())
```

---

## With Auto-Learning

```python
await agent.connect(db_executor=db.execute)
await agent.auto_learn(intensity="medium")
result = await agent.query("Show top customers")
```

---

## API Server

```python
from src.api.server import create_app, APIConfig

app = create_app(
    config=APIConfig(host="0.0.0.0", port=8000),
    sql_agent=agent,
)
```

---

## Error Handling

```python
result = await agent.query("...")

if result["success"]:
    print(result["data"])
else:
    print(f"Error: {result['error']}")
```
