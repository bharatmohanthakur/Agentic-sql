# LLM Clients

Supported LLM providers and their configuration.

---

## Overview

Agentic SQL supports multiple LLM providers:

| Provider | Class | Status |
|----------|-------|--------|
| Azure OpenAI | `AzureOpenAIClient` | :material-check-circle: Ready |
| OpenAI | `OpenAIClient` | :material-check-circle: Ready |
| Anthropic | `AnthropicClient` | :material-check-circle: Ready |
| AWS Bedrock | `BedrockClient` | :material-check-circle: Ready |
| Google | `GoogleClient` | :material-clock: Coming |
| Ollama | `OllamaClient` | :material-clock: Coming |

---

## AzureOpenAIClient

### Configuration

```python
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig

config = AzureOpenAIConfig(
    api_key: str,                    # Required
    azure_endpoint: str,             # Required
    azure_deployment: str,           # Required
    api_version: str = "2024-02-01", # Optional
    temperature: float = 0.3,        # Optional
    max_tokens: int = 2000,          # Optional
)

llm = AzureOpenAIClient(config)
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key` | `str` | Yes | - | Azure OpenAI API key |
| `azure_endpoint` | `str` | Yes | - | Azure endpoint URL |
| `azure_deployment` | `str` | Yes | - | Deployment name (e.g., "gpt-4o") |
| `api_version` | `str` | No | `"2024-02-01"` | API version |
| `temperature` | `float` | No | `0.3` | Response randomness (0-1) |
| `max_tokens` | `int` | No | `2000` | Max response tokens |

### Example

```python
from src.llm.azure_openai_client import AzureOpenAIClient, AzureOpenAIConfig
import os

llm = AzureOpenAIClient(AzureOpenAIConfig(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-4o",
))

# Use with MetaAgent
agent = MetaAgent(llm_client=llm)
```

---

## OpenAIClient

### Configuration

```python
from src.llm.openai_client import OpenAIClient, OpenAIConfig

config = OpenAIConfig(
    api_key: str,              # Required
    model: str = "gpt-4",      # Optional
    temperature: float = 0.3,  # Optional
    max_tokens: int = 2000,    # Optional
    organization: str = None,  # Optional
)

llm = OpenAIClient(config)
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key` | `str` | Yes | - | OpenAI API key |
| `model` | `str` | No | `"gpt-4"` | Model name |
| `temperature` | `float` | No | `0.3` | Response randomness |
| `max_tokens` | `int` | No | `2000` | Max response tokens |
| `organization` | `str` | No | `None` | Organization ID |

### Available Models

| Model | Best For |
|-------|----------|
| `gpt-4` | Complex queries |
| `gpt-4-turbo` | Faster, large context |
| `gpt-4o` | Optimized performance |
| `gpt-3.5-turbo` | Simple queries, lower cost |

### Example

```python
from src.llm.openai_client import OpenAIClient, OpenAIConfig

llm = OpenAIClient(OpenAIConfig(
    api_key="sk-...",
    model="gpt-4",
    temperature=0.3,
))

agent = MetaAgent(llm_client=llm)
```

---

## AnthropicClient

### Configuration

```python
from src.llm.anthropic_client import AnthropicClient, AnthropicConfig

config = AnthropicConfig(
    api_key: str,                           # Required
    model: str = "claude-3-opus-20240229",  # Optional
    temperature: float = 0.3,               # Optional
    max_tokens: int = 2000,                 # Optional
)

llm = AnthropicClient(config)
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `api_key` | `str` | Yes | - | Anthropic API key |
| `model` | `str` | No | `"claude-3-opus-20240229"` | Model name |
| `temperature` | `float` | No | `0.3` | Response randomness |
| `max_tokens` | `int` | No | `2000` | Max response tokens |

### Available Models

| Model | Best For |
|-------|----------|
| `claude-3-opus-20240229` | Best quality |
| `claude-3-sonnet-20240229` | Balanced |
| `claude-3-haiku-20240307` | Fast, low cost |

### Example

```python
from src.llm.anthropic_client import AnthropicClient, AnthropicConfig

llm = AnthropicClient(AnthropicConfig(
    api_key="sk-ant-...",
    model="claude-3-opus-20240229",
))

agent = MetaAgent(llm_client=llm)
```

---

## BedrockClient (AWS)

### Configuration

```python
from src.llm.bedrock_client import BedrockClient, BedrockConfig

config = BedrockConfig(
    region_name: str = "us-east-1",           # AWS region
    model: str = "anthropic.claude-3-sonnet-20240229-v1:0",  # Model ID
    aws_access_key_id: Optional[str] = None,  # AWS access key (optional)
    aws_secret_access_key: Optional[str] = None,  # AWS secret (optional)
    temperature: float = 0.3,                 # Optional
    max_tokens: int = 4096,                   # Optional
)

llm = BedrockClient(config)
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `region_name` | `str` | No | `"us-east-1"` | AWS region |
| `model` | `str` | No | `"anthropic.claude-3-sonnet..."` | Bedrock model ID |
| `aws_access_key_id` | `str` | No | - | AWS access key |
| `aws_secret_access_key` | `str` | No | - | AWS secret key |
| `temperature` | `float` | No | `0.3` | Response randomness |
| `max_tokens` | `int` | No | `4096` | Max response tokens |

### Available Models

| Model ID | Best For |
|----------|----------|
| `anthropic.claude-3-5-sonnet-20241022-v2:0` | Best overall |
| `anthropic.claude-3-5-haiku-20241022-v1:0` | Fast, cost-effective |
| `anthropic.claude-3-opus-20240229-v1:0` | Complex reasoning |
| `anthropic.claude-3-sonnet-20240229-v1:0` | General use |
| `anthropic.claude-3-haiku-20240307-v1:0` | Fast responses |

### Example (IAM Role)

```python
from src.llm.bedrock_client import BedrockClient, BedrockConfig

# Uses IAM role automatically (recommended for AWS)
llm = BedrockClient(BedrockConfig(
    region_name="us-east-1",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
))

agent = MetaAgent(llm_client=llm)
```

### Example (Explicit Credentials)

```python
from src.llm.bedrock_client import BedrockClient, BedrockConfig
import os

llm = BedrockClient(BedrockConfig(
    region_name="us-east-1",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
))

agent = MetaAgent(llm_client=llm)
```

See [AWS Bedrock Documentation](../llm/bedrock.md) for more details.

---

## LLM Interface

All LLM clients implement this interface:

```python
class LLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = None,
    ) -> str:
        """Generate text from prompt."""
        pass
```

---

## Custom LLM Client

Create your own LLM client:

```python
from src.llm.base import LLMClient

class MyCustomLLM(LLMClient):
    def __init__(self, config):
        self.config = config

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = None,
    ) -> str:
        # Your implementation
        response = await my_llm_api.call(prompt)
        return response.text

# Use with MetaAgent
llm = MyCustomLLM(config)
agent = MetaAgent(llm_client=llm)
```

---

## Best Practices

### Temperature

| Use Case | Temperature |
|----------|-------------|
| SQL generation | 0.1 - 0.3 |
| Analysis | 0.3 - 0.5 |
| Creative | 0.7 - 1.0 |

!!! tip "Recommendation"
    Use `temperature=0.3` for SQL generation. Lower values produce more consistent, deterministic output.

### Token Limits

| Task | Recommended `max_tokens` |
|------|--------------------------|
| Simple query | 500 |
| Complex query | 1000 |
| Auto-learning | 2000 |

### Error Handling

```python
try:
    result = await agent.query("...")
except Exception as e:
    if "rate_limit" in str(e).lower():
        # Wait and retry
        await asyncio.sleep(60)
        result = await agent.query("...")
    else:
        raise
```
