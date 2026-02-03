# AWS Bedrock

Use Claude models through AWS Bedrock.

---

## Overview

AWS Bedrock provides access to Claude models with:

- Enterprise-grade security and compliance
- AWS IAM authentication
- VPC endpoints for private connectivity
- Pay-per-use pricing
- No API key management

---

## Installation

```bash
pip install agentic-sql[bedrock]

# Or install boto3 directly
pip install boto3
```

---

## Configuration

### BedrockConfig

```python
from src.llm.bedrock_client import BedrockClient, BedrockConfig

config = BedrockConfig(
    region_name: str = "us-east-1",           # AWS region
    model: str = "anthropic.claude-3-sonnet-20240229-v1:0",  # Model ID
    aws_access_key_id: Optional[str] = None,  # AWS access key (optional)
    aws_secret_access_key: Optional[str] = None,  # AWS secret key (optional)
    aws_session_token: Optional[str] = None,  # Session token (optional)
    temperature: float = 0.3,                 # Response randomness
    max_tokens: int = 4096,                   # Max response tokens
    top_p: float = 1.0,                       # Nucleus sampling
    top_k: Optional[int] = None,              # Top-k sampling
    stop_sequences: Optional[List[str]] = None,  # Stop sequences
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
| `aws_session_token` | `str` | No | - | Session token for temp credentials |
| `temperature` | `float` | No | `0.3` | Response randomness (0-1) |
| `max_tokens` | `int` | No | `4096` | Max response tokens |
| `top_p` | `float` | No | `1.0` | Nucleus sampling |
| `top_k` | `int` | No | `None` | Top-k sampling |

---

## Available Models

### Claude 3.5 Models

| Model ID | Name | Best For |
|----------|------|----------|
| `anthropic.claude-3-5-sonnet-20241022-v2:0` | Claude 3.5 Sonnet v2 | Best overall performance |
| `anthropic.claude-3-5-sonnet-20240620-v1:0` | Claude 3.5 Sonnet | Balanced quality/speed |
| `anthropic.claude-3-5-haiku-20241022-v1:0` | Claude 3.5 Haiku | Fast, cost-effective |

### Claude 3 Models

| Model ID | Name | Best For |
|----------|------|----------|
| `anthropic.claude-3-opus-20240229-v1:0` | Claude 3 Opus | Complex reasoning |
| `anthropic.claude-3-sonnet-20240229-v1:0` | Claude 3 Sonnet | General use |
| `anthropic.claude-3-haiku-20240307-v1:0` | Claude 3 Haiku | Fast responses |

### Legacy Models

| Model ID | Name |
|----------|------|
| `anthropic.claude-v2:1` | Claude 2.1 |
| `anthropic.claude-v2` | Claude 2.0 |
| `anthropic.claude-instant-v1` | Claude Instant |

---

## Authentication Methods

### IAM Role (Recommended for AWS)

Best for EC2, Lambda, ECS, and other AWS services.

```python
# No credentials needed - uses IAM role automatically
llm = BedrockClient(BedrockConfig(
    region_name="us-east-1",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
))
```

### Environment Variables

```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

```python
# Credentials picked up automatically
llm = BedrockClient(BedrockConfig(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
))
```

### Explicit Credentials

```python
llm = BedrockClient(BedrockConfig(
    region_name="us-east-1",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    aws_access_key_id="AKIA...",
    aws_secret_access_key="...",
))
```

### AWS Profile

```bash
# ~/.aws/credentials
[my-profile]
aws_access_key_id = AKIA...
aws_secret_access_key = ...

# Set profile
export AWS_PROFILE=my-profile
```

### Temporary Credentials (STS)

```python
llm = BedrockClient(BedrockConfig(
    region_name="us-east-1",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    aws_access_key_id="ASIA...",
    aws_secret_access_key="...",
    aws_session_token="...",
))
```

---

## Usage Examples

### Basic Generation

```python
from src.llm.bedrock_client import BedrockClient, BedrockConfig

llm = BedrockClient(BedrockConfig(
    region_name="us-east-1",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
))

response = await llm.generate("Explain quantum computing in simple terms.")
print(response)
```

### With System Prompt

```python
response = await llm.generate(
    prompt="What's the capital of France?",
    system="You are a helpful geography teacher. Give concise answers.",
)
```

### Streaming

```python
async for token in llm.stream("Write a short poem about coding."):
    print(token, end="", flush=True)
```

### Chat Conversation

```python
from src.llm.base import Message, MessageRole

messages = [
    Message(role=MessageRole.USER, content="Hello!"),
    Message(role=MessageRole.ASSISTANT, content="Hi! How can I help you?"),
    Message(role=MessageRole.USER, content="What's 2+2?"),
]

response = await llm.chat(messages)
print(response.content)
```

---

## Using with MetaAgent

```python
from src.llm.bedrock_client import BedrockClient, BedrockConfig
from src.intelligence.meta_agent import MetaAgent
from src.database.multi_db import MSSQLAdapter, ConnectionConfig, DatabaseType

# Setup Bedrock LLM
llm = BedrockClient(BedrockConfig(
    region_name="us-east-1",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    temperature=0.3,
))

# Setup database
db = MSSQLAdapter(ConnectionConfig(
    name="production",
    db_type=DatabaseType.MSSQL,
    host="server.database.windows.net",
    database="MyDB",
    username="user",
    password="password",
))
await db.connect()

# Create agent with Bedrock
agent = MetaAgent(llm_client=llm)
stats = await agent.connect(db_executor=db.execute)

# Query naturally
result = await agent.query("How many orders were placed last month?")
if result["success"]:
    print(result["data"])
```

---

## Bedrock Regions

Bedrock is available in these AWS regions:

| Region | Region Name |
|--------|-------------|
| `us-east-1` | US East (N. Virginia) |
| `us-west-2` | US West (Oregon) |
| `eu-west-1` | Europe (Ireland) |
| `eu-central-1` | Europe (Frankfurt) |
| `ap-southeast-1` | Asia Pacific (Singapore) |
| `ap-northeast-1` | Asia Pacific (Tokyo) |

---

## Cost Tracking

Bedrock pricing is tracked automatically:

```python
# After making requests
if llm.cost_tracker:
    print(f"Total cost: ${llm.cost_tracker.total_cost:.4f}")
    print(f"Requests: {llm.cost_tracker.request_count}")
    print(f"Cost by model: {llm.cost_tracker.costs_by_model}")
```

### Pricing (per 1K tokens)

| Model | Input | Output |
|-------|-------|--------|
| Claude 3.5 Sonnet v2 | $0.003 | $0.015 |
| Claude 3.5 Haiku | $0.0008 | $0.004 |
| Claude 3 Opus | $0.015 | $0.075 |
| Claude 3 Sonnet | $0.003 | $0.015 |
| Claude 3 Haiku | $0.00025 | $0.00125 |

---

## IAM Permissions

Required IAM permissions for Bedrock:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
            ]
        }
    ]
}
```

### Restrictive Policy (specific model)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"
            ]
        }
    ]
}
```

---

## Environment Variables

```bash
# AWS Credentials
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# Or use AWS Profile
AWS_PROFILE=my-profile

# Bedrock-specific (optional)
BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_REGION=us-east-1
```

---

## Error Handling

```python
try:
    response = await llm.generate("Hello")
except Exception as e:
    error_msg = str(e)

    if "AccessDeniedException" in error_msg:
        print("IAM permissions issue - check Bedrock access")
    elif "ResourceNotFoundException" in error_msg:
        print("Model not available in this region")
    elif "ThrottlingException" in error_msg:
        print("Rate limited - retry with backoff")
    elif "ValidationException" in error_msg:
        print("Invalid request - check model ID and parameters")
    else:
        raise
```

---

## Best Practices

1. **Use IAM roles** - Avoid hardcoding credentials
2. **Choose the right model** - Haiku for speed, Opus for complexity
3. **Set appropriate temperature** - Lower (0.1-0.3) for SQL generation
4. **Enable cost tracking** - Monitor usage and costs
5. **Use regional endpoints** - Deploy in same region as your application
6. **Implement retry logic** - Handle transient failures gracefully

---

## Comparison with Direct Anthropic API

| Feature | AWS Bedrock | Anthropic Direct |
|---------|-------------|------------------|
| Authentication | IAM | API Key |
| VPC Support | Yes (PrivateLink) | No |
| Compliance | SOC2, HIPAA, etc. | SOC2 |
| Billing | AWS Consolidated | Separate |
| Model Access | Request required | Immediate |
| Latency | Regional | Global |
