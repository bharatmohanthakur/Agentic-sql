"""
Bedrock Orchestrator - AWS Bedrock multi-agent orchestration.

Implements the SUPERVISOR_ROUTER pattern for coordinating
specialist agents across all 4 semantic layers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .semantic_router import QueryClassification, SemanticLayer, SemanticRouter

logger = logging.getLogger(__name__)


@dataclass
class BedrockOrchestratorConfig:
    """Configuration for the Bedrock orchestrator"""
    # AWS settings
    region_name: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # Orchestrator model
    supervisor_model: str = "anthropic.claude-sonnet-4-20250514-v1:0"
    specialist_model: str = "anthropic.claude-3-5-haiku-20241022-v1:0"

    # Agent settings
    max_iterations: int = 10
    session_ttl_seconds: int = 1800

    # Managed agent IDs (populated after agent creation)
    supervisor_agent_id: Optional[str] = None
    supervisor_alias_id: Optional[str] = None


class AgentResponse(BaseModel):
    """Response from a multi-agent invocation"""
    output: str = ""
    layer_results: Dict[str, Any] = Field(default_factory=dict)
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    trace: Dict[str, Any] = Field(default_factory=dict)
    session_id: str = ""
    layers_used: List[str] = Field(default_factory=list)


class BedrockOrchestrator:
    """
    Multi-agent orchestrator using AWS Bedrock.

    Implements SUPERVISOR_ROUTER pattern:
    - Simple queries → route directly to one specialist agent
    - Complex queries → supervisor coordinates multiple agents

    Agent Roles:
    1. Ontology Agent: OWL/RDF reasoning, concept resolution
    2. Knowledge Graph Agent: Neo4j Cypher, relationship traversal
    3. Schema Agent: Canonical model validation, schema mapping
    4. SQL Agent: Snowflake dimensional SQL generation

    Can operate in two modes:
    - Managed mode: Uses pre-created Bedrock Agents (production)
    - Local mode: Uses local agents with Bedrock LLM (development)
    """

    def __init__(self, config: BedrockOrchestratorConfig):
        self.config = config
        self.router = SemanticRouter()
        self._agents: Dict[SemanticLayer, Any] = {}
        self._bedrock_client = None

    def register_agent(self, layer: SemanticLayer, agent: Any) -> None:
        """Register a local agent for a semantic layer"""
        self._agents[layer] = agent
        logger.info(f"Registered agent for layer: {layer.value}")

    async def query(
        self,
        question: str,
        user_context: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> AgentResponse:
        """
        Process a natural language query through the semantic funnel.

        1. Classify the query (determine which layers)
        2. Route to appropriate agent(s)
        3. Coordinate cross-layer execution if needed
        4. Return consolidated results
        """
        session_id = session_id or str(uuid4())

        # Step 1: Classify
        classification = await self.router.classify(question)
        logger.info(
            f"Query classified: primary={classification.primary_layer.value}, "
            f"cross_layer={classification.requires_cross_layer}"
        )

        # Step 2: Execute
        if classification.requires_cross_layer:
            return await self._execute_cross_layer(
                question, classification, user_context, session_id
            )
        else:
            return await self._execute_single_layer(
                question, classification, user_context, session_id
            )

    async def _execute_single_layer(
        self,
        question: str,
        classification: QueryClassification,
        user_context: Optional[Any],
        session_id: str,
    ) -> AgentResponse:
        """Route directly to a single agent"""
        layer = classification.primary_layer
        agent = self._agents.get(layer)

        if not agent:
            return AgentResponse(
                output=f"No agent registered for layer: {layer.value}",
                session_id=session_id,
            )

        result = await agent.execute(question, user_context)

        return AgentResponse(
            output=str(result.data) if result.success else str(result.error),
            layer_results={layer.value: result.data},
            sql=result.data.get("sql") if isinstance(result.data, dict) else None,
            data=result.data.get("data") if isinstance(result.data, dict) else None,
            session_id=session_id,
            layers_used=[layer.value],
        )

    async def _execute_cross_layer(
        self,
        question: str,
        classification: QueryClassification,
        user_context: Optional[Any],
        session_id: str,
    ) -> AgentResponse:
        """
        Coordinate execution across multiple semantic layers.

        Executes the full pipeline: Ontology → KG → Schema → SQL
        passing context from each layer to the next.
        """
        response = AgentResponse(session_id=session_id)
        current_context: Dict[str, Any] = {"question": question}

        # Determine layer execution order (top-down through funnel)
        all_layers = [classification.primary_layer] + classification.secondary_layers
        layer_order = sorted(
            all_layers,
            key=lambda l: list(SemanticLayer).index(l),
        )

        for layer in layer_order:
            agent = self._agents.get(layer)
            if not agent:
                logger.warning(f"No agent for layer {layer.value}, skipping")
                continue

            logger.info(f"Executing layer: {layer.value}")

            # Pass accumulated context to agent
            result = await agent.execute(current_context, user_context)

            if result.success and result.data:
                response.layer_results[layer.value] = result.data
                response.layers_used.append(layer.value)

                # Merge result into context for next layer
                if isinstance(result.data, dict):
                    current_context.update(result.data)

                    # Extract SQL and data if available
                    if "sql" in result.data:
                        response.sql = result.data["sql"]
                    if "data" in result.data:
                        response.data = result.data["data"]
            else:
                logger.warning(
                    f"Layer {layer.value} failed: {result.error}"
                )

        # Build final output
        response.output = self._build_output(question, response)
        return response

    async def query_managed(
        self,
        question: str,
        session_id: Optional[str] = None,
    ) -> AgentResponse:
        """
        Query using managed Bedrock Agents (production mode).

        Invokes the pre-created supervisor agent which handles
        routing and coordination natively.
        """
        if not self.config.supervisor_agent_id:
            raise ValueError("Supervisor agent ID not configured")

        client = await self._get_bedrock_client()
        session_id = session_id or str(uuid4())

        loop = asyncio.get_event_loop()

        def _invoke():
            return client.invoke_agent(
                agentId=self.config.supervisor_agent_id,
                agentAliasId=self.config.supervisor_alias_id,
                sessionId=session_id,
                inputText=question,
                enableTrace=True,
            )

        result = await loop.run_in_executor(None, _invoke)

        # Collect streamed response
        output_parts = []
        traces = []

        for event in result.get("completion", []):
            if "chunk" in event:
                chunk = event["chunk"]
                output_parts.append(chunk["bytes"].decode("utf-8"))
            if "trace" in event:
                traces.append(event["trace"])

        return AgentResponse(
            output="".join(output_parts),
            trace={"steps": traces},
            session_id=session_id,
        )

    async def _get_bedrock_client(self):
        """Lazy-load Bedrock agent runtime client"""
        if self._bedrock_client is None:
            try:
                import boto3
                from botocore.config import Config

                credentials = {}
                if self.config.aws_access_key_id:
                    credentials["aws_access_key_id"] = self.config.aws_access_key_id
                if self.config.aws_secret_access_key:
                    credentials["aws_secret_access_key"] = self.config.aws_secret_access_key

                self._bedrock_client = boto3.client(
                    "bedrock-agent-runtime",
                    config=Config(region_name=self.config.region_name),
                    **credentials,
                )
            except ImportError:
                raise ImportError("boto3 required: pip install boto3")

        return self._bedrock_client

    def _build_output(
        self,
        question: str,
        response: AgentResponse,
    ) -> str:
        """Build a consolidated output from multi-layer results"""
        parts = [f"Query: {question}", f"Layers used: {', '.join(response.layers_used)}"]

        if response.sql:
            parts.append(f"\nGenerated SQL:\n{response.sql}")

        if response.data:
            row_count = len(response.data)
            parts.append(f"\nResults: {row_count} rows returned")

        return "\n".join(parts)
