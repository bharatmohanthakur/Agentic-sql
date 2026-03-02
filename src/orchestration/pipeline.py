"""
Semantic Pipeline - Cross-layer pipeline execution.

Implements the full semantic space funnel pipeline:
Ontology → Knowledge Graph → Canonical → Dimensional

Passes enriched context from each layer to the next.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .semantic_router import QueryClassification, SemanticLayer

logger = logging.getLogger(__name__)


class PipelineStepResult(BaseModel):
    """Result from a single pipeline step"""
    layer: SemanticLayer
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Complete result from a semantic pipeline execution"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str = ""
    classification: Optional[QueryClassification] = None
    steps: List[PipelineStepResult] = Field(default_factory=list)
    final_output: Optional[Any] = None
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    success: bool = True
    total_duration_ms: float = 0.0
    layers_used: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class SemanticPipeline:
    """
    Executes the full semantic space funnel pipeline.

    The pipeline flows top-down through the 4 layers:

    1. Ontology Agent → resolves concepts, identifies fact types
    2. KG Agent → finds entity relationships, generates graph context
    3. Schema Agent → validates canonical mappings, transforms data
    4. SQL Agent → generates optimized Snowflake SQL

    Each layer enriches the context for the next layer.
    """

    def __init__(self):
        self._agents: Dict[SemanticLayer, Any] = {}

    def register_agent(self, layer: SemanticLayer, agent: Any) -> None:
        """Register an agent for a semantic layer"""
        self._agents[layer] = agent

    async def execute(
        self,
        question: str,
        classification: QueryClassification,
        user_context: Optional[Any] = None,
    ) -> PipelineResult:
        """
        Execute the semantic pipeline.

        Routes through the appropriate layers based on classification.
        """
        result = PipelineResult(
            question=question,
            classification=classification,
        )

        start_time = datetime.utcnow()

        # Determine execution order (top-down through funnel)
        layers_to_execute = self._get_execution_order(classification)
        accumulated_context: Dict[str, Any] = {
            "question": question,
            "semantic_layer": "pipeline",
        }

        for layer in layers_to_execute:
            agent = self._agents.get(layer)
            if not agent:
                logger.warning(f"No agent for {layer.value}, skipping")
                continue

            step_start = datetime.utcnow()

            try:
                # Execute agent with accumulated context
                agent_result = await agent.execute(
                    accumulated_context,
                    user_context,
                )

                step_duration = (datetime.utcnow() - step_start).total_seconds() * 1000

                if agent_result.success and agent_result.data:
                    step = PipelineStepResult(
                        layer=layer,
                        success=True,
                        data=agent_result.data,
                        duration_ms=step_duration,
                    )
                    result.steps.append(step)
                    result.layers_used.append(layer.value)

                    # Merge context for next layer
                    if isinstance(agent_result.data, dict):
                        accumulated_context.update(agent_result.data)

                        # Capture SQL and data
                        if "sql" in agent_result.data:
                            result.sql = agent_result.data["sql"]
                        if "data" in agent_result.data:
                            result.data = agent_result.data["data"]
                else:
                    step = PipelineStepResult(
                        layer=layer,
                        success=False,
                        error=agent_result.error,
                        duration_ms=step_duration,
                    )
                    result.steps.append(step)

            except Exception as e:
                step_duration = (datetime.utcnow() - step_start).total_seconds() * 1000
                result.steps.append(PipelineStepResult(
                    layer=layer,
                    success=False,
                    error=str(e),
                    duration_ms=step_duration,
                ))
                logger.exception(f"Pipeline step {layer.value} failed: {e}")

        result.final_output = accumulated_context
        result.total_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        result.success = any(s.success for s in result.steps)

        return result

    def _get_execution_order(
        self,
        classification: QueryClassification,
    ) -> List[SemanticLayer]:
        """
        Determine the order of layer execution.

        For cross-layer queries, always flows top-down:
        Ontology → KG → Canonical → Dimensional
        """
        layer_priority = {
            SemanticLayer.ONTOLOGY: 0,
            SemanticLayer.KNOWLEDGE_GRAPH: 1,
            SemanticLayer.CANONICAL: 2,
            SemanticLayer.DIMENSIONAL: 3,
        }

        if classification.requires_cross_layer:
            all_layers = [classification.primary_layer] + classification.secondary_layers
            return sorted(set(all_layers), key=lambda l: layer_priority[l])
        else:
            return [classification.primary_layer]
