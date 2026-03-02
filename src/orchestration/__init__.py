"""
Multi-Agent Orchestrator - Coordinates agents across all 4 semantic layers.

Routes queries to the correct semantic layer and coordinates
cross-layer workflows using AWS Bedrock for LLM reasoning.

Components:
- semantic_router: Routes queries to correct semantic layer
- bedrock_orchestrator: AWS Bedrock multi-agent orchestration
- embedding_service: Unified embedding management across layers
- pipeline: Cross-layer pipeline execution
"""

from .semantic_router import SemanticRouter, QueryClassification, SemanticLayer
from .bedrock_orchestrator import BedrockOrchestrator, BedrockOrchestratorConfig
from .embedding_service import UnifiedEmbeddingService
from .pipeline import SemanticPipeline, PipelineResult

__all__ = [
    "SemanticRouter",
    "QueryClassification",
    "SemanticLayer",
    "BedrockOrchestrator",
    "BedrockOrchestratorConfig",
    "UnifiedEmbeddingService",
    "SemanticPipeline",
    "PipelineResult",
]
