"""
Semantic Router - Routes queries to the correct semantic layer.

Classifies incoming natural language queries and determines which
layer(s) of the semantic space funnel should handle them.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SemanticLayer(str, Enum):
    """The 4 semantic space layers"""
    ONTOLOGY = "ontology"              # Layer 1: Highest semantic space
    KNOWLEDGE_GRAPH = "knowledge_graph"  # Layer 2: High semantic space
    CANONICAL = "canonical"            # Layer 3: Medium semantic space
    DIMENSIONAL = "dimensional"        # Layer 4: Lowest semantic space


class QueryClassification(BaseModel):
    """Classification of a query by semantic layer(s)"""
    primary_layer: SemanticLayer
    secondary_layers: List[SemanticLayer] = Field(default_factory=list)
    confidence: float = 0.8
    reasoning: str = ""
    requires_cross_layer: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticRouter:
    """
    Routes queries to the appropriate semantic layer.

    Uses a combination of heuristics and LLM-based classification
    to determine which layer(s) a query requires.

    Routing logic:
    - Concept/definition queries → Ontology Agent
    - Relationship/path queries → Knowledge Graph Agent
    - Schema/mapping queries → Schema Agent (Canonical)
    - Data/metric queries → Dimensional SQL Agent
    - Complex queries → Full pipeline (all layers)
    """

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    async def classify(self, question: str) -> QueryClassification:
        """
        Classify a question to determine which semantic layer(s) to use.
        """
        # First try heuristic classification
        classification = self._heuristic_classify(question)
        if classification.confidence >= 0.8:
            return classification

        # Fall back to LLM classification
        if self.llm_client:
            return await self._llm_classify(question)

        return classification

    def _heuristic_classify(self, question: str) -> QueryClassification:
        """Fast heuristic-based classification"""
        q = question.lower()

        # Ontology layer indicators
        ontology_signals = [
            "what is", "what does", "define", "meaning of",
            "concept", "ontology", "fact type", "business rule",
        ]

        # Knowledge graph indicators
        kg_signals = [
            "how does", "relate to", "connected", "relationship",
            "path between", "linked", "graph", "network",
        ]

        # Canonical model indicators
        canonical_signals = [
            "schema", "model", "canonical", "mapping",
            "transform", "validate", "format",
        ]

        # Dimensional indicators
        dimensional_signals = [
            "how many", "total", "average", "count",
            "sum", "revenue", "sales", "report",
            "last month", "this year", "by region",
            "trend", "compare", "top", "bottom",
        ]

        # Direct SQL
        sql_signals = ["select", "from", "where", "join", "group by"]

        scores = {
            SemanticLayer.ONTOLOGY: sum(1 for s in ontology_signals if s in q),
            SemanticLayer.KNOWLEDGE_GRAPH: sum(1 for s in kg_signals if s in q),
            SemanticLayer.CANONICAL: sum(1 for s in canonical_signals if s in q),
            SemanticLayer.DIMENSIONAL: sum(1 for s in dimensional_signals if s in q),
        }

        # SQL pass-through
        if any(s in q for s in sql_signals):
            return QueryClassification(
                primary_layer=SemanticLayer.DIMENSIONAL,
                confidence=0.9,
                reasoning="Direct SQL detected",
            )

        # Find primary layer
        max_score = max(scores.values())
        if max_score == 0:
            # Default to full pipeline
            return QueryClassification(
                primary_layer=SemanticLayer.DIMENSIONAL,
                secondary_layers=[
                    SemanticLayer.ONTOLOGY,
                    SemanticLayer.KNOWLEDGE_GRAPH,
                ],
                confidence=0.5,
                reasoning="No clear signal, using full pipeline",
                requires_cross_layer=True,
            )

        primary = max(scores, key=scores.get)
        secondary = [
            layer for layer, score in scores.items()
            if score > 0 and layer != primary
        ]

        confidence = min(0.9, 0.5 + (max_score * 0.15))

        return QueryClassification(
            primary_layer=primary,
            secondary_layers=secondary,
            confidence=confidence,
            reasoning=f"Heuristic: {primary.value} (score: {max_score})",
            requires_cross_layer=len(secondary) > 0,
        )

    async def _llm_classify(self, question: str) -> QueryClassification:
        """LLM-based classification for ambiguous queries"""
        prompt = f"""Classify this data question into one or more semantic layers.

QUESTION: {question}

LAYERS:
1. ONTOLOGY - Questions about concepts, definitions, business rules, domain meaning
2. KNOWLEDGE_GRAPH - Questions about relationships, connections, paths between entities
3. CANONICAL - Questions about data schemas, mappings, formats, transformations
4. DIMENSIONAL - Questions about metrics, aggregations, reporting, analytics

Return ONLY the primary layer name and optionally secondary layers.
Format: PRIMARY_LAYER | SECONDARY_LAYER1, SECONDARY_LAYER2
"""
        response = await self.llm_client.generate(prompt=prompt, temperature=0.1)
        return self._parse_classification(response, question)

    def _parse_classification(
        self,
        response: str,
        question: str,
    ) -> QueryClassification:
        """Parse LLM classification response"""
        response = response.strip().upper()
        parts = response.split("|")

        layer_map = {
            "ONTOLOGY": SemanticLayer.ONTOLOGY,
            "KNOWLEDGE_GRAPH": SemanticLayer.KNOWLEDGE_GRAPH,
            "CANONICAL": SemanticLayer.CANONICAL,
            "DIMENSIONAL": SemanticLayer.DIMENSIONAL,
        }

        primary_str = parts[0].strip()
        primary = layer_map.get(primary_str, SemanticLayer.DIMENSIONAL)

        secondary = []
        if len(parts) > 1:
            for s in parts[1].split(","):
                layer = layer_map.get(s.strip())
                if layer and layer != primary:
                    secondary.append(layer)

        return QueryClassification(
            primary_layer=primary,
            secondary_layers=secondary,
            confidence=0.85,
            reasoning=f"LLM classified as {primary.value}",
            requires_cross_layer=len(secondary) > 0,
        )
