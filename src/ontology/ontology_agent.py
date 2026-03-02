"""
Ontology Agent - Reasons over ontology facts using LLM.

The Ontology Agent operates at the HIGHEST semantic level, understanding
domain meaning through elementary facts rather than tables and columns.

Responsibilities:
- Interpret natural language questions in terms of ontology concepts
- Map questions to relevant fact types and object types
- Generate semantic context for downstream agents (KG, SQL)
- Validate that queries align with domain semantics
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import Field

from ..core.base import (
    Action,
    AgentConfig,
    AgentContext,
    AgentResult,
    BaseAgent,
    Thought,
    ThoughtType,
    UserContext,
)
from .fact_model import ElementaryFact, OntologyDefinition
from .ontology_store import OntologyStore
from .reasoner import FactReasoner

logger = logging.getLogger(__name__)


class OntologyAgentConfig(AgentConfig):
    """Configuration for the Ontology Agent"""
    name: str = "ontology_agent"
    description: str = "Reasons over domain ontology facts for semantic understanding"

    # Ontology settings
    ontology_id: str = ""
    max_context_facts: int = 20
    enable_inference: bool = True
    inference_depth: int = 3


class OntologyAgent(BaseAgent):
    """
    Ontology Agent - Layer 1 of the Semantic Space Funnel.

    This agent reasons at the highest semantic level, working with
    elementary facts rather than tables/columns. It:

    1. THINK: Maps natural language to ontology concepts
    2. ACT: Retrieves relevant facts and performs inference
    3. REFLECT: Validates semantic alignment

    Output: Semantic context that informs downstream agents
    (Knowledge Graph Agent, SQL Agent).
    """

    def __init__(
        self,
        config: OntologyAgentConfig,
        ontology_store: OntologyStore,
        llm_client: Any,
        memory: Optional[Any] = None,
    ):
        super().__init__(config, memory=memory)
        self.config: OntologyAgentConfig = config
        self.ontology_store = ontology_store
        self.llm_client = llm_client
        self._ontology: Optional[OntologyDefinition] = None
        self._reasoner: Optional[FactReasoner] = None

    async def _ensure_ontology(self) -> OntologyDefinition:
        """Load ontology if not already loaded"""
        if self._ontology is None:
            self._ontology = await self.ontology_store.load_ontology(
                self.config.ontology_id
            )
            if self._ontology:
                self._reasoner = FactReasoner(self._ontology)
        return self._ontology

    async def think(self, context: AgentContext, input_data: Any) -> Thought:
        """
        THINK: Map natural language question to ontology concepts.

        Uses LLM to identify which object types and fact types
        are relevant to the user's question.
        """
        question = str(input_data)
        ontology = await self._ensure_ontology()

        if not ontology:
            return Thought(
                type=ThoughtType.REASONING,
                content=f"No ontology loaded for id: {self.config.ontology_id}",
                confidence=0.1,
            )

        # Get ontology summary for LLM context
        summary = await self.ontology_store.get_ontology_summary(
            self.config.ontology_id
        )

        # Use LLM to map question to ontology concepts
        prompt = self._build_concept_mapping_prompt(question, summary)
        analysis = await self.llm_client.generate(
            prompt=prompt,
            temperature=0.3,
        )

        # Use reasoner to find matching context
        reasoner_context = self._reasoner.generate_context_for_question(question)

        return Thought(
            type=ThoughtType.REASONING,
            content=f"Ontology Analysis:\n{analysis}",
            confidence=0.8,
            metadata={
                "question": question,
                "matching_types": reasoner_context.get("matching_types", []),
                "relevant_facts": reasoner_context.get("relevant_facts", []),
                "ontology_name": ontology.name,
            },
        )

    async def act(self, context: AgentContext, thought: Thought) -> Action:
        """
        ACT: Retrieve detailed facts and perform inference.

        Builds a rich semantic context from ontology facts that
        downstream agents can use.
        """
        question = thought.metadata.get("question", "")
        matching_types = thought.metadata.get("matching_types", [])

        action = Action(
            tool_name="ontology_context_retrieval",
            arguments={"question": question},
            thought=thought,
        )

        try:
            ontology = await self._ensure_ontology()

            # Retrieve facts from store
            relevant_facts = await self.ontology_store.find_facts_for_question(
                question=question,
                ontology_id=self.config.ontology_id,
                limit=self.config.max_context_facts,
            )

            # Expand context through reasoning
            expanded_facts = []
            if self._reasoner and matching_types:
                for mt in matching_types:
                    expanded = self._reasoner.expand_context(
                        mt["name"],
                        depth=self.config.inference_depth,
                    )
                    expanded_facts.extend(expanded)

            # Build semantic context
            semantic_context = self._build_semantic_context(
                question=question,
                direct_facts=relevant_facts,
                expanded_facts=expanded_facts,
                matching_types=matching_types,
            )

            action.result = semantic_context

        except Exception as e:
            logger.exception(f"Ontology action failed: {e}")
            action.error = str(e)

        return action

    async def reflect(self, context: AgentContext) -> Thought:
        """
        REFLECT: Validate semantic alignment of the context.

        Checks if the retrieved ontology context fully covers
        the user's question semantics.
        """
        if not context.actions:
            return Thought(
                type=ThoughtType.REFLECTION,
                content="No actions to reflect on",
                confidence=0.5,
            )

        last_action = context.actions[-1]
        question = context.thoughts[0].metadata.get("question", "") if context.thoughts else ""

        semantic_context = last_action.result or {}
        fact_count = len(semantic_context.get("fact_readings", []))

        confidence = min(1.0, fact_count / 5)  # Higher confidence with more facts

        return Thought(
            type=ThoughtType.REFLECTION,
            content=(
                f"Semantic coverage: {fact_count} relevant facts found. "
                f"{'Good coverage' if confidence > 0.6 else 'May need broader search'}."
            ),
            confidence=confidence,
            metadata={
                "fact_count": fact_count,
                "coverage": "sufficient" if confidence > 0.6 else "partial",
            },
        )

    def _build_concept_mapping_prompt(
        self,
        question: str,
        ontology_summary: Dict[str, Any],
    ) -> str:
        """Build prompt for mapping question to ontology concepts"""
        types_str = "\n".join(
            f"  - {t['name']} ({t['kind']}): {t['description']}"
            for t in ontology_summary.get("object_types", [])
        )
        facts_str = "\n".join(
            f"  - {f['name']}: \"{f['reading']}\""
            for f in ontology_summary.get("fact_types", [])
        )

        return f"""You are an ontology expert. Map this question to domain concepts.

QUESTION: {question}

DOMAIN ONTOLOGY: {ontology_summary.get('name', 'Unknown')}

OBJECT TYPES:
{types_str}

ELEMENTARY FACTS:
{facts_str}

Identify:
1. Which object types are relevant to this question?
2. Which fact types connect them?
3. What is the semantic meaning the user is looking for?
4. Any implicit relationships not directly stated?

Provide a structured analysis of how this question maps to the ontology.
"""

    def _build_semantic_context(
        self,
        question: str,
        direct_facts: List[ElementaryFact],
        expanded_facts: List[ElementaryFact],
        matching_types: List[Dict],
    ) -> Dict[str, Any]:
        """Build semantic context for downstream agents"""
        # Combine and deduplicate facts
        all_facts = {f.id: f for f in direct_facts}
        for f in expanded_facts:
            all_facts[f.id] = f

        return {
            "semantic_layer": "ontology",
            "question": question,
            "matching_object_types": matching_types,
            "fact_readings": [
                {
                    "name": f.name,
                    "reading": f.reading,
                    "object_types": f.object_types,
                    "arity": f.arity,
                }
                for f in all_facts.values()
            ],
            "entity_relationships": [
                f.to_triple()
                for f in all_facts.values()
                if f.to_triple() is not None
            ],
            "total_facts": len(all_facts),
        }
