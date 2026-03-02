"""
Knowledge Graph Agent - Generates Cypher queries from semantic context.

Layer 2 agent that translates ontology-enriched questions into
Neo4j Cypher queries for knowledge graph traversal.
"""

from __future__ import annotations

import logging
import re
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
from .graph_model import GraphSchema
from .neo4j_knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class KnowledgeGraphAgentConfig(AgentConfig):
    """Configuration for the Knowledge Graph Agent"""
    name: str = "knowledge_graph_agent"
    description: str = "Generates and executes Cypher queries on the knowledge graph"

    max_traversal_depth: int = 5
    max_result_nodes: int = 100
    enable_vector_search: bool = True
    cypher_generation_temperature: float = 0.3


class KnowledgeGraphAgent(BaseAgent):
    """
    Knowledge Graph Agent - Layer 2 of the Semantic Space Funnel.

    Takes semantic context from the Ontology Agent (Layer 1) and
    generates Cypher queries to traverse the knowledge graph in Neo4j.

    Workflow:
    1. THINK: Analyze semantic context + question → plan graph query
    2. ACT: Generate Cypher, execute on Neo4j, return subgraph
    3. REFLECT: Validate results cover the question's intent
    """

    def __init__(
        self,
        config: KnowledgeGraphAgentConfig,
        knowledge_graph: KnowledgeGraph,
        llm_client: Any,
        memory: Optional[Any] = None,
    ):
        super().__init__(config, memory=memory)
        self.config: KnowledgeGraphAgentConfig = config
        self.kg = knowledge_graph
        self.llm_client = llm_client

    async def think(self, context: AgentContext, input_data: Any) -> Thought:
        """
        THINK: Plan the graph query based on semantic context.

        Input can be:
        - Raw question string
        - Semantic context from Ontology Agent
        """
        if isinstance(input_data, dict) and "semantic_layer" in input_data:
            # Received semantic context from Ontology Agent
            question = input_data.get("question", "")
            semantic_context = input_data
        else:
            question = str(input_data)
            semantic_context = {}

        # Get graph schema for context
        graph_summary = await self.kg.get_graph_summary()
        schema_text = graph_summary.get("schema", "")

        # Build planning prompt
        prompt = self._build_planning_prompt(
            question, semantic_context, schema_text
        )

        analysis = await self.llm_client.generate(
            prompt=prompt,
            temperature=self.config.temperature,
        )

        return Thought(
            type=ThoughtType.PLANNING,
            content=f"Graph Query Plan:\n{analysis}",
            confidence=0.8,
            metadata={
                "question": question,
                "semantic_context": semantic_context,
                "schema": schema_text,
            },
        )

    async def act(self, context: AgentContext, thought: Thought) -> Action:
        """
        ACT: Generate and execute Cypher query.
        """
        question = thought.metadata.get("question", "")
        semantic_context = thought.metadata.get("semantic_context", {})
        schema = thought.metadata.get("schema", "")

        action = Action(
            tool_name="cypher_query",
            arguments={"question": question},
            thought=thought,
        )

        try:
            # Generate Cypher using LLM
            cypher = await self._generate_cypher(
                question, semantic_context, schema
            )

            if not cypher:
                action.error = "Failed to generate Cypher query"
                return action

            # Validate Cypher (basic safety check)
            validation_errors = self._validate_cypher(cypher)
            if validation_errors:
                action.error = f"Cypher validation failed: {validation_errors}"
                return action

            # Execute query
            results = await self.kg.execute_cypher(cypher)

            action.result = {
                "cypher": cypher,
                "results": results[:self.config.max_result_nodes],
                "result_count": len(results),
                "truncated": len(results) > self.config.max_result_nodes,
            }

        except Exception as e:
            logger.exception(f"Knowledge graph action failed: {e}")
            action.error = str(e)

        return action

    async def reflect(self, context: AgentContext) -> Thought:
        """
        REFLECT: Validate graph query results.
        """
        if not context.actions:
            return Thought(
                type=ThoughtType.REFLECTION,
                content="No actions to reflect on",
                confidence=0.5,
            )

        last_action = context.actions[-1]
        result = last_action.result or {}
        result_count = result.get("result_count", 0)

        confidence = 0.3 if result_count == 0 else min(1.0, result_count / 5)

        return Thought(
            type=ThoughtType.REFLECTION,
            content=(
                f"Graph query returned {result_count} results. "
                f"{'Good coverage' if confidence > 0.6 else 'Consider expanding search'}."
            ),
            confidence=confidence,
        )

    async def _generate_cypher(
        self,
        question: str,
        semantic_context: Dict[str, Any],
        schema: str,
    ) -> Optional[str]:
        """Generate Cypher query using LLM"""
        prompt = self._build_cypher_generation_prompt(
            question, semantic_context, schema
        )

        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=self.config.cypher_generation_temperature,
        )

        return self._extract_cypher(response)

    def _validate_cypher(self, cypher: str) -> List[str]:
        """Basic Cypher validation (safety checks)"""
        errors = []
        dangerous = [
            r'\bDELETE\b', r'\bDETACH\b', r'\bDROP\b',
            r'\bCREATE\b', r'\bSET\b', r'\bREMOVE\b',
        ]
        for pattern in dangerous:
            if re.search(pattern, cypher, re.IGNORECASE):
                errors.append(f"Write operation not allowed: {pattern}")
        return errors

    def _build_planning_prompt(
        self,
        question: str,
        semantic_context: Dict[str, Any],
        schema: str,
    ) -> str:
        """Build prompt for query planning"""
        context_str = ""
        if semantic_context:
            facts = semantic_context.get("fact_readings", [])
            context_str = "\n".join(
                f"  - {f['reading']} (types: {f['object_types']})"
                for f in facts
            )

        return f"""You are a knowledge graph expert. Plan a Neo4j Cypher query.

QUESTION: {question}

GRAPH SCHEMA:
{schema}

ONTOLOGY CONTEXT (from semantic analysis):
{context_str or 'No ontology context available'}

Plan:
1. Which node labels are relevant?
2. Which relationships to traverse?
3. What pattern matches are needed?
4. Depth of traversal required?
"""

    def _build_cypher_generation_prompt(
        self,
        question: str,
        semantic_context: Dict[str, Any],
        schema: str,
    ) -> str:
        """Build prompt for Cypher generation"""
        entity_rels = semantic_context.get("entity_relationships", [])
        rels_str = "\n".join(
            f"  ({r[0]})-[:{r[1]}]->({r[2]})" for r in entity_rels if r
        )

        return f"""Generate a Neo4j Cypher query for this question.

QUESTION: {question}

GRAPH SCHEMA:
{schema}

KNOWN RELATIONSHIPS (from ontology):
{rels_str or 'None'}

INSTRUCTIONS:
- Generate a READ-ONLY Cypher query (no mutations)
- Use appropriate MATCH patterns
- Include RETURN clause with meaningful aliases
- Limit results to 100
- Use WHERE for filtering

Return your query in this format:
```cypher
YOUR QUERY HERE
```
"""

    def _extract_cypher(self, response: str) -> Optional[str]:
        """Extract Cypher from LLM response"""
        match = re.search(r'```cypher\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        match = re.search(r'(MATCH\s+.*?)(;|$)', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None
