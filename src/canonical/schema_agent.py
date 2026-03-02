"""
Schema Agent - Validates data against canonical models and manages schema operations.

Layer 3 agent that ensures data conforms to canonical model contracts.
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
)
from .canonical_model import CanonicalEntity
from .schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)


class SchemaAgentConfig(AgentConfig):
    """Configuration for the Schema Agent"""
    name: str = "schema_agent"
    description: str = "Validates data against canonical models and translates between formats"

    strict_mode: bool = True  # Reject vs coerce on validation failure
    auto_generate_mappings: bool = True


class SchemaAgent(BaseAgent):
    """
    Schema Agent - Layer 3 of the Semantic Space Funnel.

    Validates incoming data against canonical Pydantic models,
    manages schema versioning, and generates mappings.

    Workflow:
    1. THINK: Identify entity type from incoming data
    2. ACT: Load canonical schema, validate, transform if needed
    3. REFLECT: Report validation results, suggest fixes
    """

    def __init__(
        self,
        config: SchemaAgentConfig,
        schema_registry: SchemaRegistry,
        llm_client: Any,
        memory: Optional[Any] = None,
    ):
        super().__init__(config, memory=memory)
        self.config: SchemaAgentConfig = config
        self.registry = schema_registry
        self.llm_client = llm_client

    async def think(self, context: AgentContext, input_data: Any) -> Thought:
        """
        THINK: Identify entity type and select canonical model.
        """
        if isinstance(input_data, dict):
            entity_type = (
                input_data.get("entity_type")
                or input_data.get("_type")
                or input_data.get("type")
            )

            if not entity_type and self.config.auto_generate_mappings:
                # Use LLM to infer entity type
                entity_type = await self._infer_entity_type(input_data)

            return Thought(
                type=ThoughtType.REASONING,
                content=f"Identified entity type: {entity_type}",
                confidence=0.9 if entity_type else 0.3,
                metadata={
                    "entity_type": entity_type,
                    "input_data": input_data,
                },
            )
        else:
            return Thought(
                type=ThoughtType.REASONING,
                content=f"Input is not a dict, cannot validate: {type(input_data)}",
                confidence=0.1,
                metadata={"input_data": input_data},
            )

    async def act(self, context: AgentContext, thought: Thought) -> Action:
        """
        ACT: Validate data against canonical model.
        """
        entity_type = thought.metadata.get("entity_type")
        input_data = thought.metadata.get("input_data", {})

        action = Action(
            tool_name="validate_canonical",
            arguments={"entity_type": entity_type},
            thought=thought,
        )

        if not entity_type:
            action.error = "Could not determine entity type"
            return action

        # Validate against registry
        errors = self.registry.validate_data(entity_type, input_data)

        if errors and not self.config.strict_mode:
            # Try to coerce/fix
            fixed_data = await self._attempt_coercion(input_data, entity_type, errors)
            new_errors = self.registry.validate_data(entity_type, fixed_data)
            if not new_errors:
                action.result = {
                    "valid": True,
                    "data": fixed_data,
                    "coerced": True,
                    "original_errors": errors,
                }
                return action

        action.result = {
            "valid": len(errors) == 0,
            "data": input_data if not errors else None,
            "errors": errors,
            "entity_type": entity_type,
            "schema_version": (
                self.registry.get_latest(entity_type).version
                if self.registry.get_latest(entity_type)
                else None
            ),
        }
        return action

    async def reflect(self, context: AgentContext) -> Thought:
        """
        REFLECT: Report on validation results.
        """
        if not context.actions:
            return Thought(
                type=ThoughtType.REFLECTION,
                content="No validation performed",
                confidence=0.5,
            )

        last_action = context.actions[-1]
        result = last_action.result or {}

        if result.get("valid"):
            return Thought(
                type=ThoughtType.REFLECTION,
                content="Validation passed. Data conforms to canonical model.",
                confidence=1.0,
            )
        else:
            errors = result.get("errors", [])
            return Thought(
                type=ThoughtType.REFLECTION,
                content=f"Validation failed with {len(errors)} errors: {errors}",
                confidence=0.3,
            )

    async def _infer_entity_type(self, data: Dict[str, Any]) -> Optional[str]:
        """Use LLM to infer entity type from data structure"""
        known_types = self.registry.list_entity_types()
        if not known_types:
            return None

        registry_summary = self.registry.to_summary()
        prompt = f"""Given these canonical entity types and their properties:
{registry_summary}

What entity type best matches this data?
{data}

Return ONLY the entity type name, nothing else.
"""
        response = await self.llm_client.generate(prompt=prompt, temperature=0.1)
        inferred = response.strip()
        return inferred if inferred in known_types else None

    async def _attempt_coercion(
        self,
        data: Dict[str, Any],
        entity_type: str,
        errors: List[str],
    ) -> Dict[str, Any]:
        """Attempt to fix validation errors through coercion"""
        schema = self.registry.get_latest(entity_type)
        if not schema:
            return data

        prompt = f"""Fix this data to match the canonical schema.

Schema: {schema.json_schema}
Data: {data}
Errors: {errors}

Return the fixed data as valid JSON.
"""
        response = await self.llm_client.generate(prompt=prompt, temperature=0.1)

        try:
            import json
            return json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return data
