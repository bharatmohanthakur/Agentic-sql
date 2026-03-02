"""
Dimensional SQL Generation Agent - Semantic-aware SQL generation for star schemas.

Understands star-join patterns, SCD filtering, bridge tables,
and Snowflake-specific optimizations.
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
)
from .dimensional_model import StarSchema
from .snowflake_client import SnowflakeClient

logger = logging.getLogger(__name__)


class DimensionalSQLAgentConfig(AgentConfig):
    """Configuration for the Dimensional SQL Agent"""
    name: str = "dimensional_sql_agent"
    description: str = "Generates optimized Snowflake SQL against star schemas"

    max_result_rows: int = 1000
    enable_scd_filtering: bool = True
    sql_generation_temperature: float = 0.2
    block_destructive_queries: bool = True


class DimensionalSQLAgent(BaseAgent):
    """
    Dimensional SQL Agent - Layer 4 of the Semantic Space Funnel.

    Generates optimized SQL for Snowflake star/snowflake schemas.
    Receives enriched semantic context from upstream agents.

    Knows about:
    - Star schema join patterns (fact → dimension)
    - SCD Type 2 current-record filtering
    - Bridge table weighting for M:M
    - Conformed dimension drill-across
    - Snowflake optimizations (clustering, materialized views)

    Workflow:
    1. THINK: Analyze question + semantic context → identify tables/joins
    2. ACT: Generate SQL with star-join patterns, execute on Snowflake
    3. REFLECT: Validate results, check for data quality issues
    """

    def __init__(
        self,
        config: DimensionalSQLAgentConfig,
        star_schema: StarSchema,
        snowflake_client: SnowflakeClient,
        llm_client: Any,
        memory: Optional[Any] = None,
    ):
        super().__init__(config, memory=memory)
        self.config: DimensionalSQLAgentConfig = config
        self.star_schema = star_schema
        self.snowflake = snowflake_client
        self.llm_client = llm_client

    async def think(self, context: AgentContext, input_data: Any) -> Thought:
        """
        THINK: Plan the dimensional query.

        Uses upstream semantic context (from ontology/KG agents) plus
        the star schema definition to plan optimal SQL.
        """
        if isinstance(input_data, dict) and "semantic_layer" in input_data:
            question = input_data.get("question", "")
            upstream_context = input_data
        else:
            question = str(input_data)
            upstream_context = {}

        # Build schema context
        schema_context = self.star_schema.to_schema_context()

        prompt = f"""You are a dimensional modeling expert. Plan an optimized star-schema SQL query.

QUESTION: {question}

STAR SCHEMA:
{schema_context}

SEMANTIC CONTEXT (from upstream agents):
{upstream_context.get('fact_readings', 'None available')}

Plan:
1. Which fact table(s) are needed?
2. Which dimensions to join?
3. SCD handling needed? (Type 2 current-record filter?)
4. Any bridge tables for M:M?
5. Aggregations and GROUP BY?
6. Snowflake optimizations?
"""
        analysis = await self.llm_client.generate(
            prompt=prompt,
            temperature=self.config.temperature,
        )

        return Thought(
            type=ThoughtType.PLANNING,
            content=f"Dimensional Query Plan:\n{analysis}",
            confidence=0.8,
            metadata={
                "question": question,
                "upstream_context": upstream_context,
            },
        )

    async def act(self, context: AgentContext, thought: Thought) -> Action:
        """
        ACT: Generate and execute dimensional SQL.
        """
        question = thought.metadata.get("question", "")
        upstream_context = thought.metadata.get("upstream_context", {})

        action = Action(
            tool_name="dimensional_sql_query",
            arguments={"question": question},
            thought=thought,
        )

        try:
            # Generate SQL
            sql = await self._generate_dimensional_sql(
                question, upstream_context, thought.content
            )

            if not sql:
                action.error = "Failed to generate SQL"
                return action

            # Validate
            validation_errors = self._validate_dimensional_sql(sql)
            if validation_errors:
                action.error = f"Validation errors: {validation_errors}"
                return action

            # Execute
            results = await self.snowflake.execute_query(sql)

            action.result = {
                "sql": sql,
                "data": results[:self.config.max_result_rows],
                "row_count": len(results),
                "truncated": len(results) > self.config.max_result_rows,
            }

        except Exception as e:
            logger.exception(f"Dimensional SQL action failed: {e}")
            action.error = str(e)

        return action

    async def reflect(self, context: AgentContext) -> Thought:
        """REFLECT: Validate results."""
        if not context.actions:
            return Thought(type=ThoughtType.REFLECTION, content="No actions", confidence=0.5)

        last_action = context.actions[-1]
        result = last_action.result or {}
        row_count = result.get("row_count", 0)

        if last_action.error:
            return Thought(
                type=ThoughtType.REFLECTION,
                content=f"Query failed: {last_action.error}",
                confidence=0.2,
            )

        confidence = 0.5 if row_count == 0 else min(1.0, 0.6 + (row_count / 100) * 0.4)

        return Thought(
            type=ThoughtType.REFLECTION,
            content=f"Query returned {row_count} rows. "
                    f"{'Results look reasonable' if confidence > 0.7 else 'May need refinement'}.",
            confidence=confidence,
        )

    async def _generate_dimensional_sql(
        self,
        question: str,
        upstream_context: Dict[str, Any],
        plan: str,
    ) -> Optional[str]:
        """Generate dimensional SQL using LLM"""
        schema_context = self.star_schema.to_schema_context()

        prompt = f"""Generate optimized Snowflake SQL for this star schema query.

QUESTION: {question}

STAR SCHEMA:
{schema_context}

QUERY PLAN:
{plan}

RULES:
- Always join fact to dimension (never dimension to dimension directly)
- For SCD Type 2 dimensions, add: WHERE d.is_current = TRUE
- Use bridge tables with weight factors for M:M aggregations
- Use Snowflake functions: DATEADD, DATEDIFF, QUALIFY, FLATTEN where appropriate
- Include LIMIT {self.config.max_result_rows}
- Format SQL for readability

```sql
YOUR SQL HERE
```
"""
        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=self.config.sql_generation_temperature,
        )

        return self._extract_sql(response)

    def _validate_dimensional_sql(self, sql: str) -> List[str]:
        """Validate dimensional SQL for safety and correctness"""
        errors = []

        if self.config.block_destructive_queries:
            dangerous = [r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b',
                         r'\bUPDATE\b', r'\bINSERT\b', r'\bALTER\b']
            for pattern in dangerous:
                if re.search(pattern, sql, re.IGNORECASE):
                    errors.append(f"Destructive operation not allowed: {pattern}")

        return errors

    def _extract_sql(self, response: str) -> Optional[str]:
        """Extract SQL from LLM response"""
        match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r'(SELECT\s+.*?)(;|$)', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
