"""
Analyst Agent - Interprets SQL results and generates insights
Part of the multi-agent pipeline
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..core.base import (
    Action,
    AgentConfig,
    AgentContext,
    BaseAgent,
    Thought,
    ThoughtType,
    UserContext,
)

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """
    Analyst Agent for data interpretation

    Responsibilities:
    - Analyze query results
    - Generate natural language explanations
    - Identify trends and patterns
    - Suggest follow-up questions
    - Create visualization recommendations
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_client: Any,
    ):
        super().__init__(config)
        self.llm_client = llm_client

    async def think(self, context: AgentContext, input_data: Any) -> Thought:
        """Analyze the data and plan interpretation"""
        data = input_data if isinstance(input_data, dict) else {"raw": input_data}

        analysis_prompt = f"""
        Analyze this query result data:

        Columns: {data.get('columns', [])}
        Row Count: {data.get('row_count', 0)}
        Sample Data: {json.dumps(data.get('data', [])[:5], indent=2)}
        Original Question: {data.get('question', 'Unknown')}

        Plan your analysis:
        1. What patterns or trends do you see?
        2. What's the key insight?
        3. What visualization would best represent this?
        4. What follow-up questions might be valuable?
        """

        response = await self.llm_client.generate(prompt=analysis_prompt)

        return Thought(
            type=ThoughtType.PLANNING,
            content=response,
            metadata={"data_shape": f"{data.get('row_count', 0)} rows x {len(data.get('columns', []))} cols"},
        )

    async def act(self, context: AgentContext, thought: Thought) -> Action:
        """Generate the analysis output"""
        action = Action(
            tool_name="generate_analysis",
            arguments={"thought": thought.content},
            thought=thought,
        )

        try:
            # Get the original data from context
            original_data = context.variables.get("input_data", {})

            explanation_prompt = f"""
            Based on your analysis:
            {thought.content}

            Generate:
            1. A concise natural language explanation (2-3 sentences)
            2. Key insights (bullet points)
            3. Recommended visualization type
            4. 2-3 follow-up questions

            Format as JSON with keys: explanation, insights, viz_type, follow_up_questions
            """

            response = await self.llm_client.generate(prompt=explanation_prompt)

            # Parse response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                result = {
                    "explanation": response,
                    "insights": [],
                    "viz_type": "table",
                    "follow_up_questions": [],
                }

            action.result = result

        except Exception as e:
            logger.exception("Analysis failed")
            action.error = str(e)

        return action

    async def reflect(self, context: AgentContext) -> Thought:
        """Reflect on analysis quality"""
        if not context.actions:
            return Thought(
                type=ThoughtType.REFLECTION,
                content="No analysis to reflect on",
                confidence=0.5,
            )

        last_action = context.actions[-1]
        result = last_action.result

        if not result or last_action.error:
            return Thought(
                type=ThoughtType.REFLECTION,
                content=f"Analysis failed: {last_action.error}",
                confidence=0.2,
            )

        # Evaluate completeness
        has_explanation = bool(result.get("explanation"))
        has_insights = bool(result.get("insights"))
        has_viz = bool(result.get("viz_type"))

        confidence = 0.3
        if has_explanation:
            confidence += 0.3
        if has_insights:
            confidence += 0.2
        if has_viz:
            confidence += 0.2

        return Thought(
            type=ThoughtType.REFLECTION,
            content=f"Analysis complete. Explanation: {'Yes' if has_explanation else 'No'}, "
                    f"Insights: {len(result.get('insights', []))}, Viz: {result.get('viz_type', 'None')}",
            confidence=confidence,
        )
