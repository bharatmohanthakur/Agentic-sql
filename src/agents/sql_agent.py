"""
SQL Agent - Converts natural language to SQL with ReAct + Reflection
Best-in-class implementation combining Vanna 2.0 patterns with modern agentic design
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

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
from ..core.registry import ToolRegistry, ToolResult
from ..memory.manager import MemoryManager, MemoryType, MemoryPriority

logger = logging.getLogger(__name__)


class SQLAgentConfig(AgentConfig):
    """Configuration specific to SQL Agent"""
    name: str = "sql_agent"
    description: str = "Converts natural language to SQL queries"

    # SQL-specific settings
    max_sql_retries: int = 3
    enable_query_validation: bool = True
    enable_result_explanation: bool = True
    max_result_rows: int = 1000
    enable_visualization: bool = True

    # Security settings
    block_destructive_queries: bool = True
    require_row_level_security: bool = True
    allowed_schemas: List[str] = Field(default_factory=list)

    # LLM settings
    sql_generation_temperature: float = 0.3
    explanation_temperature: float = 0.7


class SchemaContext(BaseModel):
    """Database schema context for SQL generation"""
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    columns: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    sample_queries: List[str] = Field(default_factory=list)
    documentation: str = ""


class SQLGenerationResult(BaseModel):
    """Result of SQL generation"""
    sql: str
    confidence: float = 1.0
    explanation: str = ""
    tables_used: List[str] = Field(default_factory=list)
    is_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)


class QueryExecutionResult(BaseModel):
    """Result of query execution"""
    data: List[Dict[str, Any]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    truncated: bool = False
    error: Optional[str] = None


class SQLAgent(BaseAgent):
    """
    SQL Agent implementing ReAct + Reflection for text-to-SQL

    Workflow:
    1. THINK: Understand user question, retrieve relevant schema/patterns
    2. ACT: Generate SQL using LLM with context
    3. VALIDATE: Check SQL syntax and security
    4. EXECUTE: Run query against database
    5. REFLECT: Verify results match intent, explain if needed
    """

    def __init__(
        self,
        config: SQLAgentConfig,
        tool_registry: ToolRegistry,
        memory: MemoryManager,
        llm_client: Any,  # LLM client interface
        db_executor: Callable,  # Database execution function
    ):
        super().__init__(config, memory=memory)
        self.config: SQLAgentConfig = config
        self.tool_registry = tool_registry
        self.llm_client = llm_client
        self.db_executor = db_executor
        self._schema_cache: Optional[SchemaContext] = None

    async def think(self, context: AgentContext, input_data: Any) -> Thought:
        """
        THINK phase: Understand the question and plan SQL generation

        Steps:
        1. Parse user question
        2. Retrieve relevant schema context
        3. Search memory for similar patterns
        4. Plan SQL generation approach
        """
        question = str(input_data)

        # Retrieve relevant memories (similar queries, schema info)
        relevant_memories = await self._retrieve_relevant_context(question, context)

        # Get schema context
        schema_context = await self._get_schema_context(context.user)

        # Use LLM to analyze and plan
        analysis_prompt = self._build_analysis_prompt(
            question,
            schema_context,
            relevant_memories,
        )

        analysis = await self._call_llm(
            analysis_prompt,
            temperature=self.config.temperature,
        )

        thought = Thought(
            type=ThoughtType.PLANNING,
            content=f"Question Analysis:\n{analysis}",
            confidence=self._extract_confidence(analysis),
            metadata={
                "question": question,
                "tables_identified": self._extract_tables(analysis),
                "memory_count": len(relevant_memories),
            },
        )

        return thought

    async def act(self, context: AgentContext, thought: Thought) -> Action:
        """
        ACT phase: Generate and execute SQL

        Steps:
        1. Generate SQL from LLM
        2. Validate SQL
        3. Apply row-level security
        4. Execute query
        """
        question = thought.metadata.get("question", "")

        action = Action(
            tool_name="generate_and_execute_sql",
            arguments={"question": question},
            thought=thought,
        )

        try:
            # Generate SQL
            sql_result = await self._generate_sql(question, context)

            if not sql_result.is_valid:
                action.error = f"SQL validation failed: {sql_result.validation_errors}"
                return action

            # Apply row-level security
            if self.config.require_row_level_security and context.user:
                sql_result.sql = self._apply_row_level_security(
                    sql_result.sql,
                    context.user,
                )

            # Execute query
            execution_result = await self._execute_sql(sql_result.sql, context.user)

            if execution_result.error:
                action.error = execution_result.error
                # Store error pattern for learning
                await self._store_error_pattern(question, sql_result.sql, execution_result.error)
            else:
                action.result = {
                    "sql": sql_result.sql,
                    "data": execution_result.data,
                    "columns": execution_result.columns,
                    "row_count": execution_result.row_count,
                    "execution_time_ms": execution_result.execution_time_ms,
                    "explanation": sql_result.explanation,
                }

                # Store successful pattern
                await self._store_success_pattern(question, sql_result.sql)

        except Exception as e:
            logger.exception(f"SQL action failed: {e}")
            action.error = str(e)

        return action

    async def reflect(self, context: AgentContext) -> Thought:
        """
        REFLECT phase: Verify results match user intent

        Steps:
        1. Compare results with original question
        2. Check for potential issues (empty results, unexpected data)
        3. Suggest improvements if needed
        """
        if not context.actions:
            return Thought(
                type=ThoughtType.REFLECTION,
                content="No actions to reflect on",
                confidence=0.5,
            )

        last_action = context.actions[-1]
        original_question = context.thoughts[0].metadata.get("question", "") if context.thoughts else ""

        reflection_prompt = self._build_reflection_prompt(
            original_question,
            last_action,
        )

        reflection = await self._call_llm(
            reflection_prompt,
            temperature=0.3,  # Low temperature for accurate reflection
        )

        confidence = self._evaluate_result_quality(last_action.result)

        return Thought(
            type=ThoughtType.REFLECTION,
            content=reflection,
            confidence=confidence,
            metadata={
                "action_id": str(last_action.id),
                "had_error": bool(last_action.error),
            },
        )

    async def plan(self, context: AgentContext, goal: str) -> List[Thought]:
        """
        PLAN phase: Break down complex queries into steps
        """
        # Analyze if this is a complex multi-step query
        complexity = await self._analyze_complexity(goal)

        if complexity < 0.5:
            # Simple query - single step
            return [Thought(
                type=ThoughtType.PLANNING,
                content=f"Simple query - direct SQL generation for: {goal}",
            )]

        # Complex query - break into steps
        decomposition_prompt = f"""
        Break down this complex data question into simpler steps:
        Question: {goal}

        For each step, specify:
        1. What data to retrieve
        2. Any transformations needed
        3. How it connects to the final answer
        """

        decomposition = await self._call_llm(decomposition_prompt)
        steps = self._parse_steps(decomposition)

        return [
            Thought(
                type=ThoughtType.PLANNING,
                content=step,
                metadata={"step_index": i},
            )
            for i, step in enumerate(steps)
        ]

    async def _retrieve_relevant_context(
        self,
        question: str,
        context: AgentContext,
    ) -> List[Any]:
        """Retrieve relevant memories for the question"""
        if not self.memory:
            return []

        memories = await self.memory.search(
            query=question,
            memory_type=MemoryType.QUERY_PATTERN,
            entity_id=context.user.user_id if context.user else None,
            limit=5,
        )

        # Also get schema memories
        schema_memories = await self.memory.search(
            query=question,
            memory_type=MemoryType.SCHEMA,
            limit=3,
        )

        return memories + schema_memories

    async def _get_schema_context(
        self,
        user: Optional[UserContext],
    ) -> SchemaContext:
        """Get database schema context, respecting user permissions"""
        if self._schema_cache:
            return self._schema_cache

        # Call schema retrieval tool
        result = await self.tool_registry.execute(
            "get_schema",
            user or UserContext(user_id="system"),
        )

        if result.success:
            self._schema_cache = SchemaContext(**result.data)

        return self._schema_cache or SchemaContext()

    async def _generate_sql(
        self,
        question: str,
        context: AgentContext,
    ) -> SQLGenerationResult:
        """Generate SQL using LLM with schema context"""
        schema_context = await self._get_schema_context(context.user)

        prompt = self._build_sql_prompt(question, schema_context, context)

        sql_response = await self._call_llm(
            prompt,
            temperature=self.config.sql_generation_temperature,
        )

        # Extract SQL from response
        sql = self._extract_sql(sql_response)

        # Validate SQL
        validation_result = await self._validate_sql(sql, context.user)

        return SQLGenerationResult(
            sql=sql,
            confidence=self._extract_confidence(sql_response),
            explanation=self._extract_explanation(sql_response),
            tables_used=self._extract_tables_from_sql(sql),
            is_valid=len(validation_result) == 0,
            validation_errors=validation_result,
        )

    async def _validate_sql(
        self,
        sql: str,
        user: Optional[UserContext],
    ) -> List[str]:
        """Validate SQL for syntax and security"""
        errors = []

        # Block destructive queries
        if self.config.block_destructive_queries:
            destructive_patterns = [
                r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b',
                r'\bUPDATE\b', r'\bINSERT\b', r'\bALTER\b',
                r'\bCREATE\b', r'\bGRANT\b', r'\bREVOKE\b',
            ]
            for pattern in destructive_patterns:
                if re.search(pattern, sql, re.IGNORECASE):
                    errors.append(f"Destructive operation not allowed: {pattern}")

        # Check schema restrictions
        if self.config.allowed_schemas:
            # Extract schema/table references and validate
            pass

        # Syntax validation via EXPLAIN
        if self.config.enable_query_validation:
            try:
                explain_result = await self.db_executor(f"EXPLAIN {sql}")
            except Exception as e:
                errors.append(f"SQL syntax error: {e}")

        return errors

    def _apply_row_level_security(
        self,
        sql: str,
        user: UserContext,
    ) -> str:
        """Apply row-level security filters to SQL"""
        filters = user.get_sql_filters()
        if not filters:
            return sql

        # Parse and inject WHERE clauses
        # This is a simplified implementation
        for table, conditions in filters.items():
            for column, value in conditions.items():
                filter_clause = f"{table}.{column} = '{value}'"
                if "WHERE" in sql.upper():
                    sql = sql.replace("WHERE", f"WHERE {filter_clause} AND")
                else:
                    # Add WHERE before ORDER BY, LIMIT, etc.
                    sql = re.sub(
                        r'(ORDER BY|LIMIT|GROUP BY|$)',
                        f' WHERE {filter_clause} \\1',
                        sql,
                        count=1,
                        flags=re.IGNORECASE,
                    )

        return sql

    async def _execute_sql(
        self,
        sql: str,
        user: Optional[UserContext],
    ) -> QueryExecutionResult:
        """Execute SQL and return results"""
        start_time = datetime.utcnow()

        try:
            result = await self.db_executor(sql)

            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Handle different result formats
            if isinstance(result, list):
                data = result
                columns = list(result[0].keys()) if result else []
            else:
                data = list(result)
                columns = []

            truncated = False
            if len(data) > self.config.max_result_rows:
                data = data[:self.config.max_result_rows]
                truncated = True

            return QueryExecutionResult(
                data=data,
                columns=columns,
                row_count=len(data),
                execution_time_ms=execution_time,
                truncated=truncated,
            )

        except Exception as e:
            return QueryExecutionResult(error=str(e))

    async def _store_success_pattern(self, question: str, sql: str) -> None:
        """Store successful query pattern in memory"""
        if not self.memory:
            return

        await self.memory.ingest(
            content=json.dumps({"question": question, "sql": sql}),
            memory_type=MemoryType.QUERY_PATTERN,
            priority=MemoryPriority.HIGH,
        )

    async def _store_error_pattern(
        self,
        question: str,
        sql: str,
        error: str,
    ) -> None:
        """Store error pattern to avoid in future"""
        if not self.memory:
            return

        await self.memory.ingest(
            content=json.dumps({"question": question, "sql": sql, "error": error}),
            memory_type=MemoryType.ERROR_PATTERN,
            priority=MemoryPriority.MEDIUM,
        )

    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """Call LLM with prompt"""
        # Interface to be implemented by specific LLM providers
        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=self.config.thinking_budget,
        )
        return response

    async def _analyze_complexity(self, question: str) -> float:
        """Analyze question complexity (0-1)"""
        # Simple heuristics
        complexity = 0.0

        # Multiple conditions
        if any(word in question.lower() for word in ['and', 'or', 'but', 'except']):
            complexity += 0.2

        # Aggregations
        if any(word in question.lower() for word in ['average', 'sum', 'count', 'total', 'max', 'min']):
            complexity += 0.2

        # Time-based
        if any(word in question.lower() for word in ['yesterday', 'last week', 'this month', 'between']):
            complexity += 0.2

        # Joins (multiple entities)
        if any(word in question.lower() for word in ['with', 'from', 'each', 'per']):
            complexity += 0.2

        # Subqueries
        if any(word in question.lower() for word in ['compared to', 'relative to', 'percentage']):
            complexity += 0.2

        return min(complexity, 1.0)

    def _parse_steps(self, decomposition: str) -> List[str]:
        """Parse decomposition into steps"""
        # Simple parsing - split by numbered items
        steps = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', decomposition, re.DOTALL)
        return [s.strip() for s in steps if s.strip()]

    def _build_analysis_prompt(
        self,
        question: str,
        schema: SchemaContext,
        memories: List[Any],
    ) -> str:
        """Build prompt for question analysis"""
        schema_summary = self._format_schema_summary(schema)
        memory_context = self._format_memories(memories)

        return f"""Analyze this data question and plan the SQL generation:

QUESTION: {question}

DATABASE SCHEMA:
{schema_summary}

SIMILAR PAST QUERIES:
{memory_context}

Provide:
1. Key entities/tables involved
2. Required columns
3. Filters and conditions
4. Aggregations needed
5. Confidence level (0-1)
"""

    def _build_sql_prompt(
        self,
        question: str,
        schema: SchemaContext,
        context: AgentContext,
    ) -> str:
        """Build prompt for SQL generation"""
        schema_detail = self._format_schema_detail(schema)
        reasoning = context.get_reasoning_chain()

        return f"""Generate SQL for this question:

QUESTION: {question}

DATABASE SCHEMA:
{schema_detail}

ANALYSIS:
{reasoning}

INSTRUCTIONS:
- Generate a single, correct SQL query
- Use appropriate JOINs when relating tables
- Include necessary WHERE clauses
- Use appropriate aggregations
- Format SQL for readability

Return your response in this format:
```sql
YOUR SQL QUERY HERE
```

EXPLANATION:
Brief explanation of what the query does

CONFIDENCE: 0.0-1.0
"""

    def _build_reflection_prompt(
        self,
        question: str,
        action: Action,
    ) -> str:
        """Build prompt for reflection"""
        result_summary = json.dumps(action.result, indent=2)[:1000] if action.result else "No result"

        return f"""Reflect on whether the SQL query correctly answers the question:

ORIGINAL QUESTION: {question}

SQL EXECUTED: {action.arguments}

RESULT SUMMARY:
{result_summary}

ERROR (if any): {action.error or 'None'}

Evaluate:
1. Does the result answer the question?
2. Are there any data quality issues?
3. Could the query be improved?
4. Confidence in the result (0-1)
"""

    def _extract_sql(self, response: str) -> str:
        """Extract SQL from LLM response"""
        # Look for SQL in code blocks
        match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Look for SELECT statements
        match = re.search(r'(SELECT\s+.*?)(;|$)', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return response.strip()

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.7  # Default confidence

    def _extract_explanation(self, response: str) -> str:
        """Extract explanation from response"""
        match = re.search(r'EXPLANATION:\s*(.+?)(?=CONFIDENCE:|$)', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_tables(self, response: str) -> List[str]:
        """Extract table names from analysis"""
        # Simple extraction - look for table-like patterns
        tables = re.findall(r'\b([a-z_]+)\s+table\b', response, re.IGNORECASE)
        return list(set(tables))

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL"""
        # FROM and JOIN clauses
        tables = re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE)
        return list(set(tables))

    def _format_schema_summary(self, schema: SchemaContext) -> str:
        """Format schema as summary"""
        lines = []
        for table in schema.tables:
            lines.append(f"- {table.get('name', 'unknown')}: {table.get('description', '')}")
        return "\n".join(lines) or "No schema available"

    def _format_schema_detail(self, schema: SchemaContext) -> str:
        """Format schema with full details"""
        lines = []
        for table in schema.tables:
            table_name = table.get('name', 'unknown')
            lines.append(f"\n{table_name}:")
            columns = schema.columns.get(table_name, [])
            for col in columns:
                lines.append(f"  - {col.get('name')}: {col.get('type')} {col.get('description', '')}")
        return "\n".join(lines) or "No schema available"

    def _format_memories(self, memories: List[Any]) -> str:
        """Format memories for context"""
        if not memories:
            return "No similar queries found"

        lines = []
        for mem in memories[:5]:
            lines.append(f"- {mem.content[:200]}...")
        return "\n".join(lines)

    def _evaluate_result_quality(self, result: Any) -> float:
        """Evaluate quality of query result"""
        if not result:
            return 0.3

        if isinstance(result, dict):
            data = result.get("data", [])
            if not data:
                return 0.5  # Empty result might be correct

            row_count = result.get("row_count", 0)
            if row_count > 0:
                return 0.8

        return 0.7
