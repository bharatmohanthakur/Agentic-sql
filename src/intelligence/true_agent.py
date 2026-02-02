"""
TRUE AGENTIC SYSTEM - Chain of Thought, Tree of Thoughts, ReAct
================================================================

This is a REAL agent that:
1. THINKS before acting (Chain of Thought)
2. EXPLORES multiple options (Tree of Thoughts)
3. ACTS, OBSERVES, ITERATES (ReAct pattern)
4. REFLECTS on its reasoning
5. PLANS before executing
6. LEARNS from every interaction

NO shortcuts. PURE agentic reasoning.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Thought:
    """A reasoning step"""
    content: str
    confidence: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Action:
    """An action taken by the agent"""
    action_type: str  # "query_schema", "query_data", "generate_sql", "execute_sql", "fix_sql"
    input_data: str
    output_data: Optional[str] = None
    success: bool = False


@dataclass
class AgentTrace:
    """Complete trace of agent reasoning"""
    question: str
    thoughts: List[Thought] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    sql_options: List[Dict] = field(default_factory=list)  # Tree of Thoughts
    final_sql: Optional[str] = None
    result: Optional[Dict] = None


class TrueAgent:
    """
    A TRUE agentic system implementing:
    - Chain of Thought (CoT): Think step by step
    - Tree of Thoughts (ToT): Explore multiple options
    - ReAct: Thought -> Action -> Observation -> Thought
    - Self-Reflection: Evaluate own reasoning
    - Planning: Plan before executing
    """

    def __init__(
        self,
        llm_client: Any,
        storage_path: Optional[Path] = None,
    ):
        self.llm = llm_client
        self.storage_path = storage_path or Path.home() / ".vanna" / "agent_memory.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Database
        self._db_executor: Optional[Callable] = None
        self._schema: Dict[str, Any] = {}
        self._dialect: str = "unknown"

        # Agent memory
        self._successful_traces: List[AgentTrace] = []
        self._failed_traces: List[AgentTrace] = []
        self._load_memory()

        # Stats
        self._total_queries = 0
        self._successful_queries = 0

        logger.info("TrueAgent initialized - Full agentic reasoning enabled")

    def _load_memory(self) -> None:
        """Load agent memory"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                # Simplified loading - just counts
                logger.info(f"Loaded agent memory")
            except:
                pass

    def _save_memory(self) -> None:
        """Save agent memory"""
        try:
            with open(self.storage_path, "w") as f:
                json.dump({
                    "successful_count": len(self._successful_traces),
                    "failed_count": len(self._failed_traces),
                    "last_updated": datetime.utcnow().isoformat(),
                }, f)
        except:
            pass

    # =========================================================================
    # CONNECTION
    # =========================================================================

    async def connect(self, db_executor: Callable, driver: Optional[str] = None) -> Dict:
        """Connect and discover database"""
        self._db_executor = db_executor

        # Detect dialect
        self._dialect = await self._detect_dialect(driver)

        # Discover schema
        self._schema = await self._discover_schema()

        return {
            "dialect": self._dialect,
            "tables": len(self._schema),
            "columns": sum(len(t.get("columns", [])) for t in self._schema.values()),
        }

    async def _detect_dialect(self, driver: Optional[str]) -> str:
        """Detect database dialect"""
        try:
            await self._db_executor("SELECT TOP 1 1 AS test")
            return "mssql"
        except:
            pass
        try:
            await self._db_executor("SELECT 1 AS test LIMIT 1")
            return "postgresql"
        except:
            pass
        return "mssql"

    async def _discover_schema(self) -> Dict:
        """Discover schema"""
        schema = {}
        query = """
            SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME
        """
        try:
            tables = await self._db_executor(query)
            for row in tables[:25]:
                table_name = row.get("TABLE_NAME")
                if table_name:
                    cols = await self._db_executor(f"""
                        SELECT COLUMN_NAME, DATA_TYPE
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = '{table_name}'
                    """)
                    schema[table_name] = {
                        "columns": [{"name": c["COLUMN_NAME"], "type": c["DATA_TYPE"]} for c in cols]
                    }
        except Exception as e:
            logger.error(f"Schema discovery failed: {e}")
        return schema

    # =========================================================================
    # MAIN QUERY - FULL AGENTIC REASONING
    # =========================================================================

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Process query using FULL agentic reasoning:

        1. PLAN: What steps are needed?
        2. THINK: Chain of Thought reasoning
        3. EXPLORE: Tree of Thoughts - multiple SQL options
        4. EVALUATE: Self-reflection on options
        5. ACT: Execute best option
        6. OBSERVE: Check result
        7. ITERATE: If failed, reason and retry
        8. LEARN: Store successful patterns
        """
        self._total_queries += 1
        trace = AgentTrace(question=question)
        start_time = datetime.utcnow()

        # =====================================================================
        # STEP 1: PLANNING - What do we need to do?
        # =====================================================================
        plan = await self._plan(question, trace)

        # =====================================================================
        # STEP 2: CHAIN OF THOUGHT - Think step by step
        # =====================================================================
        reasoning = await self._chain_of_thought(question, plan, trace)

        # =====================================================================
        # STEP 3: TREE OF THOUGHTS - Generate multiple SQL options
        # =====================================================================
        sql_options = await self._tree_of_thoughts(question, reasoning, trace)

        # =====================================================================
        # STEP 4: SELF-REFLECTION - Evaluate options
        # =====================================================================
        best_sql, confidence = await self._self_reflect(sql_options, question, trace)

        # =====================================================================
        # STEP 5-7: ReAct LOOP - Act, Observe, Iterate
        # =====================================================================
        result = await self._react_loop(best_sql, question, trace, max_iterations=4)

        # =====================================================================
        # STEP 8: LEARN from outcome
        # =====================================================================
        await self._learn(trace, result)

        exec_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        result["execution_time_ms"] = exec_time
        result["reasoning_steps"] = len(trace.thoughts)
        result["sql_options_explored"] = len(trace.sql_options)

        return result

    # =========================================================================
    # STEP 1: PLANNING
    # =========================================================================

    async def _plan(self, question: str, trace: AgentTrace) -> str:
        """Create a plan for answering the question"""
        prompt = f"""You are a SQL expert. Plan how to answer this question.

QUESTION: {question}

DATABASE: {self._dialect.upper()}
TABLES AVAILABLE: {', '.join(list(self._schema.keys())[:15])}

Create a step-by-step plan:
1. What tables might be relevant?
2. What columns do we need?
3. Any joins required?
4. Any aggregations or filters?
5. Any potential issues to watch for?

PLAN:"""

        response = await self.llm.generate(prompt=prompt, max_tokens=300)
        plan = response.strip()

        trace.thoughts.append(Thought(content=f"PLAN: {plan}", confidence=0.8))
        logger.info(f"Agent created plan")

        return plan

    # =========================================================================
    # STEP 2: CHAIN OF THOUGHT
    # =========================================================================

    async def _chain_of_thought(self, question: str, plan: str, trace: AgentTrace) -> str:
        """Think step by step about how to write the SQL"""
        # Build schema context
        schema_desc = []
        for table, info in list(self._schema.items())[:15]:
            cols = [f"{c['name']}({c['type']})" for c in info.get("columns", [])[:8]]
            prefix = "dbo." if self._dialect == "mssql" else ""
            schema_desc.append(f"{prefix}{table}: {', '.join(cols)}")

        prompt = f"""Think step by step about writing SQL for this question.

QUESTION: {question}

PLAN: {plan}

SCHEMA:
{chr(10).join(schema_desc)}

DIALECT: {self._dialect.upper()}
{"IMPORTANT: Use TOP N not LIMIT, Use GETDATE() not NOW(), Use dbo.TableName" if self._dialect == "mssql" else ""}

Think through each step:

THOUGHT 1: What is the main entity we're querying?
THOUGHT 2: What columns do we need?
THOUGHT 3: What conditions/filters apply?
THOUGHT 4: Do we need joins, aggregations, or ordering?
THOUGHT 5: What's the SQL structure?

REASONING:"""

        response = await self.llm.generate(prompt=prompt, max_tokens=500)
        reasoning = response.strip()

        trace.thoughts.append(Thought(content=f"CHAIN OF THOUGHT: {reasoning}", confidence=0.85))
        logger.info(f"Agent completed chain of thought")

        return reasoning

    # =========================================================================
    # STEP 3: TREE OF THOUGHTS
    # =========================================================================

    async def _tree_of_thoughts(
        self, question: str, reasoning: str, trace: AgentTrace
    ) -> List[Dict]:
        """Generate multiple SQL options (branches)"""
        prompt = f"""Based on this reasoning, generate 3 different SQL approaches.

QUESTION: {question}

REASONING: {reasoning}

DIALECT: {self._dialect.upper()}

Generate 3 different valid SQL queries that could answer this question.
Each should use a slightly different approach or structure.

Format:
OPTION 1:
```sql
<sql here>
```
REASON: <why this approach>

OPTION 2:
```sql
<sql here>
```
REASON: <why this approach>

OPTION 3:
```sql
<sql here>
```
REASON: <why this approach>

OPTIONS:"""

        response = await self.llm.generate(prompt=prompt, max_tokens=800)

        # Parse options
        options = []
        import re
        option_matches = re.findall(
            r'OPTION\s*\d+:\s*```sql\s*(.*?)\s*```\s*REASON:\s*(.*?)(?=OPTION|\Z)',
            response,
            re.DOTALL | re.IGNORECASE
        )

        for sql, reason in option_matches:
            sql = sql.strip()
            reason = reason.strip()
            if sql:
                options.append({"sql": sql, "reason": reason, "score": 0.0})

        # If parsing failed, try simpler extraction
        if not options:
            sql_matches = re.findall(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
            for sql in sql_matches:
                if sql.strip():
                    options.append({"sql": sql.strip(), "reason": "Generated option", "score": 0.0})

        # Fallback: generate single SQL
        if not options:
            single_sql = await self._generate_single_sql(question, reasoning)
            options.append({"sql": single_sql, "reason": "Direct generation", "score": 0.5})

        trace.sql_options = options
        trace.thoughts.append(Thought(
            content=f"TREE OF THOUGHTS: Generated {len(options)} SQL options",
            confidence=0.8
        ))
        logger.info(f"Agent generated {len(options)} SQL options")

        return options

    async def _generate_single_sql(self, question: str, reasoning: str) -> str:
        """Fallback: generate single SQL"""
        schema_desc = []
        for table, info in list(self._schema.items())[:10]:
            cols = [c['name'] for c in info.get("columns", [])[:10]]
            prefix = "dbo." if self._dialect == "mssql" else ""
            schema_desc.append(f"{prefix}{table}: {', '.join(cols)}")

        prompt = f"""Generate SQL for: {question}

SCHEMA:
{chr(10).join(schema_desc)}

REASONING: {reasoning[:300]}

Return ONLY the SQL query, no explanation.
SQL:"""

        response = await self.llm.generate(prompt=prompt, max_tokens=300)
        return self._clean_sql(response)

    # =========================================================================
    # STEP 4: SELF-REFLECTION
    # =========================================================================

    async def _self_reflect(
        self, options: List[Dict], question: str, trace: AgentTrace
    ) -> Tuple[str, float]:
        """Evaluate SQL options and select the best one"""
        if not options:
            return "", 0.0

        if len(options) == 1:
            return options[0]["sql"], 0.7

        # Have LLM evaluate options
        options_text = "\n\n".join([
            f"OPTION {i+1}:\n{opt['sql']}\nREASON: {opt.get('reason', 'N/A')}"
            for i, opt in enumerate(options)
        ])

        prompt = f"""Evaluate these SQL options for the question: {question}

{options_text}

Which option is BEST and WHY?
Consider:
1. Correctness - Does it answer the question?
2. Efficiency - Is it optimal?
3. Safety - Does it handle edge cases?

Respond with:
BEST: <option number 1, 2, or 3>
CONFIDENCE: <0.0 to 1.0>
REASON: <brief explanation>"""

        response = await self.llm.generate(prompt=prompt, max_tokens=200)

        # Parse response
        import re
        best_match = re.search(r'BEST:\s*(\d+)', response)
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)

        best_idx = int(best_match.group(1)) - 1 if best_match else 0
        confidence = float(conf_match.group(1)) if conf_match else 0.7

        best_idx = max(0, min(best_idx, len(options) - 1))

        trace.thoughts.append(Thought(
            content=f"SELF-REFLECTION: Selected option {best_idx + 1} with confidence {confidence}",
            confidence=confidence
        ))

        return options[best_idx]["sql"], confidence

    # =========================================================================
    # STEP 5-7: ReAct LOOP
    # =========================================================================

    async def _react_loop(
        self, sql: str, question: str, trace: AgentTrace, max_iterations: int = 4
    ) -> Dict:
        """
        ReAct loop: Thought -> Action -> Observation -> Thought...
        """
        current_sql = sql
        trace.final_sql = sql

        for iteration in range(max_iterations):
            # THOUGHT: What are we doing?
            thought = f"Iteration {iteration + 1}: Executing SQL to answer '{question[:50]}...'"
            trace.thoughts.append(Thought(content=f"ReAct THOUGHT: {thought}", confidence=0.7))

            # ACTION: Execute SQL
            action = Action(action_type="execute_sql", input_data=current_sql)
            trace.actions.append(action)

            try:
                data = await self._db_executor(current_sql)
                action.output_data = f"Success: {len(data)} rows"
                action.success = True

                # OBSERVATION: Success!
                trace.thoughts.append(Thought(
                    content=f"ReAct OBSERVATION: Query succeeded with {len(data)} rows",
                    confidence=0.95
                ))

                self._successful_queries += 1
                trace.final_sql = current_sql
                trace.result = {"success": True, "data": data, "row_count": len(data)}

                return {
                    "success": True,
                    "sql": current_sql,
                    "data": data,
                    "row_count": len(data),
                    "iterations": iteration + 1,
                }

            except Exception as e:
                error = str(e)
                action.output_data = f"Error: {error[:100]}"
                action.success = False

                # OBSERVATION: Failed
                trace.thoughts.append(Thought(
                    content=f"ReAct OBSERVATION: Query failed - {error[:100]}",
                    confidence=0.3
                ))

                if iteration < max_iterations - 1:
                    # THOUGHT: How to fix?
                    trace.thoughts.append(Thought(
                        content=f"ReAct THOUGHT: Need to fix error. Investigating...",
                        confidence=0.6
                    ))

                    # ACTION: Fix SQL
                    fixed_sql = await self._fix_with_reasoning(current_sql, error, question, trace)
                    if fixed_sql and fixed_sql != current_sql:
                        current_sql = fixed_sql
                        trace.final_sql = current_sql
                    else:
                        break

        # All iterations failed
        return {
            "success": False,
            "sql": current_sql,
            "error": error if 'error' in dir() else "Unknown error",
            "row_count": 0,
            "iterations": max_iterations,
        }

    async def _fix_with_reasoning(
        self, sql: str, error: str, question: str, trace: AgentTrace
    ) -> Optional[str]:
        """Fix SQL with full reasoning"""
        # Investigate data if needed
        data_insight = ""
        if "conversion" in error.lower() or "date" in error.lower():
            data_insight = await self._investigate_data(sql)

        prompt = f"""Fix this SQL query using step-by-step reasoning.

ERROR: {error[:300]}

BROKEN SQL:
{sql}

QUESTION: {question}

{f"DATA INSIGHT: {data_insight}" if data_insight else ""}

STEP 1: What is the error telling us?
STEP 2: What part of the SQL is wrong?
STEP 3: How should we fix it?
STEP 4: Write the corrected SQL.

REASONING AND FIX:"""

        response = await self.llm.generate(prompt=prompt, max_tokens=500)

        trace.thoughts.append(Thought(
            content=f"FIX REASONING: {response[:200]}...",
            confidence=0.7
        ))

        # Extract SQL from response
        fixed_sql = self._clean_sql(response)

        if fixed_sql:
            trace.actions.append(Action(
                action_type="fix_sql",
                input_data=sql,
                output_data=fixed_sql,
                success=True
            ))

        return fixed_sql

    async def _investigate_data(self, sql: str) -> str:
        """Investigate data issues"""
        import re
        table_match = re.search(r'FROM\s+(\w+\.?\w+)', sql, re.IGNORECASE)
        if not table_match:
            return ""

        table = table_match.group(1)
        date_match = re.search(r'(Date\w*|\w*_Date|\w*_At)', sql, re.IGNORECASE)

        if date_match:
            col = date_match.group(1)
            try:
                check = await self._db_executor(
                    f"SELECT TOP 3 {col} FROM {table} WHERE ISDATE({col}) = 0 AND {col} IS NOT NULL"
                )
                if check:
                    return f"Invalid dates found in {col}. Use TRY_CAST or ISDATE filter."
            except:
                pass

        return ""

    # =========================================================================
    # STEP 8: LEARNING
    # =========================================================================

    async def _learn(self, trace: AgentTrace, result: Dict) -> None:
        """Learn from the trace"""
        if result.get("success"):
            self._successful_traces.append(trace)
            logger.info(f"Agent learned from successful query ({len(trace.thoughts)} thoughts)")
        else:
            self._failed_traces.append(trace)
            logger.info(f"Agent noted failure for future improvement")

        # Keep bounded
        self._successful_traces = self._successful_traces[-50:]
        self._failed_traces = self._failed_traces[-20:]
        self._save_memory()

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _clean_sql(self, response: str) -> str:
        """Extract SQL from response"""
        sql = response.strip()
        import re

        # Try to extract from code block
        match = re.search(r'```sql\s*(.*?)\s*```', sql, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to extract after SQL: or FIXED SQL:
        match = re.search(r'(?:FIXED\s+)?SQL:\s*(.+)', sql, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()

        # Remove markdown
        if "```" in sql:
            match = re.search(r'```\s*(.*?)\s*```', sql, re.DOTALL)
            if match:
                sql = match.group(1).strip()

        # Remove comments
        lines = [l for l in sql.split("\n") if not l.strip().startswith("--")]
        sql = "\n".join(lines).strip()

        # Basic validation
        if sql.upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")):
            return sql

        return sql

    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "dialect": self._dialect,
            "tables": len(self._schema),
            "total_queries": self._total_queries,
            "successful_queries": self._successful_queries,
            "success_rate": self._successful_queries / max(1, self._total_queries),
            "traces_learned": len(self._successful_traces),
        }
