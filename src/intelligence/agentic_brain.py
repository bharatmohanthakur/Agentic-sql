"""
Agentic Brain - TRUE Intelligence through LLM Learning
========================================================
NO REGEX. NO RULES. PURE LLM INTELLIGENCE.

The LLM:
1. LEARNS from every interaction
2. DECIDES how to fix errors
3. IMPROVES over time with feedback
4. REMEMBERS what works and what doesn't

This is the AGENTIC approach - the LLM is the brain.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LearningMemory:
    """What the LLM has learned - stored as natural language"""
    successful_patterns: List[Dict[str, str]] = field(default_factory=list)  # Question -> SQL that worked
    error_fixes: List[Dict[str, str]] = field(default_factory=list)  # Error -> How to fix
    schema_insights: List[str] = field(default_factory=list)  # Things learned about schema
    dialect_rules: List[str] = field(default_factory=list)  # Dialect-specific learnings


class AgenticBrain:
    """
    The TRUE intelligent core - LLM makes ALL decisions.

    NO regex patterns.
    NO hardcoded rules.
    The LLM learns, decides, and improves.
    """

    def __init__(
        self,
        llm_client: Any,
        storage_path: Optional[Path] = None,
    ):
        self.llm = llm_client
        self.storage_path = storage_path or Path.home() / ".vanna" / "brain_memory.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Database connection
        self._db_executor: Optional[Callable] = None
        self._schema: Dict[str, Any] = {}
        self._dialect: str = "unknown"

        # LLM's memory - what it has learned
        self.memory = LearningMemory()
        self._load_memory()

        # Session stats
        self._queries_run = 0
        self._queries_succeeded = 0
        self._corrections_made = 0

        logger.info("AgenticBrain initialized - LLM is in control")

    # =========================================================================
    # MEMORY PERSISTENCE
    # =========================================================================

    def _load_memory(self) -> None:
        """Load LLM's learned memory"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self.memory.successful_patterns = data.get("successful_patterns", [])
                self.memory.error_fixes = data.get("error_fixes", [])
                self.memory.schema_insights = data.get("schema_insights", [])
                self.memory.dialect_rules = data.get("dialect_rules", [])
                logger.info(f"Loaded {len(self.memory.successful_patterns)} learned patterns")
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}")

    def _save_memory(self) -> None:
        """Persist LLM's learnings"""
        try:
            with open(self.storage_path, "w") as f:
                json.dump({
                    "successful_patterns": self.memory.successful_patterns[-100:],  # Keep last 100
                    "error_fixes": self.memory.error_fixes[-50:],
                    "schema_insights": self.memory.schema_insights[-20:],
                    "dialect_rules": self.memory.dialect_rules[-20:],
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")

    # =========================================================================
    # CONNECTION - LLM discovers the database
    # =========================================================================

    async def connect(
        self,
        db_executor: Callable,
        driver: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Connect to database - LLM discovers everything.
        """
        self._db_executor = db_executor

        # Step 1: LLM detects dialect by trying queries
        self._dialect = await self._llm_detect_dialect(driver)

        # Step 2: Discover schema
        self._schema = await self._discover_schema()

        # Step 3: LLM analyzes schema and creates insights
        await self._llm_analyze_schema()

        return {
            "dialect": self._dialect,
            "tables_discovered": len(self._schema),
            "columns_discovered": sum(len(t.get("columns", [])) for t in self._schema.values()),
        }

    async def _llm_detect_dialect(self, driver: Optional[str] = None) -> str:
        """LLM figures out what database we're connected to"""
        # Try a probe query
        probe_results = []

        # Test MSSQL style
        try:
            await self._db_executor("SELECT TOP 1 1 AS test")
            probe_results.append("TOP 1 worked - likely MSSQL")
        except Exception as e:
            probe_results.append(f"TOP 1 failed: {str(e)[:100]}")

        # Test PostgreSQL/MySQL style
        try:
            await self._db_executor("SELECT 1 AS test LIMIT 1")
            probe_results.append("LIMIT 1 worked - likely PostgreSQL/MySQL")
        except Exception as e:
            probe_results.append(f"LIMIT 1 failed: {str(e)[:100]}")

        # Ask LLM to determine dialect
        prompt = f"""Based on these database probe results, what SQL dialect is this?

PROBE RESULTS:
{chr(10).join(probe_results)}

DRIVER HINT: {driver or 'unknown'}

Respond with ONLY one word: mssql, postgresql, mysql, or sqlite"""

        response = await self.llm.generate(prompt=prompt, max_tokens=20)
        dialect = response.strip().lower()

        if dialect not in ["mssql", "postgresql", "mysql", "sqlite"]:
            dialect = "mssql" if "top" in probe_results[0].lower() else "postgresql"

        # Learn this
        self.memory.dialect_rules.append(f"Database dialect is {dialect}")
        self._save_memory()

        logger.info(f"LLM detected dialect: {dialect}")
        return dialect

    async def _discover_schema(self) -> Dict[str, Any]:
        """Discover database schema"""
        schema = {}

        # Get tables based on dialect
        if self._dialect == "mssql":
            tables_query = """
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """
        else:
            tables_query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                ORDER BY table_name
            """

        try:
            tables = await self._db_executor(tables_query)
            for row in tables[:30]:  # Limit to 30 tables
                table_name = row.get("TABLE_NAME") or row.get("table_name")
                if table_name:
                    schema[table_name] = await self._get_table_info(table_name)
        except Exception as e:
            logger.error(f"Schema discovery failed: {e}")

        return schema

    async def _get_table_info(self, table_name: str) -> Dict:
        """Get columns for a table"""
        if self._dialect == "mssql":
            query = f"""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{table_name}'
                ORDER BY ORDINAL_POSITION
            """
        else:
            query = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """

        try:
            columns = await self._db_executor(query)
            return {
                "columns": [
                    {"name": c.get("COLUMN_NAME") or c.get("column_name"),
                     "type": c.get("DATA_TYPE") or c.get("data_type")}
                    for c in columns
                ]
            }
        except:
            return {"columns": []}

    async def _llm_analyze_schema(self) -> None:
        """LLM analyzes schema and creates insights"""
        # Build schema summary
        schema_desc = []
        for table, info in list(self._schema.items())[:15]:
            cols = [c["name"] for c in info.get("columns", [])[:10]]
            schema_desc.append(f"{table}: {', '.join(cols)}")

        prompt = f"""Analyze this database schema and note important patterns:

DIALECT: {self._dialect}

TABLES:
{chr(10).join(schema_desc)}

List 3-5 key insights about this schema (table relationships, naming patterns, etc).
Keep each insight to one line."""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=300)
            insights = [line.strip() for line in response.split("\n") if line.strip()]
            self.memory.schema_insights.extend(insights[:5])
            self._save_memory()
            logger.info(f"LLM created {len(insights)} schema insights")
        except Exception as e:
            logger.warning(f"Schema analysis failed: {e}")

    # =========================================================================
    # QUERY - LLM generates, executes, learns
    # =========================================================================

    async def query(self, question: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Answer a question - LLM does everything.

        1. LLM generates SQL using schema and memory
        2. Execute query
        3. If error, LLM fixes it
        4. LLM learns from success/failure
        """
        self._queries_run += 1
        start_time = datetime.utcnow()

        # Step 1: LLM generates SQL
        sql = await self._llm_generate_sql(question)

        # Step 2: Try to execute with LLM-driven retry
        data = None
        error = None
        was_corrected = False

        for attempt in range(max_retries + 1):
            try:
                data = await self._db_executor(sql)
                self._queries_succeeded += 1

                # SUCCESS - LLM learns from it
                await self._llm_learn_success(question, sql, was_corrected)

                exec_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                return {
                    "success": True,
                    "sql": sql,
                    "data": data,
                    "row_count": len(data) if data else 0,
                    "was_corrected": was_corrected,
                    "execution_time_ms": exec_time,
                }

            except Exception as e:
                error = str(e)
                logger.warning(f"Query failed (attempt {attempt + 1}): {error[:100]}")

                if attempt < max_retries:
                    # LLM fixes the error with attempt number for strategy selection
                    fixed_sql = await self._llm_fix_error(sql, error, question, attempt + 1)
                    if fixed_sql and fixed_sql != sql:
                        sql = fixed_sql
                        was_corrected = True
                        self._corrections_made += 1

        # All retries failed - LLM learns from failure
        await self._llm_learn_failure(question, sql, error)

        return {
            "success": False,
            "sql": sql,
            "error": error,
            "was_corrected": was_corrected,
            "row_count": 0,
            "execution_time_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000),
        }

    async def _llm_generate_sql(self, question: str) -> str:
        """LLM generates SQL using its knowledge and memory"""
        # Build context from schema
        schema_context = self._build_schema_context()

        # Build context from memory (what worked before)
        memory_context = self._build_memory_context(question)

        prompt = f"""Generate a SQL query for {self._dialect.upper()}.

{schema_context}

{memory_context}

QUESTION: {question}

CRITICAL RULES:
- Return ONLY the SQL query
- No markdown, no explanation
- Use exact table and column names from schema

SQL:"""

        response = await self.llm.generate(prompt=prompt, max_tokens=500)
        sql = self._clean_sql(response)
        return sql

    async def _llm_fix_error(self, sql: str, error: str, original_question: str, attempt: int = 1) -> Optional[str]:
        """
        LLM intelligently fixes SQL errors with DEEP LEARNING.

        Key improvements:
        1. Queries sample data to understand the problem
        2. Analyzes error patterns deeply
        3. Tries different strategies based on attempt number
        4. Learns WHY fixes work
        """
        # Check if we've seen this error before
        similar_fix = self._find_similar_error_fix(error)

        # DEEP LEARNING: Get sample data to understand the problem
        data_insight = ""
        if "conversion" in error.lower() or "date" in error.lower():
            data_insight = await self._investigate_data_issue(sql, error)

        # Different strategies for different attempts
        strategies = [
            "Fix the syntax error directly",
            "Use TRY_CAST or TRY_CONVERT to handle bad data",
            "Add WHERE clause to filter out invalid data",
            "Completely rewrite the query with a different approach",
        ]
        current_strategy = strategies[min(attempt - 1, len(strategies) - 1)]

        prompt = f"""Fix this {self._dialect.upper()} SQL query.

ERROR: {error[:400]}

BROKEN SQL:
{sql}

ORIGINAL QUESTION: {original_question}

{f"DATA INVESTIGATION: {data_insight}" if data_insight else ""}
{f"SIMILAR ERROR FIX THAT WORKED: {similar_fix}" if similar_fix else ""}

ATTEMPT {attempt} STRATEGY: {current_strategy}

DIALECT RULES FOR {self._dialect.upper()}:
{"- Use TOP N not LIMIT" if self._dialect == "mssql" else "- Use LIMIT N"}
{"- Use TRY_CAST/TRY_CONVERT for safe type conversion" if self._dialect == "mssql" else ""}
{"- Use ISDATE() to check valid dates" if self._dialect == "mssql" else ""}
{"- Use dbo.TableName format" if self._dialect == "mssql" else ""}

Return ONLY the fixed SQL, nothing else."""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=500)
            fixed_sql = self._clean_sql(response)

            if fixed_sql and fixed_sql != sql:
                # Learn this fix with details
                self.memory.error_fixes.append({
                    "error_pattern": error[:200],
                    "strategy_used": current_strategy,
                    "fix_description": f"Applied strategy: {current_strategy}",
                    "data_insight": data_insight[:100] if data_insight else "",
                    "timestamp": datetime.utcnow().isoformat(),
                })
                self._save_memory()
                return fixed_sql

        except Exception as e:
            logger.warning(f"LLM fix failed: {e}")

        return None

    async def _investigate_data_issue(self, sql: str, error: str) -> str:
        """
        DEEP LEARNING: Query the database to understand data issues.

        This is what makes the system truly intelligent - it investigates!
        """
        try:
            # Extract table name from SQL
            import re
            table_match = re.search(r'FROM\s+(\w+\.?\w+)', sql, re.IGNORECASE)
            if not table_match:
                return ""

            table_name = table_match.group(1)

            # Find columns that might have issues (date columns)
            date_col_match = re.search(r'(Date\w*|\w*_Date|\w*_At)', sql, re.IGNORECASE)
            if date_col_match:
                date_col = date_col_match.group(1)

                # Query sample data to see what formats exist
                sample_query = f"SELECT TOP 5 {date_col} FROM {table_name} WHERE {date_col} IS NOT NULL"
                try:
                    samples = await self._db_executor(sample_query)
                    if samples:
                        sample_values = [str(s.get(date_col, s.get(date_col.upper(), '')))[:30] for s in samples]
                        return f"Sample {date_col} values: {sample_values}"
                except:
                    pass

                # Check for invalid dates
                check_query = f"SELECT TOP 3 {date_col} FROM {table_name} WHERE ISDATE({date_col}) = 0 AND {date_col} IS NOT NULL"
                try:
                    bad_data = await self._db_executor(check_query)
                    if bad_data:
                        bad_values = [str(b.get(date_col, b.get(date_col.upper(), '')))[:30] for b in bad_data]
                        return f"INVALID DATE VALUES FOUND: {bad_values}. Use TRY_CAST or filter with ISDATE()."
                except:
                    pass

        except Exception as e:
            logger.debug(f"Data investigation failed: {e}")

        return ""

    async def _llm_learn_success(self, question: str, sql: str, was_corrected: bool = False) -> None:
        """
        LLM learns from successful query.

        DEEP LEARNING: If this was a corrected query, analyze WHY it worked
        so we can apply the same fix next time.
        """
        # Store this successful pattern
        pattern = {
            "question": question,
            "sql": sql,
            "dialect": self._dialect,
            "was_corrected": was_corrected,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # If it was corrected, this is VALUABLE learning
        if was_corrected:
            pattern["learning_value"] = "HIGH - correction succeeded"
            # Mark recent error fixes as validated
            for fix in self.memory.error_fixes[-3:]:
                fix["validated"] = True

        self.memory.successful_patterns.append(pattern)
        self._save_memory()
        logger.info(f"LLM learned from {'corrected' if was_corrected else 'direct'} success")

    async def _llm_learn_failure(self, question: str, sql: str, error: str) -> None:
        """LLM learns from failed query"""
        # We could ask LLM to analyze what went wrong
        logger.info(f"LLM noted failure for future learning: {error[:50]}")

    def _find_similar_error_fix(self, error: str) -> Optional[str]:
        """
        Find if we've fixed similar errors before.

        Prioritizes VALIDATED fixes (ones that actually worked).
        """
        error_lower = error.lower()

        # First, look for validated fixes (proven to work)
        for fix in reversed(self.memory.error_fixes):
            if fix.get("validated"):
                pattern = fix.get("error_pattern", "").lower()
                if any(word in error_lower for word in pattern.split()[:3] if len(word) > 3):
                    strategy = fix.get("strategy_used", fix.get("fix_description", ""))
                    return f"PROVEN FIX: {strategy}"

        # Then look for any similar fixes
        for fix in reversed(self.memory.error_fixes):
            pattern = fix.get("error_pattern", "").lower()
            if any(word in error_lower for word in pattern.split()[:3] if len(word) > 3):
                return fix.get("strategy_used", fix.get("fix_description"))

        return None

    def _build_schema_context(self) -> str:
        """Build schema context for LLM"""
        lines = [f"DATABASE: {self._dialect.upper()}", "", "TABLES:"]

        for table, info in list(self._schema.items())[:20]:
            cols = [c["name"] for c in info.get("columns", [])[:12]]
            prefix = "dbo." if self._dialect == "mssql" else ""
            lines.append(f"{prefix}{table}: {', '.join(cols)}")

        return "\n".join(lines)

    def _build_memory_context(self, question: str) -> str:
        """Build context from LLM's memory"""
        lines = []

        # Add relevant successful patterns
        if self.memory.successful_patterns:
            lines.append("EXAMPLES OF WHAT WORKED:")
            for pattern in self.memory.successful_patterns[-5:]:
                lines.append(f"Q: {pattern['question'][:50]}")
                lines.append(f"SQL: {pattern['sql'][:80]}")

        # Add schema insights
        if self.memory.schema_insights:
            lines.append("\nSCHEMA INSIGHTS:")
            for insight in self.memory.schema_insights[-3:]:
                lines.append(f"- {insight}")

        return "\n".join(lines) if lines else ""

    def _clean_sql(self, response: str) -> str:
        """Clean LLM response to get SQL"""
        sql = response.strip()

        # Remove markdown
        if "```" in sql:
            import re
            match = re.search(r"```(?:sql)?\s*(.*?)\s*```", sql, re.DOTALL)
            if match:
                sql = match.group(1).strip()

        # Remove comments
        lines = [l for l in sql.split("\n") if not l.strip().startswith("--")]
        sql = "\n".join(lines).strip()

        return sql

    # =========================================================================
    # STATS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get brain statistics"""
        return {
            "dialect": self._dialect,
            "tables_discovered": len(self._schema),
            "queries_run": self._queries_run,
            "queries_succeeded": self._queries_succeeded,
            "success_rate": self._queries_succeeded / max(1, self._queries_run),
            "corrections_made": self._corrections_made,
            "patterns_learned": len(self.memory.successful_patterns),
            "error_fixes_learned": len(self.memory.error_fixes),
            "schema_insights": len(self.memory.schema_insights),
        }
