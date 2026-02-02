"""
META-LEARNING AGENT - PURE INTELLIGENCE
========================================

This agent is TRULY intelligent:
- NO fixed prompts - prompts are DESIGNED per problem
- NO fixed rules - LLM learns everything
- NO regex patterns - LLM understands context
- NO hardcoded syntax - LLM knows SQL dialects

EVERYTHING is learned. EVERYTHING is dynamic.

THINK → RESEARCH → DESIGN → EXECUTE → LEARN
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
class ProblemType:
    """Learned problem classification"""
    name: str  # Dynamically learned type names
    indicators: List[str]  # LLM-identified indicators
    successful_approaches: List[str]  # What worked
    failed_approaches: List[str]  # What didn't work
    success_rate: float = 0.0
    attempts: int = 0


@dataclass
class LearnedSolution:
    """A solution that worked"""
    question: str
    problem_type: str
    sql: str
    was_corrected: bool
    corrections_applied: List[str]
    timestamp: str


@dataclass
class LearnedFailure:
    """A failure to learn from"""
    question: str
    problem_type: str
    sql: str
    error: str
    llm_analysis: str
    timestamp: str


@dataclass
class MetaKnowledge:
    """Meta-learning knowledge base - ALL learned, nothing hardcoded"""
    problem_types: Dict[str, ProblemType] = field(default_factory=dict)
    successful_solutions: List[LearnedSolution] = field(default_factory=list)
    failed_attempts: List[LearnedFailure] = field(default_factory=list)
    dialect_learnings: List[str] = field(default_factory=list)  # LLM's dialect discoveries
    schema_insights: List[str] = field(default_factory=list)  # LLM's schema discoveries
    fix_strategies: List[Dict] = field(default_factory=list)  # What fixes worked
    # TRUE LEARNING - Corrections that apply to future queries
    name_corrections: Dict[str, str] = field(default_factory=dict)  # wrong_name -> correct_name
    table_relationships: Dict[str, str] = field(default_factory=dict)  # table -> related_tables
    column_mappings: Dict[str, str] = field(default_factory=dict)  # concept -> actual_column


class MetaAgent:
    """
    META-LEARNING AGENT - PURE INTELLIGENCE

    This agent:
    1. THINKS - Analyzes each problem uniquely
    2. RESEARCHES - Finds what worked for similar problems
    3. DESIGNS - Creates custom approach (dynamic prompts!)
    4. EXECUTES - Runs with intelligent error recovery
    5. LEARNS - Updates knowledge from EVERY interaction

    NO RULES. NO PATTERNS. PURE LLM INTELLIGENCE.
    """

    def __init__(self, llm_client: Any, storage_path: Optional[Path] = None):
        self.llm = llm_client
        self.storage_path = storage_path or Path.home() / ".vanna" / "meta_agent.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Database
        self._db_executor: Optional[Callable] = None
        self._schema: Dict[str, Any] = {}
        self._dialect: str = "unknown"

        # Meta-learning knowledge - starts EMPTY, learns everything
        self.knowledge = MetaKnowledge()
        self._load_knowledge()

        logger.info("MetaAgent initialized - Pure intelligence, no rules")

    # =========================================================================
    # KNOWLEDGE PERSISTENCE
    # =========================================================================

    def _load_knowledge(self):
        """Load learned knowledge"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)

                # Load successful solutions
                for sol in data.get("successful_solutions", []):
                    self.knowledge.successful_solutions.append(LearnedSolution(**sol))

                # Load failed attempts
                for fail in data.get("failed_attempts", []):
                    self.knowledge.failed_attempts.append(LearnedFailure(**fail))

                # Load dialect learnings
                self.knowledge.dialect_learnings = data.get("dialect_learnings", [])

                # Load schema insights
                self.knowledge.schema_insights = data.get("schema_insights", [])

                # Load fix strategies
                self.knowledge.fix_strategies = data.get("fix_strategies", [])

                # Load TRUE LEARNING corrections
                self.knowledge.name_corrections = data.get("name_corrections", {})
                self.knowledge.table_relationships = data.get("table_relationships", {})
                self.knowledge.column_mappings = data.get("column_mappings", {})

                logger.info(f"Loaded {len(self.knowledge.successful_solutions)} solutions, "
                           f"{len(self.knowledge.failed_attempts)} failures, "
                           f"{len(self.knowledge.name_corrections)} name corrections")
            except Exception as e:
                logger.warning(f"Failed to load knowledge: {e}")

    def _save_knowledge(self):
        """Save learned knowledge"""
        try:
            with open(self.storage_path, "w") as f:
                json.dump({
                    "successful_solutions": [
                        {
                            "question": s.question,
                            "problem_type": s.problem_type,
                            "sql": s.sql,
                            "was_corrected": s.was_corrected,
                            "corrections_applied": s.corrections_applied,
                            "timestamp": s.timestamp,
                        }
                        for s in self.knowledge.successful_solutions[-100:]
                    ],
                    "failed_attempts": [
                        {
                            "question": f.question,
                            "problem_type": f.problem_type,
                            "sql": f.sql,
                            "error": f.error,
                            "llm_analysis": f.llm_analysis,
                            "timestamp": f.timestamp,
                        }
                        for f in self.knowledge.failed_attempts[-50:]
                    ],
                    "dialect_learnings": self.knowledge.dialect_learnings[-20:],
                    "schema_insights": self.knowledge.schema_insights[-20:],
                    "fix_strategies": self.knowledge.fix_strategies[-30:],
                    # TRUE LEARNING - Applied corrections
                    "name_corrections": self.knowledge.name_corrections,
                    "table_relationships": self.knowledge.table_relationships,
                    "column_mappings": self.knowledge.column_mappings,
                    "timestamp": datetime.utcnow().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save knowledge: {e}")

    # =========================================================================
    # CONNECTION - LLM DISCOVERS EVERYTHING
    # =========================================================================

    async def connect(self, db_executor: Callable, driver: Optional[str] = None) -> Dict:
        """Connect to database - LLM discovers everything"""
        self._db_executor = db_executor

        # LLM detects dialect by probing
        self._dialect = await self._llm_detect_dialect(driver)

        # Discover schema
        self._schema = await self._discover_schema()

        # LLM analyzes schema to create insights
        await self._llm_analyze_schema()

        return {"dialect": self._dialect, "tables": len(self._schema)}

    async def _llm_detect_dialect(self, driver: Optional[str] = None) -> str:
        """LLM figures out what database we're connected to"""
        probe_results = []

        # Try different probe queries
        probes = [
            ("SELECT TOP 1 1 AS test", "TOP syntax"),
            ("SELECT 1 AS test LIMIT 1", "LIMIT syntax"),
            ("SELECT GETDATE()", "GETDATE function"),
            ("SELECT NOW()", "NOW function"),
        ]

        for query, description in probes:
            try:
                await self._db_executor(query)
                probe_results.append(f"✓ {description} worked")
            except Exception as e:
                probe_results.append(f"✗ {description} failed: {str(e)[:50]}")

        # LLM determines dialect from probe results
        prompt = f"""Based on these database probe results, determine the SQL dialect.

PROBE RESULTS:
{chr(10).join(probe_results)}

DRIVER HINT: {driver or 'unknown'}

What SQL dialect is this database? Think about what worked and what failed.

Respond with:
DIALECT: <mssql/postgresql/mysql/sqlite>
REASONING: <why you determined this>
KEY_LEARNINGS: <what syntax works in this dialect>"""

        response = await self.llm.generate(prompt=prompt, max_tokens=200)

        # Extract dialect
        dialect = "mssql"  # Default if parsing fails
        for line in response.split('\n'):
            if line.strip().upper().startswith('DIALECT:'):
                detected = line.split(':', 1)[1].strip().lower()
                if detected in ["mssql", "postgresql", "mysql", "sqlite"]:
                    dialect = detected
                    break

        # Store LLM's learnings about this dialect
        self.knowledge.dialect_learnings.append(response)
        self._save_knowledge()

        logger.info(f"LLM detected dialect: {dialect}")
        return dialect

    async def _discover_schema(self) -> Dict:
        """Discover database schema"""
        schema = {}

        # Query for tables - LLM will learn the right query format
        try:
            tables = await self._db_executor("""
                SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME
            """)

            for row in tables[:25]:
                table_name = row.get("TABLE_NAME") or row.get("table_name")
                if table_name:
                    cols = await self._db_executor(f"""
                        SELECT COLUMN_NAME, DATA_TYPE
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = '{table_name}'
                    """)
                    schema[table_name] = {
                        "columns": [
                            {"name": c.get("COLUMN_NAME") or c.get("column_name"),
                             "type": c.get("DATA_TYPE") or c.get("data_type")}
                            for c in cols
                        ]
                    }
        except Exception as e:
            logger.error(f"Schema discovery failed: {e}")

        return schema

    async def _llm_analyze_schema(self) -> None:
        """LLM analyzes schema and creates insights - no rules, pure learning"""
        schema_desc = []
        for table, info in list(self._schema.items())[:15]:
            cols = [f"{c['name']}({c['type']})" for c in info.get("columns", [])[:10]]
            schema_desc.append(f"{table}: {', '.join(cols)}")

        prompt = f"""Analyze this database schema and learn important patterns.

DATABASE DIALECT: {self._dialect}

TABLES AND COLUMNS:
{chr(10).join(schema_desc)}

Discover and learn:
1. What are the main entities/tables?
2. How do tables relate to each other? (look for ID columns, naming patterns)
3. What columns might have data type issues (dates, nulls)?
4. What naming conventions are used?
5. Any potential join relationships?

Provide insights as bullet points:"""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=400)
            insights = [line.strip() for line in response.split('\n') if line.strip().startswith('-')]
            self.knowledge.schema_insights.extend(insights[:10])
            self._save_knowledge()
            logger.info(f"LLM discovered {len(insights)} schema insights")
        except Exception as e:
            logger.warning(f"Schema analysis failed: {e}")

    # =========================================================================
    # MAIN QUERY - META-LEARNING APPROACH
    # =========================================================================

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Process query using META-LEARNING:

        1. THINK: Analyze problem (LLM classifies, no fixed types)
        2. RESEARCH: Find similar problems that worked
        3. DESIGN: Create DYNAMIC prompt based on problem + research
        4. EXECUTE: Run with intelligent retry
        5. LEARN: Update knowledge from outcome
        """
        start_time = datetime.utcnow()
        trace = {"question": question, "steps": [], "corrections": []}

        # THINK: LLM analyzes the problem
        problem_analysis = await self._think(question, trace)

        # RESEARCH: Find what worked for similar problems
        research = await self._research(question, problem_analysis, trace)

        # DESIGN: Create dynamic approach based on research
        approach = await self._design(question, problem_analysis, research, trace)

        # EXECUTE: Run with intelligent error handling
        result = await self._execute(question, approach, trace)

        # LEARN: Update knowledge
        await self._learn(question, trace, result)

        exec_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        result["execution_time_ms"] = exec_time
        result["steps_taken"] = len(trace["steps"])
        result["problem_type"] = problem_analysis.get("type", "unknown")

        return result

    # =========================================================================
    # THINK: LLM Analyzes the Problem
    # =========================================================================

    async def _think(self, question: str, trace: Dict) -> Dict:
        """
        THINK phase: LLM analyzes the problem.
        NO fixed problem types - LLM determines everything.
        """
        prompt = f"""Analyze this question to understand what SQL solution is needed.

QUESTION: {question}

Think deeply about:
1. What is being asked? (the core intent)
2. What data entities are involved?
3. What operations are needed? (count, filter, join, aggregate, etc.)
4. What might be challenging about this query?
5. What have we learned before that might help?

PREVIOUS LEARNINGS (if relevant):
{chr(10).join(self.knowledge.dialect_learnings[-3:]) if self.knowledge.dialect_learnings else "No previous learnings yet."}

Respond:
TYPE: <brief problem classification>
ENTITIES: <tables/data involved>
OPERATIONS: <what SQL operations>
CHALLENGES: <potential issues>
APPROACH_HINT: <suggested approach>"""

        response = await self.llm.generate(prompt=prompt, max_tokens=300)

        # Extract type from response
        problem_type = "general"
        for line in response.split('\n'):
            if line.strip().upper().startswith('TYPE:'):
                problem_type = line.split(':', 1)[1].strip().lower()[:30]
                break

        analysis = {
            "type": problem_type,
            "raw_analysis": response,
        }

        trace["steps"].append({"phase": "THINK", "analysis": analysis})
        logger.info(f"THINK: LLM classified problem as '{problem_type}'")

        return analysis

    # =========================================================================
    # RESEARCH: Find What Worked Before
    # =========================================================================

    async def _research(self, question: str, problem_analysis: Dict, trace: Dict) -> Dict:
        """
        RESEARCH phase: Find what approaches worked for similar problems.
        LLM searches through past successes and failures.
        """
        problem_type = problem_analysis.get("type", "unknown")

        # Find similar successful solutions
        similar_solutions = []
        for sol in self.knowledge.successful_solutions[-30:]:
            # LLM will match by similarity, not exact type
            similar_solutions.append({
                "question": sol.question[:60],
                "sql": sol.sql[:100],
                "type": sol.problem_type,
                "was_corrected": sol.was_corrected,
            })

        # Find relevant failures to avoid
        relevant_failures = []
        for fail in self.knowledge.failed_attempts[-10:]:
            relevant_failures.append({
                "question": fail.question[:40],
                "error": fail.error[:60],
                "analysis": fail.llm_analysis[:80],
            })

        # Find fix strategies that worked
        working_fixes = self.knowledge.fix_strategies[-5:]

        research = {
            "similar_solutions": similar_solutions[-5:],
            "relevant_failures": relevant_failures[-3:],
            "working_fixes": working_fixes,
            "dialect_learnings": self.knowledge.dialect_learnings[-3:],
            "schema_insights": self.knowledge.schema_insights[-5:],
        }

        trace["steps"].append({"phase": "RESEARCH", "findings_count": len(similar_solutions)})
        logger.info(f"RESEARCH: Found {len(similar_solutions)} similar solutions")

        return research

    # =========================================================================
    # DESIGN: Create Dynamic Approach
    # =========================================================================

    async def _design(
        self, question: str, problem_analysis: Dict, research: Dict, trace: Dict
    ) -> Dict:
        """
        DESIGN phase: Create a DYNAMIC, CUSTOM approach.
        The prompt itself is DESIGNED based on the problem and research.
        NO fixed prompts. NO hardcoded rules.
        """
        # Build schema context
        schema_lines = []
        for table, info in list(self._schema.items())[:15]:
            cols = [c["name"] for c in info.get("columns", [])[:10]]
            schema_lines.append(f"{table}: {', '.join(cols)}")

        # Build dynamic context from research
        examples_context = ""
        if research.get("similar_solutions"):
            examples_context = "SIMILAR QUERIES THAT WORKED:\n"
            for sol in research["similar_solutions"][:3]:
                examples_context += f"- Q: {sol['question']}\n  SQL: {sol['sql']}\n"

        failures_context = ""
        if research.get("relevant_failures"):
            failures_context = "MISTAKES TO AVOID (learned from failures):\n"
            for fail in research["relevant_failures"][:2]:
                failures_context += f"- {fail['error']} - {fail['analysis']}\n"

        # TRUE LEARNING: Apply name corrections we learned
        corrections_context = ""
        if self.knowledge.name_corrections:
            corrections_context = "IMPORTANT - LEARNED NAME CORRECTIONS (use correct names!):\n"
            for wrong, correct in self.knowledge.name_corrections.items():
                corrections_context += f"- Do NOT use '{wrong}', use '{correct}' instead\n"

        # TRUE LEARNING: Apply table relationship learnings
        table_context = ""
        if self.knowledge.table_relationships:
            table_context = "IMPORTANT - LEARNED TABLE MAPPINGS (use correct tables!):\n"
            for concept, table in self.knowledge.table_relationships.items():
                table_context += f"- For '{concept}' queries, use table '{table}'\n"

        dialect_context = ""
        if research.get("dialect_learnings"):
            dialect_context = "DIALECT LEARNINGS:\n" + "\n".join(research["dialect_learnings"][:2])

        schema_context = ""
        if research.get("schema_insights"):
            schema_context = "SCHEMA INSIGHTS:\n" + "\n".join(research["schema_insights"][:3])

        # THE KEY: Dynamic prompt designed for THIS specific problem
        prompt = f"""Design a SQL solution for this specific problem.

QUESTION: {question}

PROBLEM ANALYSIS:
{problem_analysis.get('raw_analysis', '')}

DATABASE DIALECT: {self._dialect}

SCHEMA:
{chr(10).join(schema_lines)}

{examples_context}

{failures_context}

{corrections_context}

{table_context}

{dialect_context}

{schema_context}

Based on ALL the above context:
1. What SQL approach will work best?
2. What syntax is correct for {self._dialect}?
3. What potential issues should be handled?

DESIGN YOUR SOLUTION:

APPROACH:
<your reasoning about the best approach>

SQL:
<your SQL query>"""

        response = await self.llm.generate(prompt=prompt, max_tokens=700)

        # Extract SQL from response
        sql = self._extract_sql(response)

        approach = {
            "prompt_was_dynamic": True,
            "response": response,
            "sql": sql,
            "problem_type": problem_analysis.get("type", "unknown"),
        }

        trace["steps"].append({"phase": "DESIGN", "sql_generated": sql[:100] if sql else None})
        logger.info(f"DESIGN: Created custom solution")

        return approach

    # =========================================================================
    # EXECUTE: Run with Intelligent Error Handling
    # =========================================================================

    async def _execute(self, question: str, approach: Dict, trace: Dict) -> Dict:
        """
        EXECUTE phase: Run with LLM-driven error handling.
        NO hardcoded error patterns. LLM figures out how to fix.
        """
        sql = approach.get("sql", "")
        problem_type = approach.get("problem_type", "unknown")

        for iteration in range(4):
            trace["steps"].append({
                "phase": "EXECUTE",
                "iteration": iteration + 1,
                "sql": sql[:100],
            })

            try:
                data = await self._db_executor(sql)

                return {
                    "success": True,
                    "sql": sql,
                    "data": data,
                    "row_count": len(data) if data else 0,
                    "iterations": iteration + 1,
                }

            except Exception as e:
                error = str(e)
                logger.warning(f"EXECUTE failed (iteration {iteration + 1}): {error[:80]}")

                if iteration < 3:
                    # LLM fixes the error - no hardcoded patterns
                    sql = await self._llm_fix_error(sql, error, question, problem_type, trace, iteration + 1)

        return {
            "success": False,
            "sql": sql,
            "error": error,
            "row_count": 0,
            "iterations": 4,
        }

    async def _llm_fix_error(
        self, sql: str, error: str, question: str, problem_type: str, trace: Dict, attempt: int
    ) -> str:
        """
        Fix errors using PURE LLM INTELLIGENCE.
        NO hardcoded patterns. LLM analyzes and fixes.
        """
        # Get data insight if LLM thinks it would help
        data_insight = await self._llm_investigate(sql, error)

        # Get relevant fix strategies that worked before
        fix_hints = []
        for fix in self.knowledge.fix_strategies[-5:]:
            if fix.get("worked"):
                fix_hints.append(f"- {fix.get('strategy', '')}: {fix.get('description', '')[:50]}")

        # Different strategies for different attempts
        strategy_prompt = ""
        if attempt == 1:
            strategy_prompt = "First, try to fix the syntax error directly."
        elif attempt == 2:
            strategy_prompt = "Try a safer approach - use type conversion functions if data issues."
        elif attempt == 3:
            strategy_prompt = "Try filtering out problematic data in WHERE clause."
        else:
            strategy_prompt = "Completely rethink the approach - try a different SQL structure."

        prompt = f"""Fix this SQL query. You are an expert in {self._dialect.upper()} databases.

ERROR MESSAGE:
{error[:500]}

FAILED SQL:
{sql}

ORIGINAL QUESTION: {question}

{f"DATA INVESTIGATION: {data_insight}" if data_insight else ""}

{f"FIX STRATEGIES THAT WORKED BEFORE:{chr(10)}{chr(10).join(fix_hints)}" if fix_hints else ""}

ATTEMPT {attempt} STRATEGY: {strategy_prompt}

Think about:
1. What exactly is the error telling us?
2. What part of the SQL is causing it?
3. How do we fix it correctly for {self._dialect.upper()}?

Return ONLY the corrected SQL query, nothing else."""

        response = await self.llm.generate(prompt=prompt, max_tokens=500)
        fixed_sql = self._extract_sql(response)

        # Record this fix attempt
        trace["corrections"].append({
            "attempt": attempt,
            "error": error[:100],
            "strategy": strategy_prompt,
            "fixed_sql": fixed_sql[:100] if fixed_sql else None,
        })

        return fixed_sql if fixed_sql else sql

    async def _llm_investigate(self, sql: str, error: str) -> str:
        """LLM decides what to investigate - no hardcoded queries"""
        prompt = f"""Given this SQL error, should we investigate the data?

SQL: {sql[:200]}
ERROR: {error[:200]}
DATABASE: {self._dialect}

If investigation would help, suggest a simple diagnostic query.
If no investigation needed, say "NONE".

Response:"""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=150)

            if "NONE" in response.upper():
                return ""

            # If LLM suggested a query, try to run it
            suggested_query = self._extract_sql(response)
            if suggested_query and "SELECT" in suggested_query.upper():
                try:
                    result = await self._db_executor(suggested_query)
                    if result:
                        return f"Investigation found: {str(result[:3])[:200]}"
                except Exception as e:
                    return f"Investigation query failed: {str(e)[:50]}"
        except:
            pass

        return ""

    # =========================================================================
    # LEARN: Update Knowledge
    # =========================================================================

    async def _learn(self, question: str, trace: Dict, result: Dict) -> None:
        """
        LEARN phase: Update knowledge from EVERY interaction.
        Learn from success AND failure.
        """
        problem_type = result.get("problem_type", "unknown")
        success = result.get("success", False)

        if success:
            # Store successful solution
            solution = LearnedSolution(
                question=question,
                problem_type=problem_type,
                sql=result.get("sql", ""),
                was_corrected=result.get("iterations", 1) > 1,
                corrections_applied=[c.get("strategy", "") for c in trace.get("corrections", [])],
                timestamp=datetime.utcnow().isoformat(),
            )
            self.knowledge.successful_solutions.append(solution)

            # If corrections worked, learn from them
            if trace.get("corrections"):
                for correction in trace["corrections"]:
                    self.knowledge.fix_strategies.append({
                        "error_type": correction.get("error", "")[:50],
                        "strategy": correction.get("strategy", ""),
                        "worked": True,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            logger.info(f"LEARN: Stored successful solution for '{problem_type}'")

        else:
            # Analyze and learn from failure
            failure_analysis = await self._analyze_failure(question, result, trace)

            failure = LearnedFailure(
                question=question,
                problem_type=problem_type,
                sql=result.get("sql", ""),
                error=result.get("error", "")[:200],
                llm_analysis=failure_analysis,
                timestamp=datetime.utcnow().isoformat(),
            )
            self.knowledge.failed_attempts.append(failure)

            logger.info(f"LEARN: Analyzed failure for future improvement")

        self._save_knowledge()

    async def _analyze_failure(self, question: str, result: Dict, trace: Dict) -> str:
        """LLM analyzes why a query failed - AND extracts actionable corrections"""
        error = result.get("error", "")
        sql = result.get("sql", "")

        # STEP 1: Extract the wrong name from error
        wrong_name = await self._extract_wrong_name(error)

        # STEP 2: If we found a wrong name, search database for correct name
        if wrong_name:
            correct_name = await self._find_correct_name(wrong_name)
            if correct_name:
                # TRUE LEARNING: Store the correction!
                self.knowledge.name_corrections[wrong_name.lower()] = correct_name
                logger.info(f"LEARNED: '{wrong_name}' should be '{correct_name}'")

        # STEP 3: DEEP LEARNING - Find which table has the needed columns
        await self._learn_table_relationships(question, sql, error)

        # STEP 3: LLM analyzes for additional insights
        prompt = f"""Analyze this SQL failure to learn from it.

QUESTION: {question}

FAILED SQL:
{sql}

ERROR: {error[:300]}

CORRECTIONS TRIED: {len(trace.get('corrections', []))}

Analyze:
1. What was the root cause?
2. Why didn't the corrections work?
3. What should be done differently next time?
4. What should the system remember for future similar questions?

ANALYSIS:"""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=300)
            return response.strip()
        except:
            return "Analysis failed"

    async def _learn_table_relationships(self, question: str, failed_sql: str, error: str) -> None:
        """DEEP LEARNING: Figure out which tables should be used for this type of question"""

        # Get schema info for LLM to analyze
        schema_info = []
        for table, info in list(self._schema.items())[:20]:
            cols = [c["name"] for c in info.get("columns", [])[:15]]
            schema_info.append(f"{table}: {', '.join(cols)}")

        prompt = f"""A SQL query failed. Analyze and learn which tables should be used.

QUESTION: {question}

FAILED SQL (used wrong table?):
{failed_sql[:300]}

ERROR: {error[:200]}

AVAILABLE TABLES AND COLUMNS:
{chr(10).join(schema_info)}

Analyze:
1. What concept is the question asking about? (e.g., "categories", "legislations")
2. Which table(s) actually have columns for this concept?
3. What is the correct table to use?

Respond in format:
CONCEPT: <what the question is about>
WRONG_TABLE: <table used in failed SQL>
CORRECT_TABLE: <table that should be used>
JOIN_HINT: <if join needed, how to join>"""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=200)

            # Extract learnings
            concept = None
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('CONCEPT:'):
                    concept = line.split(':', 1)[1].strip().lower()
                elif line.startswith('CORRECT_TABLE:'):
                    correct_table = line.split(':', 1)[1].strip()
                    if concept and correct_table:
                        self.knowledge.table_relationships[concept] = correct_table
                        logger.info(f"LEARNED: For '{concept}' use table '{correct_table}'")
                elif line.startswith('JOIN_HINT:'):
                    join_hint = line.split(':', 1)[1].strip()
                    if join_hint and len(join_hint) > 5:
                        self.knowledge.schema_insights.append(f"JOIN: {join_hint}")

        except Exception as e:
            logger.warning(f"Table relationship learning failed: {e}")

    async def _extract_wrong_name(self, error: str) -> Optional[str]:
        """Extract the wrong table/column name from error message"""
        error_lower = error.lower()

        # LLM extracts the problematic name
        prompt = f"""Extract the invalid/wrong name from this SQL error.

ERROR: {error[:300]}

What table or column name is invalid?
Respond with ONLY the name, nothing else. If no clear name, say NONE."""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=30)
            name = response.strip().strip("'\"")
            if name and name.upper() != "NONE" and len(name) > 1:
                return name
        except:
            pass
        return None

    async def _find_correct_name(self, wrong_name: str) -> Optional[str]:
        """LLM finds the correct table/column name - NO hardcoded rules"""

        # Get all table names from database
        all_tables = []
        try:
            tables = await self._db_executor("""
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE'
            """)
            all_tables = [row.get("TABLE_NAME") or row.get("table_name") for row in tables if row]
        except:
            pass

        # Get all column names from database
        all_columns = []
        try:
            columns = await self._db_executor("""
                SELECT DISTINCT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
            """)
            all_columns = [row.get("COLUMN_NAME") or row.get("column_name") for row in columns if row]
        except:
            pass

        # LLM decides which is the correct name
        prompt = f"""A SQL query failed because '{wrong_name}' doesn't exist in the database.

AVAILABLE TABLES:
{', '.join(all_tables[:50])}

AVAILABLE COLUMNS:
{', '.join(all_columns[:100])}

Which table or column name was the user likely trying to use?
Think about:
- Spelling similarities
- Singular vs plural forms
- Different naming conventions
- What makes semantic sense

Respond with ONLY the correct name from the lists above.
If no match found, say NONE."""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=50)
            correct_name = response.strip().strip("'\"")

            # Trust LLM's decision if it gave a valid name
            if correct_name and correct_name.upper() != "NONE" and len(correct_name) > 1:
                # Find exact match from available names (LLM might have slight case difference)
                all_names = all_tables + all_columns
                for name in all_names:
                    if name.lower() == correct_name.lower():
                        return name
                # If no exact match, still return LLM's answer (it might know better)
                return correct_name
        except:
            pass

        return None

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _extract_sql(self, response: str) -> str:
        """Extract SQL from LLM response"""
        text = response.strip()

        # Try code block first
        if "```sql" in text.lower():
            start = text.lower().find("```sql") + 6
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Try after SQL:
        if "SQL:" in text.upper():
            idx = text.upper().find("SQL:")
            sql = text[idx + 4:].strip()
            # Remove any markdown
            if "```" in sql:
                sql = sql.split("```")[0].strip()
            return sql

        # Last resort: find SELECT statement
        text_upper = text.upper()
        for keyword in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]:
            if keyword in text_upper:
                idx = text_upper.find(keyword)
                sql = text[idx:].strip()
                # Remove any trailing markdown or explanation
                if "```" in sql:
                    sql = sql.split("```")[0].strip()
                return sql

        return text

    # =========================================================================
    # AUTO-LEARN: LLM-driven self-training
    # =========================================================================

    async def auto_learn(self, intensity: str = "medium") -> Dict:
        """
        LLM-driven self-learning process:
        1. LLM explores database and understands domain
        2. LLM generates diverse test questions
        3. LLM runs questions and learns from results
        4. LLM identifies weak areas and trains more

        ALL driven by LLM - NO hardcoded questions or patterns.
        """
        logger.info(f"AUTO-LEARN: Starting self-training (intensity={intensity})")

        question_counts = {"light": 5, "medium": 15, "heavy": 30}
        target_questions = question_counts.get(intensity, 15)

        results = {
            "questions_generated": 0,
            "questions_tested": 0,
            "successes": 0,
            "failures": 0,
            "learnings": [],
        }

        # STEP 1: LLM explores and understands the database
        logger.info("AUTO-LEARN: Step 1 - LLM exploring database...")
        domain_understanding = await self._llm_explore_domain()
        results["domain"] = domain_understanding.get("domain", "unknown")

        # STEP 2: LLM generates diverse test questions
        logger.info("AUTO-LEARN: Step 2 - LLM generating test questions...")
        questions = await self._llm_generate_questions(domain_understanding, target_questions)
        results["questions_generated"] = len(questions)

        # STEP 3: Run questions and learn
        logger.info(f"AUTO-LEARN: Step 3 - Testing {len(questions)} questions...")
        for i, q in enumerate(questions, 1):
            logger.info(f"AUTO-LEARN: Testing [{i}/{len(questions)}] {q[:50]}...")

            result = await self.query(q)
            results["questions_tested"] += 1

            if result["success"]:
                results["successes"] += 1
            else:
                results["failures"] += 1
                results["learnings"].append({
                    "question": q,
                    "error": result.get("error", "")[:100],
                })

        # STEP 4: LLM identifies weak areas and generates targeted questions
        if results["failures"] > 0:
            logger.info("AUTO-LEARN: Step 4 - Targeting weak areas...")
            targeted = await self._llm_target_weak_areas(results["learnings"])

            for q in targeted[:5]:
                result = await self.query(q)
                results["questions_tested"] += 1
                if result["success"]:
                    results["successes"] += 1
                else:
                    results["failures"] += 1

        # Calculate success rate
        if results["questions_tested"] > 0:
            results["success_rate"] = results["successes"] / results["questions_tested"]
        else:
            results["success_rate"] = 0

        self._save_knowledge()
        logger.info(f"AUTO-LEARN: Complete - {results['success_rate']*100:.1f}% success rate")
        return results

    async def _llm_explore_domain(self) -> Dict:
        """LLM explores database to understand what domain it represents"""

        schema_summary = []
        for table, info in list(self._schema.items())[:15]:
            cols = [c["name"] for c in info.get("columns", [])[:10]]
            schema_summary.append(f"{table}: {', '.join(cols)}")

        prompt = f"""Analyze this database to understand its domain and purpose.

TABLES: {list(self._schema.keys())}

SCHEMA:
{chr(10).join(schema_summary)}

Analyze and respond in this EXACT format:
DOMAIN: <one word domain like: legislation, ecommerce, healthcare, etc>
ENTITIES: <main entities comma-separated>
RELATIONSHIPS: <how tables relate>
QUESTION_TYPES: <what users would ask>"""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=400)

            result = {"raw": response}
            for line in response.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    result[key.strip().lower()] = value.strip()

            return result
        except Exception as e:
            logger.warning(f"Domain exploration failed: {e}")
            return {"domain": "unknown"}

    async def _llm_generate_questions(self, domain: Dict, count: int) -> List[str]:
        """LLM generates diverse test questions based on domain understanding"""

        schema_summary = []
        for table, info in list(self._schema.items())[:15]:
            cols = [c["name"] for c in info.get("columns", [])[:10]]
            schema_summary.append(f"{table}: {', '.join(cols)}")

        prompt = f"""Generate {count} diverse natural language questions for this database.

DOMAIN: {domain.get('domain', 'unknown')}
ENTITIES: {domain.get('entities', 'unknown')}

SCHEMA:
{chr(10).join(schema_summary)}

Generate {count} questions covering:
- Simple counts and listings
- Filtering by conditions
- Date-based queries
- Aggregations (totals, averages)
- Comparisons and rankings
- Joins between related tables
- Data quality checks

Write as a business user would ask. One question per line. No numbers or bullets."""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=800)

            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and len(line) > 10 and ('?' in line or len(line) > 20):
                    if line[0].isdigit() and '.' in line[:4]:
                        line = line.split('.', 1)[1].strip()
                    if line.startswith('-'):
                        line = line[1:].strip()
                    questions.append(line)

            return questions[:count]
        except Exception as e:
            logger.warning(f"Question generation failed: {e}")
            return []

    async def _llm_target_weak_areas(self, failures: List[Dict]) -> List[str]:
        """LLM generates targeted questions to improve weak areas"""

        failure_summary = []
        for f in failures[:5]:
            failure_summary.append(f"Q: {f['question'][:50]}... Error: {f['error'][:50]}")

        prompt = f"""These questions failed. Generate simpler versions to learn correct patterns.

FAILURES:
{chr(10).join(failure_summary)}

Generate 5 simpler questions targeting same concepts.
One per line, no numbering."""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=300)

            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and len(line) > 10:
                    if line[0].isdigit() and '.' in line[:4]:
                        line = line.split('.', 1)[1].strip()
                    questions.append(line)

            return questions
        except:
            return []

    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "dialect": self._dialect,
            "tables": len(self._schema),
            "problem_types_learned": len(set(s.problem_type for s in self.knowledge.successful_solutions)),
            "actions_learned": len(self.knowledge.fix_strategies),
            "solutions_stored": len(self.knowledge.successful_solutions),
            "failures_analyzed": len(self.knowledge.failed_attempts),
            "dialect_learnings": len(self.knowledge.dialect_learnings),
            "schema_insights": len(self.knowledge.schema_insights),
        }
