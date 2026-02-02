"""
Intelligent Core - The Central Brain of the Text-to-SQL System
================================================================
This is the SINGLE source of truth that:

1. AUTO-DISCOVERS database schema on connect
2. BUILDS knowledge graph automatically
3. LEARNS from every interaction
4. ADAPTS to errors and corrects them
5. IMPROVES accuracy over time

Everything flows through here - no manual configuration needed.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .adaptive_learning import AdaptiveLearningEngine, DatabaseDialect, LearnedColumnMapping
from .auto_discovery import SchemaDiscovery, DatabaseDialect as DiscoveryDialect, TableProfile
from .knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of an intelligent query"""
    success: bool
    sql: str
    data: Optional[List[Dict]] = None
    row_count: int = 0
    error: Optional[str] = None
    was_corrected: bool = False
    corrections_applied: List[str] = field(default_factory=list)
    confidence: float = 0.0
    execution_time_ms: int = 0


class IntelligentCore:
    """
    The central intelligence that coordinates all learning and adaptation.

    This is the ONLY class you need to interact with.
    Everything else is automatic.
    """

    def __init__(
        self,
        llm_client: Any,
        storage_path: Optional[Path] = None,
    ):
        self.llm = llm_client
        self.storage_path = storage_path or Path.home() / ".vanna"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Core components - initialized lazily
        self._db_executor: Optional[Callable] = None
        self._schema_profiles: Dict[str, TableProfile] = {}
        self._schema_context: str = ""

        # Intelligence components
        self.adaptive_learning = AdaptiveLearningEngine(
            storage_path=self.storage_path / "adaptive_learning.json",
            llm_client=llm_client,
        )

        self.knowledge_base = KnowledgeBase(
            embedding_fn=lambda t: llm_client.generate_embedding(t),
            llm_client=llm_client,
        )

        # State
        self._is_connected = False
        self._connection_info: Dict[str, Any] = {}

        logger.info("IntelligentCore initialized")

    # =========================================================================
    # CONNECTION & AUTO-DISCOVERY
    # =========================================================================

    async def connect(
        self,
        db_executor: Callable,
        connection_string: Optional[str] = None,
        driver: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Connect to database and AUTO-DISCOVER everything.

        This single call:
        1. Detects database type automatically
        2. Discovers all tables and columns
        3. Builds knowledge graph
        4. Loads previous learnings

        Args:
            db_executor: Async function to execute SQL
            connection_string: Optional connection string (helps detect dialect)
            driver: Optional driver name (helps detect dialect)

        Returns:
            Discovery statistics
        """
        logger.info("Connecting and discovering...")
        self._db_executor = db_executor

        # Step 1: Detect dialect from connection info
        if connection_string or driver:
            self.adaptive_learning.detect_dialect_from_connection(connection_string, driver)

        # Step 2: Try a simple query to detect dialect from any error
        dialect = await self._detect_dialect_from_probe()

        # Step 3: Auto-discover schema
        discovery_dialect = self._map_to_discovery_dialect(dialect)
        discovery = SchemaDiscovery(
            db_executor=db_executor,
            llm_client=self.llm,
            dialect=discovery_dialect,
        )

        self._schema_profiles = await discovery.discover(
            include_stats=True,
            include_samples=False,
            max_tables=50,
        )

        # Step 4: Build knowledge from discovered schema
        await self._build_knowledge_from_schema()

        # Step 5: Build schema context for LLM
        self._schema_context = self._build_schema_context()

        self._is_connected = True
        self._connection_info = {
            "dialect": dialect.value,
            "tables_discovered": len(self._schema_profiles),
            "columns_discovered": sum(len(p.columns) for p in self._schema_profiles.values()),
        }

        logger.info(f"Connected! Discovered {len(self._schema_profiles)} tables")
        return self._connection_info

    async def _detect_dialect_from_probe(self) -> DatabaseDialect:
        """Detect database dialect by probing with test queries"""
        probe_queries = [
            # MSSQL-specific
            ("SELECT TOP 1 1", DatabaseDialect.MSSQL),
            # PostgreSQL-specific
            ("SELECT 1 LIMIT 1", DatabaseDialect.POSTGRESQL),
            # MySQL-specific
            ("SELECT 1 LIMIT 1", DatabaseDialect.MYSQL),
        ]

        # If we already have high confidence, skip probing
        if self.adaptive_learning._dialect_confidence > 0.8:
            return self.adaptive_learning.dialect

        # Try MSSQL first (most common in enterprise)
        try:
            await self._db_executor("SELECT TOP 1 1 AS test")
            self.adaptive_learning._detected_dialect = DatabaseDialect.MSSQL
            self.adaptive_learning._dialect_confidence = 0.95
            logger.info("Detected MSSQL from probe query")
            return DatabaseDialect.MSSQL
        except Exception as e:
            # Learn from error
            self.adaptive_learning.detect_dialect_from_error(str(e))

        # Try PostgreSQL/MySQL style
        try:
            await self._db_executor("SELECT 1 AS test LIMIT 1")
            if "postgres" in str(type(self._db_executor)).lower():
                self.adaptive_learning._detected_dialect = DatabaseDialect.POSTGRESQL
            else:
                self.adaptive_learning._detected_dialect = DatabaseDialect.MYSQL
            self.adaptive_learning._dialect_confidence = 0.9
            return self.adaptive_learning.dialect
        except Exception as e:
            self.adaptive_learning.detect_dialect_from_error(str(e))

        return self.adaptive_learning.dialect

    def _map_to_discovery_dialect(self, dialect: DatabaseDialect) -> DiscoveryDialect:
        """Map adaptive learning dialect to discovery dialect"""
        mapping = {
            DatabaseDialect.MSSQL: DiscoveryDialect.MSSQL,
            DatabaseDialect.POSTGRESQL: DiscoveryDialect.POSTGRESQL,
            DatabaseDialect.MYSQL: DiscoveryDialect.MYSQL,
            DatabaseDialect.SQLITE: DiscoveryDialect.SQLITE,
        }
        return mapping.get(dialect, DiscoveryDialect.POSTGRESQL)

    async def _build_knowledge_from_schema(self) -> None:
        """Build knowledge base from discovered schema - LIGHTWEIGHT version"""
        logger.info("Building knowledge from discovered schema...")

        # Skip heavy embedding generation for now
        # Instead, just learn column name variations locally (no API calls)
        for table_name, profile in self._schema_profiles.items():
            for col in profile.columns:
                # Learn column name variations locally
                variations = self._generate_column_variations(col.name)
                for variation in variations:
                    # Use sync version to avoid slow API calls during startup
                    self.adaptive_learning._column_mappings[f"{variation}:{table_name}"] = LearnedColumnMapping(
                        user_term=variation,
                        actual_column=col.name,
                        table_name=table_name,
                        dialect=self.adaptive_learning.dialect,
                        confidence=0.8,
                        usage_count=0,
                    )

        logger.info(f"Built knowledge with {len(self.adaptive_learning._column_mappings)} column mappings")

    def _generate_column_variations(self, column_name: str) -> List[str]:
        """Generate common variations of a column name - avoiding SQL keywords"""
        # SQL keywords to avoid generating as variations
        sql_keywords = {
            'select', 'from', 'where', 'and', 'or', 'not', 'in', 'like',
            'order', 'by', 'group', 'having', 'join', 'left', 'right', 'inner',
            'count', 'sum', 'avg', 'min', 'max', 'case', 'when', 'then', 'else',
            'end', 'null', 'is', 'between', 'exists', 'union', 'all',
            'top', 'limit', 'offset', 'asc', 'desc', 'with', 'over',
            'year', 'month', 'day', 'date', 'name', 'type', 'status', 'id',
            'category', 'article', 'text', 'body', 'title', 'number', 'value',
        }

        variations = []

        # Remove underscores and camelCase
        words = column_name.replace("_", " ").split()

        # Individual words - but skip SQL keywords
        for word in words:
            word_lower = word.lower()
            if len(word) > 3 and word_lower not in sql_keywords:
                variations.append(word_lower)

        return variations[:3]  # Limit variations

    def _build_schema_context(self) -> str:
        """Build CONCISE schema context for LLM with working examples"""
        dialect = self.adaptive_learning.dialect

        parts = [
            "=== MSSQL DATABASE SCHEMA ===",
            "",
            "SYNTAX RULES (MUST FOLLOW):",
            "1. Use dbo.TableName format: FROM dbo.Legislations",
            "2. Use TOP N not LIMIT: SELECT TOP 10 * FROM dbo.Table",
            "3. Use GETDATE() not NOW()",
            "4. Use YEAR(col) for date extraction",
            "",
            "WORKING EXAMPLE QUERIES:",
            "-- Count records: SELECT COUNT(*) FROM dbo.Category",
            "-- Get top 10: SELECT TOP 10 * FROM dbo.Legislations ORDER BY Created_at DESC",
            "-- Filter: SELECT * FROM dbo.Legislations WHERE Status = 'In Force'",
            "-- Join: SELECT l.*, c.Name FROM dbo.Legislations l JOIN dbo.Category c ON l.Category_Id = c.Category_Id",
            "",
            "TABLES AND COLUMNS (use EXACT column names):",
        ]

        # Build table list with explicit column names
        table_count = 0
        for table_name, profile in self._schema_profiles.items():
            if table_count >= 20:  # Limit to 20 tables
                break

            cols = [col.name for col in profile.columns[:12]]
            parts.append(f"dbo.{table_name}: {', '.join(cols)}")
            table_count += 1

        if len(self._schema_profiles) > 20:
            parts.append(f"... and {len(self._schema_profiles) - 20} more tables")

        return "\n".join(parts)

    # =========================================================================
    # INTELLIGENT QUERY EXECUTION
    # =========================================================================

    async def query(self, question: str, max_retries: int = 3) -> QueryResult:
        """
        Process a natural language question intelligently.

        This:
        1. Searches knowledge base for relevant context
        2. Generates SQL using LLM with discovered schema
        3. Applies learned corrections BEFORE execution
        4. Executes with automatic error recovery
        5. Learns from any errors for future improvement
        """
        if not self._is_connected:
            return QueryResult(success=False, sql="", error="Not connected. Call connect() first.")

        start_time = datetime.utcnow()

        try:
            # Step 1: Search knowledge base
            relevant_knowledge = await self.knowledge_base.search(question, limit=5)

            # Step 2: Generate SQL
            sql = await self._generate_sql(question, relevant_knowledge)

            # Step 3: Apply learned corrections BEFORE execution
            original_sql = sql
            sql = self.adaptive_learning.apply_learned_corrections(sql)
            sql = self.adaptive_learning.apply_column_mappings(sql)

            corrections_applied = []
            if sql != original_sql:
                corrections_applied.append("auto_corrections")
                logger.info(f"Applied learned corrections to SQL")

            # Step 4: Execute with retry and learning
            data = None
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    data = await self._db_executor(sql)

                    # Success! Record it
                    self.adaptive_learning.record_success(sql)

                    execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

                    return QueryResult(
                        success=True,
                        sql=sql,
                        data=data,
                        row_count=len(data) if data else 0,
                        was_corrected=sql != original_sql,
                        corrections_applied=corrections_applied,
                        confidence=1.0,
                        execution_time_ms=execution_time,
                    )

                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Query failed (attempt {attempt + 1}): {last_error[:100]}")

                    # LEARN from error
                    await self.adaptive_learning.learn_from_error(
                        original_sql=sql,
                        error_message=last_error,
                    )

                    # Try to fix
                    if attempt < max_retries:
                        fixed_sql = await self._try_fix_sql(sql, last_error)
                        if fixed_sql and fixed_sql != sql:
                            sql = fixed_sql
                            corrections_applied.append(f"fix_attempt_{attempt + 1}")

            # All retries failed
            return QueryResult(
                success=False,
                sql=sql,
                error=last_error,
                was_corrected=sql != original_sql,
                corrections_applied=corrections_applied,
            )

        except Exception as e:
            return QueryResult(success=False, sql="", error=str(e))

    async def _generate_sql(
        self,
        question: str,
        relevant_knowledge: List[Any],
    ) -> str:
        """Generate SQL using LLM with full context"""
        # Build knowledge context
        knowledge_context = ""
        if relevant_knowledge:
            knowledge_context = "\n\nRELEVANT KNOWLEDGE:\n"
            for item in relevant_knowledge[:3]:
                knowledge_context += f"- {item.content[:200]}\n"

        prompt = f"""{self._schema_context}

QUESTION: {question}

Write a valid MSSQL query to answer this question.
IMPORTANT:
- Use ONLY tables and columns from the schema above
- Use dbo.TableName format (e.g., dbo.Legislations)
- Use TOP N not LIMIT
- Return ONLY the raw SQL query, nothing else
- NO markdown, NO explanations, NO comments

SQL:"""

        response = await self.llm.generate(prompt=prompt, max_tokens=500)

        # Clean up response
        sql = response.strip()

        # Remove markdown code blocks if present
        if "```" in sql:
            import re
            match = re.search(r"```(?:sql)?\s*(.*?)\s*```", sql, re.DOTALL)
            if match:
                sql = match.group(1).strip()

        # Remove comments at start
        lines = sql.split("\n")
        sql_lines = [l for l in lines if not l.strip().startswith("--")]
        sql = "\n".join(sql_lines).strip()

        return sql

    async def _try_fix_sql(self, sql: str, error_message: str) -> Optional[str]:
        """
        Try to fix SQL using LLM intelligence.

        The LLM understands context and can make smart decisions
        that rule-based systems cannot.
        """
        # Use LLM-based intelligent fixing
        fixed_sql = await self.adaptive_learning.fix_sql_with_llm(sql, error_message)

        if fixed_sql and fixed_sql != sql:
            # Track this fix for potential future learning
            await self.adaptive_learning.learn_from_error(
                original_sql=sql,
                error_message=error_message,
                fixed_sql=fixed_sql,
                success=False,  # Will be updated if execution succeeds
            )
            return fixed_sql

        return None

    # =========================================================================
    # STATISTICS AND INSIGHTS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "connected": self._is_connected,
            "connection_info": self._connection_info,
            "learning_stats": self.adaptive_learning.get_stats(),
            "knowledge_stats": self.knowledge_base.get_stats() if hasattr(self.knowledge_base, 'get_stats') else {},
            "schema": {
                "tables": len(self._schema_profiles),
                "columns": sum(len(p.columns) for p in self._schema_profiles.values()),
            },
        }

    def get_learned_corrections(self) -> List[Dict]:
        """Get all learned SQL corrections"""
        return self.adaptive_learning.get_learned_corrections()
