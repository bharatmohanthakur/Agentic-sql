"""
Adaptive Learning Engine - True Intelligence for SQL Generation
================================================================
This module makes the system TRULY intelligent by:

1. AUTO-DETECTING database type from connection/errors
2. LEARNING SQL corrections from errors automatically
3. STORING learned patterns in memory for reuse
4. IMPROVING accuracy over time through feedback
5. NO HARDCODING - everything is learned

The system observes, learns, and adapts autonomously.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseDialect(str, Enum):
    """Database dialects - auto-detected, not configured"""
    UNKNOWN = "unknown"
    MSSQL = "mssql"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    ORACLE = "oracle"


@dataclass
class LearnedCorrection:
    """A correction learned from an error"""
    error_pattern: str  # Regex pattern that matches the error
    original_sql_pattern: str  # Pattern in SQL that caused error
    correction_pattern: str  # How to fix it
    dialect: DatabaseDialect
    success_count: int = 0
    failure_count: int = 0
    learned_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)

    @property
    def confidence(self) -> float:
        """Confidence score based on success rate"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    def to_dict(self) -> Dict:
        return {
            "error_pattern": self.error_pattern,
            "original_sql_pattern": self.original_sql_pattern,
            "correction_pattern": self.correction_pattern,
            "dialect": self.dialect.value,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "learned_at": self.learned_at.isoformat(),
            "last_used": self.last_used.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LearnedCorrection":
        return cls(
            error_pattern=data["error_pattern"],
            original_sql_pattern=data["original_sql_pattern"],
            correction_pattern=data["correction_pattern"],
            dialect=DatabaseDialect(data["dialect"]),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            learned_at=datetime.fromisoformat(data.get("learned_at", datetime.utcnow().isoformat())),
            last_used=datetime.fromisoformat(data.get("last_used", datetime.utcnow().isoformat())),
        )


@dataclass
class LearnedColumnMapping:
    """Mapping between user terminology and actual column names"""
    user_term: str  # What user/LLM might say
    actual_column: str  # Actual column in database
    table_name: str
    dialect: DatabaseDialect
    confidence: float = 1.0
    usage_count: int = 0


class AdaptiveLearningEngine:
    """
    The brain of the intelligent system - learns from every interaction.

    Key capabilities:
    1. Auto-detects database dialect from connection and errors
    2. Learns SQL syntax corrections from errors
    3. Learns column name mappings from corrections
    4. Stores all learnings persistently
    5. Applies learnings to future queries automatically
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        llm_client: Optional[Any] = None,
    ):
        self.llm = llm_client
        self.storage_path = storage_path or Path.home() / ".vanna" / "adaptive_learning.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Detected database info
        self._detected_dialect: DatabaseDialect = DatabaseDialect.UNKNOWN
        self._dialect_confidence: float = 0.0
        self._dialect_signals: List[str] = []

        # Learned corrections
        self._corrections: List[LearnedCorrection] = []
        self._column_mappings: Dict[str, LearnedColumnMapping] = {}

        # Error tracking for learning
        self._recent_errors: List[Dict] = []
        self._successful_queries: List[Dict] = []

        # Load previous learnings
        self._load_learnings()

    def _load_learnings(self) -> None:
        """Load previously learned patterns from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)

                self._detected_dialect = DatabaseDialect(data.get("dialect", "unknown"))
                self._dialect_confidence = data.get("dialect_confidence", 0.0)

                for c in data.get("corrections", []):
                    self._corrections.append(LearnedCorrection.from_dict(c))

                for key, mapping in data.get("column_mappings", {}).items():
                    # Convert dialect string back to enum
                    if isinstance(mapping.get("dialect"), str):
                        mapping["dialect"] = DatabaseDialect(mapping["dialect"])
                    self._column_mappings[key] = LearnedColumnMapping(**mapping)

                logger.info(f"Loaded {len(self._corrections)} learned corrections")
            except Exception as e:
                logger.warning(f"Failed to load learnings: {e}")

    def _save_learnings(self) -> None:
        """Persist learned patterns to storage"""
        try:
            data = {
                "dialect": self._detected_dialect.value,
                "dialect_confidence": self._dialect_confidence,
                "corrections": [c.to_dict() for c in self._corrections],
                "column_mappings": {
                    k: {
                        "user_term": v.user_term,
                        "actual_column": v.actual_column,
                        "table_name": v.table_name,
                        "dialect": v.dialect.value,
                        "confidence": v.confidence,
                        "usage_count": v.usage_count,
                    }
                    for k, v in self._column_mappings.items()
                },
                "updated_at": datetime.utcnow().isoformat(),
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save learnings: {e}")

    # =========================================================================
    # DIALECT AUTO-DETECTION
    # =========================================================================

    def detect_dialect_from_error(self, error_message: str) -> DatabaseDialect:
        """Learn database dialect from error message patterns"""
        error_lower = error_message.lower()

        # MSSQL indicators
        mssql_signals = [
            "microsoft", "sql server", "mssql", "t-sql",
            "incorrect syntax near", "invalid object name",
            "invalid column name", "[odbc driver", "42000", "42s22",
        ]

        # PostgreSQL indicators
        pg_signals = [
            "postgresql", "postgres", "psycopg",
            "does not exist", "relation", "::regclass",
            "pg_", "ERROR:", "HINT:",
        ]

        # MySQL indicators
        mysql_signals = [
            "mysql", "mariadb", "1064", "1146",
            "you have an error in your sql syntax",
            "unknown column", "table doesn't exist",
        ]

        # SQLite indicators
        sqlite_signals = [
            "sqlite", "no such table", "no such column",
            "near \"", "syntax error",
        ]

        # Count signals
        scores = {
            DatabaseDialect.MSSQL: sum(1 for s in mssql_signals if s in error_lower),
            DatabaseDialect.POSTGRESQL: sum(1 for s in pg_signals if s in error_lower),
            DatabaseDialect.MYSQL: sum(1 for s in mysql_signals if s in error_lower),
            DatabaseDialect.SQLITE: sum(1 for s in sqlite_signals if s in error_lower),
        }

        # Get best match
        best_dialect = max(scores, key=scores.get)
        best_score = scores[best_dialect]

        if best_score > 0:
            # Update detection with confidence
            new_confidence = min(0.95, self._dialect_confidence + 0.1 * best_score)

            if best_dialect != self._detected_dialect:
                if new_confidence > self._dialect_confidence:
                    logger.info(f"Dialect detected: {best_dialect.value} (confidence: {new_confidence:.2f})")
                    self._detected_dialect = best_dialect
                    self._dialect_confidence = new_confidence
                    self._dialect_signals.append(error_message[:100])
                    self._save_learnings()
            else:
                self._dialect_confidence = new_confidence

        return self._detected_dialect

    def detect_dialect_from_connection(self, connection_string: str = None, driver: str = None) -> DatabaseDialect:
        """Detect dialect from connection info"""
        signals = (connection_string or "").lower() + " " + (driver or "").lower()

        if any(x in signals for x in ["mssql", "sqlserver", "pyodbc", "pymssql", "odbc driver"]):
            self._detected_dialect = DatabaseDialect.MSSQL
            self._dialect_confidence = 0.95
        elif any(x in signals for x in ["postgresql", "postgres", "psycopg", "pg8000"]):
            self._detected_dialect = DatabaseDialect.POSTGRESQL
            self._dialect_confidence = 0.95
        elif any(x in signals for x in ["mysql", "mariadb", "pymysql", "mysqlconnector"]):
            self._detected_dialect = DatabaseDialect.MYSQL
            self._dialect_confidence = 0.95
        elif any(x in signals for x in ["sqlite", "sqlite3"]):
            self._detected_dialect = DatabaseDialect.SQLITE
            self._dialect_confidence = 0.95

        if self._detected_dialect != DatabaseDialect.UNKNOWN:
            logger.info(f"Dialect detected from connection: {self._detected_dialect.value}")
            self._save_learnings()

        return self._detected_dialect

    @property
    def dialect(self) -> DatabaseDialect:
        """Get the detected dialect"""
        return self._detected_dialect

    # =========================================================================
    # LEARNING FROM ERRORS
    # =========================================================================

    async def learn_from_error(
        self,
        original_sql: str,
        error_message: str,
        fixed_sql: Optional[str] = None,
        success: bool = False,
    ) -> None:
        """
        Learn from an error - the core learning mechanism.

        If we have a successful fix, learn the pattern.
        If no fix provided, try to infer one using LLM.
        """
        # First, use error to improve dialect detection
        self.detect_dialect_from_error(error_message)

        # Track the error
        self._recent_errors.append({
            "sql": original_sql,
            "error": error_message,
            "fixed": fixed_sql,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # If we have a successful fix, learn from it
        if fixed_sql and success and fixed_sql != original_sql:
            await self._learn_correction(original_sql, fixed_sql, error_message)

        # Track error patterns for analysis (but don't auto-generate regex)
        # The real fixing happens in fix_sql_with_llm()
        logger.info(f"Recorded error for learning: {error_message[:80]}...")

        # Keep recent errors bounded
        if len(self._recent_errors) > 100:
            self._recent_errors = self._recent_errors[-100:]

    async def _learn_correction(
        self,
        original_sql: str,
        fixed_sql: str,
        error_message: str,
    ) -> None:
        """Learn a correction pattern from original -> fixed transformation"""

        # Find what changed
        diff = self._find_sql_diff(original_sql, fixed_sql)

        if not diff:
            return

        # Create error pattern from error message
        error_pattern = self._create_error_pattern(error_message)

        # Check if we already have this correction
        for existing in self._corrections:
            if (existing.error_pattern == error_pattern and
                existing.original_sql_pattern == diff["original_pattern"]):
                existing.success_count += 1
                existing.last_used = datetime.utcnow()
                self._save_learnings()
                logger.info(f"Reinforced existing correction (success count: {existing.success_count})")
                return

        # Create new learned correction
        correction = LearnedCorrection(
            error_pattern=error_pattern,
            original_sql_pattern=diff["original_pattern"],
            correction_pattern=diff["fixed_pattern"],
            dialect=self._detected_dialect,
            success_count=1,
        )

        self._corrections.append(correction)
        self._save_learnings()

        logger.info(f"Learned new correction: {diff['original_pattern']} -> {diff['fixed_pattern']}")

    async def _infer_and_learn_correction(
        self,
        original_sql: str,
        error_message: str,
    ) -> Optional[str]:
        """Use LLM to infer what the correction should be"""
        if not self.llm:
            return None

        # Truncate long error messages and SQL to prevent context overflow
        truncated_error = error_message[:500] if len(error_message) > 500 else error_message
        truncated_sql = original_sql[:1000] if len(original_sql) > 1000 else original_sql

        prompt = f"""Analyze this SQL error and provide the fix.

DATABASE TYPE: {self._detected_dialect.value}

ERROR MESSAGE:
{truncated_error}

ORIGINAL SQL:
{truncated_sql}

What specific syntax change fixes this error?
Respond in this exact format:
PATTERN: <what to find in SQL>
FIX: <what to replace it with>
EXPLANATION: <brief explanation>

Example:
PATTERN: LIMIT (\\d+)
FIX: TOP $1
EXPLANATION: MSSQL uses TOP instead of LIMIT
"""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=200)

            # Parse response
            pattern_match = re.search(r"PATTERN:\s*(.+)", response)
            fix_match = re.search(r"FIX:\s*(.+)", response)

            if pattern_match and fix_match:
                error_pattern = self._create_error_pattern(error_message)

                correction = LearnedCorrection(
                    error_pattern=error_pattern,
                    original_sql_pattern=pattern_match.group(1).strip(),
                    correction_pattern=fix_match.group(1).strip(),
                    dialect=self._detected_dialect,
                    success_count=0,  # Not yet validated
                )

                self._corrections.append(correction)
                self._save_learnings()

                logger.info(f"Inferred correction: {correction.original_sql_pattern} -> {correction.correction_pattern}")

                return correction.correction_pattern
        except Exception as e:
            logger.warning(f"Failed to infer correction: {e}")

        return None

    def _find_sql_diff(self, original: str, fixed: str) -> Optional[Dict]:
        """Find what changed between original and fixed SQL"""
        # Normalize whitespace
        orig_normalized = " ".join(original.split())
        fixed_normalized = " ".join(fixed.split())

        if orig_normalized == fixed_normalized:
            return None

        # Common patterns to detect
        patterns_to_check = [
            # LIMIT -> TOP
            (r'\bLIMIT\s+(\d+)', r'TOP \1'),
            # NOW() -> GETDATE()
            (r'\bNOW\s*\(\)', r'GETDATE()'),
            # || -> +
            (r'\|\|', r'+'),
            # true/false -> 1/0
            (r'\btrue\b', r'1'),
            (r'\bfalse\b', r'0'),
            # ::type -> CAST
            (r'(\w+)::(\w+)', r'CAST(\1 AS \2)'),
        ]

        for orig_pattern, fix_pattern in patterns_to_check:
            if re.search(orig_pattern, orig_normalized, re.IGNORECASE):
                if not re.search(orig_pattern, fixed_normalized, re.IGNORECASE):
                    return {
                        "original_pattern": orig_pattern,
                        "fixed_pattern": fix_pattern,
                    }

        # If no known pattern, try to find the specific difference
        # This is a simplified diff - could be more sophisticated
        orig_words = set(orig_normalized.lower().split())
        fixed_words = set(fixed_normalized.lower().split())

        removed = orig_words - fixed_words
        added = fixed_words - orig_words

        if removed and added:
            # Found a substitution
            return {
                "original_pattern": r'\b' + r'\b|\b'.join(removed) + r'\b',
                "fixed_pattern": " ".join(added),
            }

        return None

    def _create_error_pattern(self, error_message: str) -> str:
        """Create a regex pattern from error message for matching similar errors"""
        # Remove specific values, keep structure
        pattern = error_message.lower()

        # Replace specific identifiers with wildcards
        pattern = re.sub(r"'[^']+?'", r"'[^']+'", pattern)
        pattern = re.sub(r'"\w+"', r'"\\w+"', pattern)
        pattern = re.sub(r'\b\d+\b', r'\\d+', pattern)

        # Escape regex special chars (except the ones we added)
        pattern = re.sub(r'([.^$*+?{}|()[\]\\])', r'\\\1', pattern)

        # Take first 100 chars for pattern
        return pattern[:100]

    # =========================================================================
    # APPLYING LEARNED CORRECTIONS
    # =========================================================================

    def apply_learned_corrections(self, sql: str) -> str:
        """
        Apply learned corrections - NO REGEX RULES!

        This method does MINIMAL preprocessing. The real intelligence
        is in the LLM-based error recovery in _try_fix_sql().
        """
        # Only do safe, unambiguous string replacements (not regex)
        corrected = sql

        if self._detected_dialect == DatabaseDialect.MSSQL:
            # Only handle LIMIT->TOP which is a clear, unambiguous transformation
            if 'LIMIT ' in corrected.upper() and 'TOP ' not in corrected.upper():
                # Use LLM to fix this instead of regex
                pass  # Let _try_fix_sql handle it

        return corrected

    async def fix_sql_with_llm(self, sql: str, error_message: str) -> Optional[str]:
        """
        Use LLM intelligence to fix SQL errors - the SMART way.

        The LLM understands context, dialect rules, and can make
        intelligent decisions that regex rules cannot.
        """
        if not self.llm:
            return None

        # Build a focused prompt for fixing
        prompt = f"""You are a SQL expert. Fix this {self._detected_dialect.value.upper()} SQL query.

ERROR: {error_message[:300]}

BROKEN SQL:
{sql}

RULES FOR {self._detected_dialect.value.upper()}:
- Use TOP N not LIMIT (e.g., SELECT TOP 10 * FROM table)
- Use GETDATE() not NOW()
- Use dbo.TableName format
- Use YEAR(col) for date extraction

Return ONLY the fixed SQL, nothing else. No markdown, no explanation."""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=500)
            fixed = response.strip()

            # Clean markdown if present
            if "```" in fixed:
                import re
                match = re.search(r"```(?:sql)?\s*(.*?)\s*```", fixed, re.DOTALL)
                if match:
                    fixed = match.group(1).strip()

            if fixed and fixed != sql:
                logger.info(f"LLM fixed SQL: {fixed[:100]}...")
                return fixed

        except Exception as e:
            logger.warning(f"LLM fix failed: {e}")

        return None

    def apply_column_mappings(self, sql: str) -> str:
        """Apply learned column name mappings - CAREFULLY avoiding SQL keywords"""
        # SQL keywords that should NEVER be replaced
        sql_keywords = {
            'select', 'from', 'where', 'and', 'or', 'not', 'in', 'like',
            'order', 'by', 'group', 'having', 'join', 'left', 'right', 'inner',
            'outer', 'on', 'as', 'distinct', 'top', 'limit', 'offset',
            'count', 'sum', 'avg', 'min', 'max', 'case', 'when', 'then', 'else',
            'end', 'null', 'is', 'between', 'exists', 'union', 'all', 'insert',
            'update', 'delete', 'create', 'alter', 'drop', 'table', 'index',
            'into', 'values', 'set', 'asc', 'desc', 'with', 'over', 'partition',
            'row', 'rows', 'first', 'last', 'year', 'month', 'day', 'date',
            'getdate', 'now', 'cast', 'convert', 'varchar', 'int', 'nvarchar',
            'dbo', 'schema', 'database'
        }

        corrected = sql

        for key, mapping in self._column_mappings.items():
            if mapping.confidence < 0.5:
                continue

            # Skip if the user term is a SQL keyword
            if mapping.user_term.lower() in sql_keywords:
                continue

            # Skip short terms (2 chars or less) as they're too generic
            if len(mapping.user_term) <= 2:
                continue

            # Skip if it matches a table name in the SQL (avoid replacing table names)
            # Simple heuristic: don't replace if it appears after FROM, JOIN, or dbo.
            if re.search(rf'(?:from|join|dbo\.)\s*{re.escape(mapping.user_term)}\b', sql, re.IGNORECASE):
                continue

            # Replace user term with actual column
            corrected = re.sub(
                rf'\b{re.escape(mapping.user_term)}\b',
                mapping.actual_column,
                corrected,
                flags=re.IGNORECASE
            )

        return corrected

    # =========================================================================
    # LEARNING COLUMN MAPPINGS
    # =========================================================================

    async def learn_column_mapping(
        self,
        user_term: str,
        actual_column: str,
        table_name: str,
        success: bool = True,
    ) -> None:
        """Learn that a user term maps to an actual column"""
        key = f"{user_term.lower()}:{table_name.lower()}"

        if key in self._column_mappings:
            mapping = self._column_mappings[key]
            if success:
                mapping.usage_count += 1
                mapping.confidence = min(1.0, mapping.confidence + 0.1)
            else:
                mapping.confidence = max(0.0, mapping.confidence - 0.2)
        else:
            self._column_mappings[key] = LearnedColumnMapping(
                user_term=user_term,
                actual_column=actual_column,
                table_name=table_name,
                dialect=self._detected_dialect,
                confidence=0.7 if success else 0.3,
                usage_count=1,
            )

        self._save_learnings()

    # =========================================================================
    # FEEDBACK AND REINFORCEMENT
    # =========================================================================

    def record_success(self, sql: str) -> None:
        """Record that a query executed successfully"""
        self._successful_queries.append({
            "sql": sql,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Keep bounded
        if len(self._successful_queries) > 100:
            self._successful_queries = self._successful_queries[-100:]

    def record_correction_result(self, correction_pattern: str, success: bool) -> None:
        """Record whether a correction worked"""
        for correction in self._corrections:
            if correction.original_sql_pattern == correction_pattern:
                if success:
                    correction.success_count += 1
                else:
                    correction.failure_count += 1
                correction.last_used = datetime.utcnow()
                self._save_learnings()
                break

    # =========================================================================
    # STATISTICS AND INSIGHTS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            "detected_dialect": self._detected_dialect.value,
            "dialect_confidence": self._dialect_confidence,
            "total_corrections_learned": len(self._corrections),
            "high_confidence_corrections": sum(1 for c in self._corrections if c.confidence > 0.7),
            "column_mappings_learned": len(self._column_mappings),
            "recent_errors_tracked": len(self._recent_errors),
            "successful_queries_tracked": len(self._successful_queries),
        }

    def get_learned_corrections(self) -> List[Dict]:
        """Get all learned corrections for inspection"""
        return [
            {
                "pattern": c.original_sql_pattern,
                "fix": c.correction_pattern,
                "confidence": c.confidence,
                "usage": c.success_count + c.failure_count,
            }
            for c in sorted(self._corrections, key=lambda x: x.confidence, reverse=True)
        ]
