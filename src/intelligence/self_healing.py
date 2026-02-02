"""
Self-Healing Engine - Auto-correction and optimization
Automatically fixes errors, optimizes queries, and adapts to changes
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Types of SQL errors"""
    SYNTAX = "syntax"
    COLUMN_NOT_FOUND = "column_not_found"
    TABLE_NOT_FOUND = "table_not_found"
    AMBIGUOUS_COLUMN = "ambiguous_column"
    TYPE_MISMATCH = "type_mismatch"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    MSSQL_LIMIT = "mssql_limit"  # LIMIT not supported in MSSQL
    MSSQL_SYNTAX = "mssql_syntax"  # Other MSSQL-specific syntax issues
    UNKNOWN = "unknown"


class FixStrategy(str, Enum):
    """Strategies for fixing errors"""
    COLUMN_ALIAS = "column_alias"
    TABLE_ALIAS = "table_alias"
    ADD_QUALIFIER = "add_qualifier"
    FIX_TYPO = "fix_typo"
    CHANGE_TYPE = "change_type"
    ADD_LIMIT = "add_limit"
    ADD_INDEX_HINT = "add_index_hint"
    SIMPLIFY_QUERY = "simplify_query"
    REGENERATE = "regenerate"
    MSSQL_CONVERT = "mssql_convert"  # Convert PostgreSQL/MySQL syntax to MSSQL


@dataclass
class ErrorPattern:
    """Pattern for identifying and fixing errors"""
    error_type: ErrorType
    pattern: str  # Regex pattern
    fix_strategy: FixStrategy
    fix_template: str = ""
    priority: int = 0


@dataclass
class FixAttempt:
    """Record of a fix attempt"""
    original_sql: str
    fixed_sql: str
    error_type: ErrorType
    fix_strategy: FixStrategy
    success: bool = False
    attempt_number: int = 1
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ErrorCorrector:
    """
    Automatic error detection and correction

    Handles:
    - Column not found -> suggests similar columns
    - Table not found -> suggests similar tables
    - Syntax errors -> common fixes
    - Type mismatches -> type casting
    - Ambiguous columns -> adds table qualifiers
    """

    # Common error patterns with fixes
    ERROR_PATTERNS = [
        # MSSQL-specific errors (check first - higher priority)
        ErrorPattern(
            error_type=ErrorType.MSSQL_LIMIT,
            pattern=r"incorrect syntax near 'limit'",
            fix_strategy=FixStrategy.MSSQL_CONVERT,
            priority=10,
        ),
        ErrorPattern(
            error_type=ErrorType.MSSQL_LIMIT,
            pattern=r"incorrect syntax near '\d+'",  # LIMIT N error
            fix_strategy=FixStrategy.MSSQL_CONVERT,
            priority=10,
        ),
        ErrorPattern(
            error_type=ErrorType.MSSQL_SYNTAX,
            pattern=r"incorrect syntax near the keyword",
            fix_strategy=FixStrategy.MSSQL_CONVERT,
            priority=9,
        ),
        ErrorPattern(
            error_type=ErrorType.MSSQL_SYNTAX,
            pattern=r"conversion failed when converting",  # Date conversion errors
            fix_strategy=FixStrategy.MSSQL_CONVERT,
            priority=9,
        ),
        ErrorPattern(
            error_type=ErrorType.COLUMN_NOT_FOUND,
            pattern=r"invalid column name",  # MSSQL
            fix_strategy=FixStrategy.FIX_TYPO,
            priority=8,
        ),
        ErrorPattern(
            error_type=ErrorType.TABLE_NOT_FOUND,
            pattern=r"invalid object name",  # MSSQL
            fix_strategy=FixStrategy.FIX_TYPO,
            priority=8,
        ),
        # PostgreSQL/Generic errors
        ErrorPattern(
            error_type=ErrorType.COLUMN_NOT_FOUND,
            pattern=r"column [\"']?(\w+)[\"']? does not exist",
            fix_strategy=FixStrategy.FIX_TYPO,
        ),
        ErrorPattern(
            error_type=ErrorType.TABLE_NOT_FOUND,
            pattern=r"relation [\"']?(\w+)[\"']? does not exist",
            fix_strategy=FixStrategy.FIX_TYPO,
        ),
        ErrorPattern(
            error_type=ErrorType.AMBIGUOUS_COLUMN,
            pattern=r"column [\"']?(\w+)[\"']? is ambiguous",
            fix_strategy=FixStrategy.ADD_QUALIFIER,
        ),
        ErrorPattern(
            error_type=ErrorType.TYPE_MISMATCH,
            pattern=r"operator does not exist: (\w+) (=|<|>) (\w+)",
            fix_strategy=FixStrategy.CHANGE_TYPE,
        ),
        ErrorPattern(
            error_type=ErrorType.SYNTAX,
            pattern=r"syntax error at or near",
            fix_strategy=FixStrategy.REGENERATE,
        ),
        ErrorPattern(
            error_type=ErrorType.TIMEOUT,
            pattern=r"(timeout|cancelled|too long)",
            fix_strategy=FixStrategy.ADD_LIMIT,
        ),
    ]

    def __init__(
        self,
        schema_profiles: Dict[str, Any],
        llm_client: Optional[Any] = None,
        max_fix_attempts: int = 3,
    ):
        self.schema = schema_profiles
        self.llm = llm_client
        self.max_attempts = max_fix_attempts
        self._fix_history: List[FixAttempt] = []
        self._column_index: Dict[str, List[str]] = {}  # column -> [tables]
        self._build_column_index()

    def _build_column_index(self) -> None:
        """Build index of columns to tables"""
        for table_name, profile in self.schema.items():
            for col in profile.columns if hasattr(profile, 'columns') else []:
                col_name = col.name if hasattr(col, 'name') else col
                if col_name not in self._column_index:
                    self._column_index[col_name] = []
                self._column_index[col_name].append(table_name)

    async def fix_error(
        self,
        sql: str,
        error_message: str,
        context: Optional[Dict] = None,
    ) -> Tuple[str, bool]:
        """
        Attempt to fix SQL error

        Returns: (fixed_sql, success)
        """
        # Detect error type
        error_type = self._detect_error_type(error_message)

        logger.info(f"Attempting to fix {error_type.value} error")

        for attempt in range(self.max_attempts):
            fixed_sql = await self._apply_fix(
                sql,
                error_message,
                error_type,
                attempt,
            )

            if fixed_sql and fixed_sql != sql:
                fix_attempt = FixAttempt(
                    original_sql=sql,
                    fixed_sql=fixed_sql,
                    error_type=error_type,
                    fix_strategy=self._get_strategy(error_type),
                    attempt_number=attempt + 1,
                )
                self._fix_history.append(fix_attempt)

                return fixed_sql, True

            sql = fixed_sql or sql

        return sql, False

    def _detect_error_type(self, error_message: str) -> ErrorType:
        """Detect the type of error from message"""
        error_lower = error_message.lower()

        for pattern in self.ERROR_PATTERNS:
            if re.search(pattern.pattern, error_lower):
                return pattern.error_type

        return ErrorType.UNKNOWN

    def _get_strategy(self, error_type: ErrorType) -> FixStrategy:
        """Get fix strategy for error type"""
        for pattern in self.ERROR_PATTERNS:
            if pattern.error_type == error_type:
                return pattern.fix_strategy
        return FixStrategy.REGENERATE

    async def _apply_fix(
        self,
        sql: str,
        error_message: str,
        error_type: ErrorType,
        attempt: int,
    ) -> Optional[str]:
        """Apply fix based on error type"""
        # MSSQL-specific fixes (try these first)
        if error_type in [ErrorType.MSSQL_LIMIT, ErrorType.MSSQL_SYNTAX]:
            fixed = self._fix_mssql_syntax(sql, error_message)
            if fixed and fixed != sql:
                return fixed
            # If basic fix didn't work, try LLM
            if self.llm:
                return await self._fix_mssql_with_llm(sql, error_message)

        if error_type == ErrorType.COLUMN_NOT_FOUND:
            return await self._fix_column_not_found(sql, error_message)

        elif error_type == ErrorType.TABLE_NOT_FOUND:
            return await self._fix_table_not_found(sql, error_message)

        elif error_type == ErrorType.AMBIGUOUS_COLUMN:
            return self._fix_ambiguous_column(sql, error_message)

        elif error_type == ErrorType.TYPE_MISMATCH:
            return self._fix_type_mismatch(sql, error_message)

        elif error_type == ErrorType.TIMEOUT:
            return self._fix_timeout(sql)

        elif error_type == ErrorType.SYNTAX and self.llm:
            return await self._fix_syntax_with_llm(sql, error_message)

        return None

    def _fix_mssql_syntax(self, sql: str, error_message: str) -> Optional[str]:
        """Fix common MSSQL syntax issues - LIMIT to TOP conversion, etc."""
        fixed = sql
        error_lower = error_message.lower()

        # 1. Convert LIMIT N to TOP N
        limit_match = re.search(r'\bLIMIT\s+(\d+)\s*;?\s*$', fixed, re.IGNORECASE)
        if limit_match:
            limit_value = limit_match.group(1)
            fixed = re.sub(r'\s*\bLIMIT\s+\d+\s*;?\s*$', '', fixed, flags=re.IGNORECASE)
            fixed = re.sub(
                r'\bSELECT\s+(DISTINCT\s+)?',
                lambda m: f"SELECT {m.group(1) or ''}TOP {limit_value} ",
                fixed,
                count=1,
                flags=re.IGNORECASE
            )
            logger.info(f"Fixed LIMIT -> TOP: {limit_value}")

        # 2. Convert LIMIT N OFFSET M to OFFSET M ROWS FETCH NEXT N ROWS ONLY
        offset_limit = re.search(r'\bLIMIT\s+(\d+)\s+OFFSET\s+(\d+)', fixed, re.IGNORECASE)
        if offset_limit:
            limit_val, offset_val = offset_limit.groups()
            fixed = re.sub(
                r'\s*\bLIMIT\s+\d+\s+OFFSET\s+\d+\s*;?\s*$',
                f" OFFSET {offset_val} ROWS FETCH NEXT {limit_val} ROWS ONLY",
                fixed,
                flags=re.IGNORECASE
            )

        # 3. Fix date conversion errors (22007)
        if 'conversion failed' in error_lower and 'date' in error_lower:
            # Try to use TRY_CONVERT for safer date handling
            fixed = re.sub(
                r"CONVERT\s*\(\s*date\s*,\s*([^,)]+)\s*\)",
                r"TRY_CONVERT(date, \1)",
                fixed,
                flags=re.IGNORECASE
            )
            # Fix common date comparison patterns
            fixed = re.sub(
                r"(\w+)\s*>=?\s*'(\d{4})'",
                r"YEAR(\1) >= \2",
                fixed
            )
            # Fix DATEDIFF with string dates
            fixed = re.sub(
                r"DATEDIFF\s*\(\s*(\w+)\s*,\s*([^,]+)\s*,\s*'([^']+)'\s*\)",
                r"DATEDIFF(\1, TRY_CONVERT(date, \2), '\3')",
                fixed,
                flags=re.IGNORECASE
            )

        # 4. Fix :: type casting to CAST()
        cast_pattern = r'(\w+)::(\w+)'
        if re.search(cast_pattern, fixed):
            fixed = re.sub(cast_pattern, r'CAST(\1 AS \2)', fixed)

        # 5. Fix ILIKE to LIKE
        if 'ILIKE' in fixed.upper():
            fixed = re.sub(r'\bILIKE\b', 'LIKE', fixed, flags=re.IGNORECASE)

        # 6. Fix boolean literals
        fixed = re.sub(r'\btrue\b', '1', fixed, flags=re.IGNORECASE)
        fixed = re.sub(r'\bfalse\b', '0', fixed, flags=re.IGNORECASE)

        # 7. Fix NOW() -> GETDATE()
        fixed = re.sub(r'\bNOW\(\)', 'GETDATE()', fixed, flags=re.IGNORECASE)

        # 8. Fix string concatenation
        fixed = re.sub(r'\|\|', '+', fixed)

        # 9. Fix column naming issues detected from error
        if 'invalid column name' in error_lower:
            # Extract the bad column name
            col_match = re.search(r"invalid column name '(\w+)'", error_lower)
            if col_match:
                bad_col = col_match.group(1)
                # Common column name mappings
                mappings = {
                    'created_at': 'Date_of_Issuance',
                    'createdat': 'Date_of_Issuance',
                    'created': 'Date_of_Issuance',
                    'date': 'Date_of_Issuance',
                    'legislation_type': 'Type',
                    'legislationtype': 'Type',
                }
                if bad_col.lower() in mappings:
                    fixed = re.sub(
                        rf'\b{bad_col}\b',
                        mappings[bad_col.lower()],
                        fixed,
                        flags=re.IGNORECASE
                    )

        return fixed if fixed != sql else None

    async def _fix_mssql_with_llm(self, sql: str, error_message: str) -> Optional[str]:
        """Use LLM to fix MSSQL-specific syntax errors"""
        prompt = f"""Fix this SQL query for Microsoft SQL Server (MSSQL/T-SQL).

ERROR: {error_message}

ORIGINAL SQL:
{sql}

MSSQL RULES:
1. Use TOP N instead of LIMIT N (e.g., SELECT TOP 10 * FROM table)
2. Use GETDATE() instead of NOW()
3. Use + for string concatenation instead of ||
4. Use CAST(x AS type) instead of x::type
5. Use 1/0 instead of true/false
6. For pagination use: OFFSET N ROWS FETCH NEXT M ROWS ONLY
7. Use dbo.TableName format
8. Use square brackets for reserved words: [Order], [User]

Return ONLY the corrected SQL, nothing else:"""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=500)
            fixed = response.strip()
            # Extract SQL from markdown if present
            if "```" in fixed:
                match = re.search(r"```(?:sql)?\s*(.*?)\s*```", fixed, re.DOTALL)
                if match:
                    fixed = match.group(1).strip()
            return fixed
        except Exception as e:
            logger.warning(f"LLM fix failed: {e}")
            return None

    async def _fix_column_not_found(
        self,
        sql: str,
        error_message: str,
    ) -> Optional[str]:
        """Fix column not found error"""
        # Extract bad column name
        match = re.search(r"column [\"']?(\w+)[\"']?", error_message.lower())
        if not match:
            return None

        bad_column = match.group(1)

        # Find similar columns
        similar = self._find_similar(bad_column, list(self._column_index.keys()))

        if similar:
            # Replace in SQL
            return re.sub(
                rf'\b{bad_column}\b',
                similar[0],
                sql,
                flags=re.IGNORECASE,
            )

        return None

    async def _fix_table_not_found(
        self,
        sql: str,
        error_message: str,
    ) -> Optional[str]:
        """Fix table not found error"""
        match = re.search(r"relation [\"']?(\w+)[\"']?", error_message.lower())
        if not match:
            return None

        bad_table = match.group(1)
        similar = self._find_similar(bad_table, list(self.schema.keys()))

        if similar:
            return re.sub(
                rf'\b{bad_table}\b',
                similar[0],
                sql,
                flags=re.IGNORECASE,
            )

        return None

    def _fix_ambiguous_column(
        self,
        sql: str,
        error_message: str,
    ) -> Optional[str]:
        """Fix ambiguous column by adding table qualifier"""
        match = re.search(r"column [\"']?(\w+)[\"']?", error_message.lower())
        if not match:
            return None

        column = match.group(1)
        tables = self._column_index.get(column, [])

        if tables:
            # Add first table as qualifier
            return re.sub(
                rf'(?<![.\w]){column}(?![.\w])',
                f'{tables[0]}.{column}',
                sql,
                flags=re.IGNORECASE,
            )

        return None

    def _fix_type_mismatch(
        self,
        sql: str,
        error_message: str,
    ) -> Optional[str]:
        """Fix type mismatch with casting"""
        # Look for string/number comparison issues
        # Add CAST() or ::type as needed

        # Simple fix: quote unquoted strings
        sql = re.sub(
            r"= (\w+)(?!\s*')",
            r"= '\1'",
            sql,
        )

        return sql

    def _fix_timeout(self, sql: str) -> Optional[str]:
        """Fix timeout by adding LIMIT/TOP"""
        has_limit = (
            "LIMIT" in sql.upper() or
            re.search(r'\bTOP\s+\d+', sql, re.IGNORECASE)
        )
        if not has_limit:
            # For now, add LIMIT - the optimizer will convert to TOP for MSSQL
            sql = sql.rstrip().rstrip(";")
            return f"{sql} LIMIT 1000"
        return None

    async def _fix_syntax_with_llm(
        self,
        sql: str,
        error_message: str,
    ) -> Optional[str]:
        """Use LLM to fix syntax error"""
        prompt = f"""
        Fix this SQL query that has a syntax error:

        SQL: {sql}
        Error: {error_message}

        Return only the corrected SQL, nothing else:
        """

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=500)
            # Extract SQL from response
            fixed = response.strip()
            if fixed.startswith("```"):
                fixed = re.search(r"```(?:sql)?\s*(.*?)\s*```", fixed, re.DOTALL)
                fixed = fixed.group(1) if fixed else response.strip()
            return fixed
        except:
            return None

    def _find_similar(
        self,
        target: str,
        candidates: List[str],
        threshold: float = 0.6,
    ) -> List[str]:
        """Find similar strings using Levenshtein-like distance"""
        similar = []
        target_lower = target.lower()

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Exact match
            if target_lower == candidate_lower:
                return [candidate]

            # Substring match
            if target_lower in candidate_lower or candidate_lower in target_lower:
                similar.append((candidate, 0.8))
                continue

            # Calculate similarity
            similarity = self._string_similarity(target_lower, candidate_lower)
            if similarity >= threshold:
                similar.append((candidate, similarity))

        similar.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in similar[:3]]

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity (simplified)"""
        if not s1 or not s2:
            return 0.0

        # Common prefix
        prefix_len = 0
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                prefix_len += 1
            else:
                break

        max_len = max(len(s1), len(s2))
        return prefix_len / max_len


class QueryOptimizer:
    """
    Automatic query optimization

    Optimizes:
    - Removes unnecessary columns in SELECT
    - Adds appropriate indexes hints
    - Optimizes JOINs
    - Adds LIMIT for safety
    - Pushes down predicates
    - Converts syntax for target database (MSSQL, PostgreSQL, etc.)
    """

    def __init__(
        self,
        schema_profiles: Dict[str, Any],
        db_executor: Optional[Callable] = None,
        dialect: str = "mssql",  # Target database dialect
    ):
        self.schema = schema_profiles
        self.db_executor = db_executor
        self.dialect = dialect.lower()

    async def optimize(
        self,
        sql: str,
        context: Optional[Dict] = None,
    ) -> str:
        """Optimize a SQL query"""
        optimized = sql

        # Convert syntax for target database FIRST
        if self.dialect == "mssql":
            optimized = self._convert_to_mssql(optimized)

        # Remove SELECT *
        optimized = self._optimize_select_star(optimized)

        # Optimize JOINs order
        optimized = self._optimize_joins(optimized)

        # Add index hints if beneficial
        if self.db_executor:
            optimized = await self._add_index_hints(optimized)

        return optimized

    def _convert_to_mssql(self, sql: str) -> str:
        """Convert PostgreSQL/MySQL syntax to MSSQL T-SQL"""
        converted = sql

        # 1. LIMIT N -> TOP N (at end of query)
        limit_match = re.search(r'\bLIMIT\s+(\d+)\s*;?\s*$', converted, re.IGNORECASE)
        if limit_match:
            limit_value = limit_match.group(1)
            # Remove LIMIT
            converted = re.sub(r'\s*\bLIMIT\s+\d+\s*;?\s*$', '', converted, flags=re.IGNORECASE)
            # Add TOP after SELECT (handle DISTINCT)
            converted = re.sub(
                r'\bSELECT\s+(DISTINCT\s+)?',
                lambda m: f"SELECT {m.group(1) or ''}TOP {limit_value} ",
                converted,
                count=1,
                flags=re.IGNORECASE
            )

        # 2. LIMIT N OFFSET M -> OFFSET M ROWS FETCH NEXT N ROWS ONLY
        offset_limit = re.search(r'\bLIMIT\s+(\d+)\s+OFFSET\s+(\d+)', converted, re.IGNORECASE)
        if offset_limit:
            limit_val, offset_val = offset_limit.groups()
            converted = re.sub(
                r'\s*\bLIMIT\s+\d+\s+OFFSET\s+\d+\s*;?\s*$',
                f" OFFSET {offset_val} ROWS FETCH NEXT {limit_val} ROWS ONLY",
                converted,
                flags=re.IGNORECASE
            )

        # 3. :: type casting -> CAST()
        converted = re.sub(r'(\w+)::(\w+)', r'CAST(\1 AS \2)', converted)

        # 4. ILIKE -> LIKE
        converted = re.sub(r'\bILIKE\b', 'LIKE', converted, flags=re.IGNORECASE)

        # 5. true/false -> 1/0
        converted = re.sub(r'\btrue\b', '1', converted, flags=re.IGNORECASE)
        converted = re.sub(r'\bfalse\b', '0', converted, flags=re.IGNORECASE)

        # 6. NOW() -> GETDATE()
        converted = re.sub(r'\bNOW\s*\(\)', 'GETDATE()', converted, flags=re.IGNORECASE)

        # 7. || -> + (string concat)
        converted = re.sub(r'\|\|', '+', converted)

        # 8. CURRENT_TIMESTAMP -> GETDATE()
        converted = re.sub(r'\bCURRENT_TIMESTAMP\b', 'GETDATE()', converted, flags=re.IGNORECASE)

        # 9. EXTRACT(YEAR FROM date) -> YEAR(date)
        converted = re.sub(
            r"EXTRACT\s*\(\s*(\w+)\s+FROM\s+(\w+)\s*\)",
            r"\1(\2)",
            converted,
            flags=re.IGNORECASE
        )

        return converted

    def _optimize_select_star(self, sql: str) -> str:
        """Replace SELECT * with specific columns"""
        # Only if we know the schema
        match = re.search(
            r'SELECT\s+\*\s+FROM\s+(\w+)',
            sql,
            re.IGNORECASE,
        )

        if match:
            table_name = match.group(1)
            if table_name in self.schema:
                profile = self.schema[table_name]
                columns = [c.name for c in profile.columns[:10]]  # Limit columns
                columns_str = ", ".join(columns)
                sql = re.sub(
                    r'SELECT\s+\*',
                    f'SELECT {columns_str}',
                    sql,
                    count=1,
                    flags=re.IGNORECASE,
                )

        return sql

    def _add_limit_if_missing(
        self,
        sql: str,
        default_limit: int = 1000,
    ) -> str:
        """Add LIMIT/TOP clause if missing (dialect-aware)"""
        # Check if already has limit
        has_limit = (
            re.search(r'\bLIMIT\b', sql, re.IGNORECASE) or
            re.search(r'\bTOP\s+\d+', sql, re.IGNORECASE) or
            re.search(r'\bFETCH\s+NEXT', sql, re.IGNORECASE)
        )

        if not has_limit:
            if self.dialect == "mssql":
                # Add TOP after SELECT
                sql = re.sub(
                    r'\bSELECT\s+(DISTINCT\s+)?',
                    lambda m: f"SELECT {m.group(1) or ''}TOP {default_limit} ",
                    sql,
                    count=1,
                    flags=re.IGNORECASE
                )
            else:
                sql = sql.rstrip().rstrip(";")
                sql = f"{sql} LIMIT {default_limit}"
        return sql

    def _optimize_joins(self, sql: str) -> str:
        """Optimize JOIN order (smaller tables first)"""
        # This is a placeholder - real implementation would
        # analyze table sizes and reorder
        return sql

    async def _add_index_hints(self, sql: str) -> str:
        """Add index hints based on EXPLAIN analysis"""
        if not self.db_executor:
            return sql

        try:
            explain_result = await self.db_executor(f"EXPLAIN {sql}")
            # Analyze explain and add hints
            # This is database-specific
            return sql
        except:
            return sql


class SelfHealingEngine:
    """
    Master self-healing engine that coordinates all healing activities

    Features:
    - Error detection and classification
    - Automatic error correction
    - Query optimization
    - Schema drift adaptation
    - Performance monitoring and tuning
    - MSSQL/PostgreSQL/MySQL syntax auto-conversion
    """

    def __init__(
        self,
        schema_profiles: Dict[str, Any],
        db_executor: Callable,
        llm_client: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        dialect: str = "mssql",  # Target database dialect
    ):
        self.schema = schema_profiles
        self.db_executor = db_executor
        self.llm = llm_client
        self.memory = memory_manager
        self.dialect = dialect.lower()

        self.error_corrector = ErrorCorrector(schema_profiles, llm_client)
        self.query_optimizer = QueryOptimizer(schema_profiles, db_executor, dialect)

        self._error_counts: Dict[ErrorType, int] = {}
        self._successful_fixes: int = 0
        self._failed_fixes: int = 0

    async def execute_with_healing(
        self,
        sql: str,
        max_retries: int = 3,
    ) -> Tuple[Any, str, bool]:
        """
        Execute SQL with automatic healing on failure

        Returns: (result, final_sql, healed)
        """
        current_sql = sql
        healed = False

        for attempt in range(max_retries + 1):
            try:
                # Optimize before execution
                if attempt == 0:
                    current_sql = await self.query_optimizer.optimize(current_sql)

                result = await self.db_executor(current_sql)

                if healed:
                    self._successful_fixes += 1
                    await self._learn_fix(sql, current_sql)

                return result, current_sql, healed

            except Exception as e:
                error_msg = str(e)
                error_type = self.error_corrector._detect_error_type(error_msg)

                self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

                logger.warning(
                    f"Execution failed (attempt {attempt + 1}): {error_msg[:100]}"
                )

                if attempt < max_retries:
                    fixed_sql, success = await self.error_corrector.fix_error(
                        current_sql,
                        error_msg,
                    )

                    if success and fixed_sql != current_sql:
                        current_sql = fixed_sql
                        healed = True
                    else:
                        self._failed_fixes += 1
                        break
                else:
                    self._failed_fixes += 1

        # All retries failed
        raise Exception(f"Query failed after {max_retries} healing attempts")

    async def _learn_fix(self, original: str, fixed: str) -> None:
        """Store successful fix for future learning"""
        if self.memory:
            from ..memory.manager import MemoryType, MemoryPriority

            await self.memory.ingest(
                content=f"Original: {original}\nFixed: {fixed}",
                memory_type=MemoryType.ERROR_PATTERN,
                priority=MemoryPriority.HIGH,
                metadata={"fix_type": "auto_heal"},
            )

    async def adapt_to_schema_changes(self) -> None:
        """Detect and adapt to schema changes"""
        # Re-discover schema
        from .auto_discovery import SchemaDiscovery

        discovery = SchemaDiscovery(self.db_executor, self.llm)
        new_schema = await discovery.discover()

        # Compare with current schema
        changes = self._detect_schema_changes(new_schema)

        if changes:
            logger.info(f"Detected {len(changes)} schema changes")
            self.schema = new_schema
            self.error_corrector = ErrorCorrector(new_schema, self.llm)
            self.query_optimizer = QueryOptimizer(new_schema, self.db_executor)

    def _detect_schema_changes(
        self,
        new_schema: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Detect changes between old and new schema"""
        changes = []

        old_tables = set(self.schema.keys())
        new_tables = set(new_schema.keys())

        # New tables
        for table in new_tables - old_tables:
            changes.append({"type": "table_added", "table": table})

        # Removed tables
        for table in old_tables - new_tables:
            changes.append({"type": "table_removed", "table": table})

        # Changed tables
        for table in old_tables & new_tables:
            old_cols = {c.name for c in self.schema[table].columns}
            new_cols = {c.name for c in new_schema[table].columns}

            for col in new_cols - old_cols:
                changes.append({
                    "type": "column_added",
                    "table": table,
                    "column": col,
                })

            for col in old_cols - new_cols:
                changes.append({
                    "type": "column_removed",
                    "table": table,
                    "column": col,
                })

        return changes

    def get_health_stats(self) -> Dict[str, Any]:
        """Get healing statistics"""
        total_fixes = self._successful_fixes + self._failed_fixes
        success_rate = (
            self._successful_fixes / total_fixes if total_fixes > 0 else 0
        )

        return {
            "successful_fixes": self._successful_fixes,
            "failed_fixes": self._failed_fixes,
            "success_rate": success_rate,
            "error_distribution": dict(self._error_counts),
        }
