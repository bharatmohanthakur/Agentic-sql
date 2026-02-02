"""
Validator Agent - Validates SQL queries before execution
Implements security checks and query optimization suggestions
"""
from __future__ import annotations

import logging
import re
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


class ValidationResult:
    """Result of SQL validation"""

    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []
        self.security_issues: List[str] = []
        self.estimated_cost: Optional[str] = None

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_suggestion(self, msg: str):
        self.suggestions.append(msg)

    def add_security_issue(self, msg: str):
        self.security_issues.append(msg)
        self.is_valid = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "security_issues": self.security_issues,
            "estimated_cost": self.estimated_cost,
        }


class ValidatorAgent(BaseAgent):
    """
    Validator Agent for SQL query validation

    Checks:
    - SQL syntax
    - Security vulnerabilities (injection, destructive ops)
    - Performance issues
    - Best practices
    - Schema compliance
    """

    # Dangerous patterns
    INJECTION_PATTERNS = [
        r";\s*--",  # Comment after semicolon
        r"'\s*OR\s+'?\d*'?\s*=\s*'?\d*",  # OR 1=1 style
        r"UNION\s+SELECT",  # UNION injection
        r";\s*DROP\s+",  # Semicolon followed by DROP
        r";\s*DELETE\s+",  # Semicolon followed by DELETE
        r"xp_cmdshell",  # SQL Server command execution
        r"LOAD_FILE",  # MySQL file access
        r"INTO\s+OUTFILE",  # MySQL file write
    ]

    DESTRUCTIVE_PATTERNS = [
        (r"\bDROP\s+TABLE\b", "DROP TABLE"),
        (r"\bDROP\s+DATABASE\b", "DROP DATABASE"),
        (r"\bTRUNCATE\b", "TRUNCATE"),
        (r"\bDELETE\s+FROM\b", "DELETE"),
        (r"\bUPDATE\s+\w+\s+SET\b", "UPDATE"),
        (r"\bALTER\s+TABLE\b", "ALTER TABLE"),
        (r"\bGRANT\b", "GRANT"),
        (r"\bREVOKE\b", "REVOKE"),
    ]

    def __init__(
        self,
        config: AgentConfig,
        allowed_operations: Optional[List[str]] = None,
        max_result_rows: int = 10000,
    ):
        super().__init__(config)
        self.allowed_operations = allowed_operations or ["SELECT"]
        self.max_result_rows = max_result_rows

    async def think(self, context: AgentContext, input_data: Any) -> Thought:
        """Analyze SQL for potential issues"""
        sql = str(input_data).strip()

        thought_content = []
        thought_content.append(f"Analyzing SQL query ({len(sql)} chars)")

        # Check query type
        query_type = self._detect_query_type(sql)
        thought_content.append(f"Query type: {query_type}")

        # Preliminary checks
        if query_type not in self.allowed_operations:
            thought_content.append(f"WARNING: {query_type} not in allowed operations")

        if len(sql) > 10000:
            thought_content.append("WARNING: Very long query, may impact performance")

        return Thought(
            type=ThoughtType.PLANNING,
            content="\n".join(thought_content),
            metadata={"query_type": query_type, "query_length": len(sql)},
        )

    async def act(self, context: AgentContext, thought: Thought) -> Action:
        """Perform full validation"""
        sql = context.variables.get("input_data", "")
        if not sql and context.thoughts:
            # Try to get from context
            sql = str(context.thoughts[0].metadata.get("sql", ""))

        action = Action(
            tool_name="validate",
            arguments={"sql": sql},
            thought=thought,
        )

        result = ValidationResult()

        try:
            # Security checks
            self._check_injection(sql, result)
            self._check_destructive(sql, result)

            # Syntax and structure
            self._check_syntax(sql, result)
            self._check_structure(sql, result)

            # Performance
            self._check_performance(sql, result)

            action.result = result.to_dict()

        except Exception as e:
            logger.exception("Validation failed")
            action.error = str(e)
            result.add_error(f"Validation error: {e}")
            action.result = result.to_dict()

        return action

    async def reflect(self, context: AgentContext) -> Thought:
        """Summarize validation findings"""
        if not context.actions:
            return Thought(
                type=ThoughtType.REFLECTION,
                content="No validation performed",
                confidence=0.5,
            )

        last_action = context.actions[-1]
        result = last_action.result or {}

        errors = result.get("errors", [])
        warnings = result.get("warnings", [])
        security = result.get("security_issues", [])

        if security:
            confidence = 0.1
            content = f"SECURITY ISSUES FOUND: {security}"
        elif errors:
            confidence = 0.3
            content = f"Validation failed: {errors}"
        elif warnings:
            confidence = 0.7
            content = f"Validation passed with warnings: {warnings}"
        else:
            confidence = 0.95
            content = "Validation passed successfully"

        return Thought(
            type=ThoughtType.REFLECTION,
            content=content,
            confidence=confidence,
        )

    def _detect_query_type(self, sql: str) -> str:
        """Detect the type of SQL query"""
        sql_upper = sql.strip().upper()

        if sql_upper.startswith("SELECT"):
            return "SELECT"
        elif sql_upper.startswith("INSERT"):
            return "INSERT"
        elif sql_upper.startswith("UPDATE"):
            return "UPDATE"
        elif sql_upper.startswith("DELETE"):
            return "DELETE"
        elif sql_upper.startswith("CREATE"):
            return "CREATE"
        elif sql_upper.startswith("DROP"):
            return "DROP"
        elif sql_upper.startswith("ALTER"):
            return "ALTER"
        elif sql_upper.startswith("WITH"):
            return "CTE/SELECT"
        else:
            return "UNKNOWN"

    def _check_injection(self, sql: str, result: ValidationResult) -> None:
        """Check for SQL injection patterns"""
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                result.add_security_issue(f"Potential SQL injection: pattern '{pattern}' detected")

    def _check_destructive(self, sql: str, result: ValidationResult) -> None:
        """Check for destructive operations"""
        for pattern, name in self.DESTRUCTIVE_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                if name not in self.allowed_operations:
                    result.add_security_issue(f"Destructive operation not allowed: {name}")

    def _check_syntax(self, sql: str, result: ValidationResult) -> None:
        """Basic syntax validation"""
        # Check for balanced parentheses
        if sql.count("(") != sql.count(")"):
            result.add_error("Unbalanced parentheses")

        # Check for balanced quotes
        single_quotes = sql.count("'")
        if single_quotes % 2 != 0:
            result.add_error("Unbalanced single quotes")

        # Check for common typos
        typos = [
            (r"\bFROM\s+WHERE\b", "Missing table name between FROM and WHERE"),
            (r"\bSELECT\s+FROM\b", "Missing columns between SELECT and FROM"),
            (r"\bGROUP\s+(?!BY)\b", "GROUP should be followed by BY"),
            (r"\bORDER\s+(?!BY)\b", "ORDER should be followed by BY"),
        ]
        for pattern, msg in typos:
            if re.search(pattern, sql, re.IGNORECASE):
                result.add_error(msg)

    def _check_structure(self, sql: str, result: ValidationResult) -> None:
        """Check query structure"""
        sql_upper = sql.upper()

        # Check for SELECT *
        if re.search(r"SELECT\s+\*", sql, re.IGNORECASE):
            result.add_warning("SELECT * may return unnecessary columns")

        # Check for missing WHERE on potentially large operations
        if "SELECT" in sql_upper and "WHERE" not in sql_upper:
            if "LIMIT" not in sql_upper:
                result.add_warning("No WHERE or LIMIT clause - may return large result set")

        # Check for Cartesian products
        if "," in sql and "JOIN" not in sql_upper and "WHERE" not in sql_upper:
            result.add_warning("Possible Cartesian product - missing JOIN or WHERE")

    def _check_performance(self, sql: str, result: ValidationResult) -> None:
        """Check for performance issues"""
        sql_upper = sql.upper()

        # Functions on indexed columns
        if re.search(r"WHERE\s+\w+\s*\(\s*\w+\s*\)", sql, re.IGNORECASE):
            result.add_suggestion("Function on column in WHERE may prevent index usage")

        # LIKE with leading wildcard
        if re.search(r"LIKE\s+'%", sql, re.IGNORECASE):
            result.add_suggestion("LIKE with leading % cannot use indexes efficiently")

        # OR conditions
        if sql_upper.count(" OR ") > 3:
            result.add_suggestion("Multiple OR conditions may impact performance - consider UNION or IN")

        # Subqueries in SELECT
        if re.search(r"SELECT\s+.*\(\s*SELECT", sql, re.IGNORECASE):
            result.add_suggestion("Correlated subquery in SELECT may be slow - consider JOIN")

        # DISTINCT with many columns
        if "DISTINCT" in sql_upper and sql_upper.count(",") > 5:
            result.add_suggestion("DISTINCT with many columns may be slow")

        # ORDER BY without LIMIT
        if "ORDER BY" in sql_upper and "LIMIT" not in sql_upper:
            result.add_suggestion("ORDER BY without LIMIT may sort entire result set")
