"""
Business Logic Learning System
Learns and applies business rules, metrics, and domain knowledge
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BusinessRuleType(str, Enum):
    """Types of business rules"""
    METRIC_DEFINITION = "metric_definition"  # How metrics are calculated
    DATA_FILTER = "data_filter"  # Required filters (e.g., exclude test data)
    TERMINOLOGY = "terminology"  # Business term definitions
    RELATIONSHIP = "relationship"  # Entity relationships
    VALIDATION = "validation"  # Data validation rules
    CALCULATION = "calculation"  # Formula definitions
    ACCESS_CONTROL = "access_control"  # Who can see what
    TIME_LOGIC = "time_logic"  # Fiscal year, quarters, etc.


class RuleConfidence(str, Enum):
    """Confidence levels for learned rules"""
    EXPLICIT = "explicit"  # Explicitly defined by user
    INFERRED = "inferred"  # Inferred from patterns
    SUGGESTED = "suggested"  # Suggested, needs confirmation


@dataclass
class BusinessRule:
    """A single business rule"""
    id: str = field(default_factory=lambda: str(uuid4()))
    type: BusinessRuleType = BusinessRuleType.TERMINOLOGY
    name: str = ""
    description: str = ""

    # Rule definition
    condition: str = ""  # When to apply
    action: str = ""  # What to do (SQL snippet, filter, etc.)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    confidence: RuleConfidence = RuleConfidence.SUGGESTED
    source: str = ""  # Where it came from
    verified: bool = False

    # Usage tracking
    use_count: int = 0
    success_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def get_success_rate(self) -> float:
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "condition": self.condition,
            "action": self.action,
            "parameters": self.parameters,
            "confidence": self.confidence.value,
            "verified": self.verified,
            "success_rate": self.get_success_rate(),
        }


@dataclass
class MetricDefinition:
    """Definition of a business metric"""
    name: str
    display_name: str = ""
    description: str = ""

    # Calculation
    sql_template: str = ""  # SQL to calculate
    base_table: str = ""
    aggregation: str = "SUM"  # SUM, AVG, COUNT, etc.

    # Dimensions
    default_dimensions: List[str] = field(default_factory=list)
    allowed_dimensions: List[str] = field(default_factory=list)

    # Filters
    required_filters: List[str] = field(default_factory=list)

    # Time handling
    time_column: str = ""
    default_time_range: str = "last_30_days"

    # Formatting
    format_type: str = "number"  # number, currency, percent
    decimal_places: int = 2

    # Metadata
    category: str = ""
    owner: str = ""
    verified: bool = False


class BusinessLogicLearner:
    """
    Learns business logic from:
    - User corrections
    - Documentation
    - Query patterns
    - Explicit definitions

    Applies learned logic to:
    - Query generation
    - Result interpretation
    - Validation
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        knowledge_base: Optional[Any] = None,
    ):
        self.llm = llm_client
        self.knowledge = knowledge_base

        self._rules: Dict[str, BusinessRule] = {}
        self._metrics: Dict[str, MetricDefinition] = {}
        self._terminology: Dict[str, str] = {}  # term -> definition
        self._synonyms: Dict[str, List[str]] = {}  # term -> synonyms

    async def learn_from_documentation(
        self,
        text: str,
        source: str = "documentation",
    ) -> int:
        """Extract business rules from documentation text"""
        if not self.llm:
            return 0

        prompt = f"""
        Extract business rules, metrics, and terminology from this documentation:

        {text}

        Return as JSON with:
        {{
            "rules": [
                {{
                    "name": "rule name",
                    "description": "what it means",
                    "type": "metric_definition|data_filter|terminology|calculation|time_logic",
                    "sql_snippet": "optional SQL",
                    "condition": "when to apply"
                }}
            ],
            "metrics": [
                {{
                    "name": "metric_name",
                    "display_name": "Display Name",
                    "description": "what it measures",
                    "calculation": "how to calculate (SQL or formula)",
                    "base_table": "main table",
                    "aggregation": "SUM|AVG|COUNT"
                }}
            ],
            "terminology": {{
                "term": "definition"
            }}
        }}

        JSON:
        """

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=2000)

            # Parse JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return 0

            data = json.loads(json_match.group())
            count = 0

            # Process rules
            for rule_data in data.get("rules", []):
                rule = BusinessRule(
                    type=BusinessRuleType(rule_data.get("type", "terminology")),
                    name=rule_data.get("name", ""),
                    description=rule_data.get("description", ""),
                    action=rule_data.get("sql_snippet", ""),
                    condition=rule_data.get("condition", ""),
                    confidence=RuleConfidence.INFERRED,
                    source=source,
                )
                self._rules[rule.id] = rule
                count += 1

            # Process metrics
            for metric_data in data.get("metrics", []):
                metric = MetricDefinition(
                    name=metric_data.get("name", ""),
                    display_name=metric_data.get("display_name", ""),
                    description=metric_data.get("description", ""),
                    sql_template=metric_data.get("calculation", ""),
                    base_table=metric_data.get("base_table", ""),
                    aggregation=metric_data.get("aggregation", "SUM"),
                )
                self._metrics[metric.name.lower()] = metric
                count += 1

            # Process terminology
            for term, definition in data.get("terminology", {}).items():
                self._terminology[term.lower()] = definition
                count += 1

            logger.info(f"Learned {count} items from documentation")
            return count

        except Exception as e:
            logger.warning(f"Documentation learning failed: {e}")
            return 0

    async def learn_from_correction(
        self,
        question: str,
        wrong_sql: str,
        correct_sql: str,
        user_explanation: Optional[str] = None,
    ) -> List[BusinessRule]:
        """Learn business rules from user corrections"""
        if not self.llm:
            return []

        prompt = f"""
        A user corrected a SQL query. Extract the business rules that explain the correction:

        Question: {question}
        Wrong SQL: {wrong_sql}
        Correct SQL: {correct_sql}
        User explanation: {user_explanation or "Not provided"}

        What business rules can we learn?
        Return as JSON array:
        [
            {{
                "name": "rule name",
                "description": "what the rule is",
                "type": "data_filter|calculation|terminology|validation",
                "sql_pattern": "SQL pattern to use",
                "applies_when": "when to apply this rule"
            }}
        ]

        JSON:
        """

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=500)

            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []

            rules_data = json.loads(json_match.group())
            learned_rules = []

            for rule_data in rules_data:
                rule = BusinessRule(
                    type=BusinessRuleType(rule_data.get("type", "data_filter")),
                    name=rule_data.get("name", ""),
                    description=rule_data.get("description", ""),
                    action=rule_data.get("sql_pattern", ""),
                    condition=rule_data.get("applies_when", ""),
                    confidence=RuleConfidence.INFERRED,
                    source="user_correction",
                )
                self._rules[rule.id] = rule
                learned_rules.append(rule)

            logger.info(f"Learned {len(learned_rules)} rules from correction")
            return learned_rules

        except Exception as e:
            logger.warning(f"Correction learning failed: {e}")
            return []

    def add_metric(self, metric: MetricDefinition) -> None:
        """Add or update a metric definition"""
        self._metrics[metric.name.lower()] = metric
        logger.info(f"Added metric: {metric.name}")

    def add_rule(self, rule: BusinessRule) -> None:
        """Add or update a business rule"""
        self._rules[rule.id] = rule
        logger.info(f"Added rule: {rule.name}")

    def add_terminology(
        self,
        term: str,
        definition: str,
        synonyms: Optional[List[str]] = None,
    ) -> None:
        """Add business terminology"""
        self._terminology[term.lower()] = definition

        if synonyms:
            self._synonyms[term.lower()] = [s.lower() for s in synonyms]
            # Also map synonyms to original term
            for syn in synonyms:
                self._terminology[syn.lower()] = f"Same as {term}: {definition}"

    def get_applicable_rules(
        self,
        question: str,
        tables: List[str],
    ) -> List[BusinessRule]:
        """Get rules that apply to a query"""
        applicable = []
        question_lower = question.lower()

        for rule in self._rules.values():
            # Check if condition matches
            if rule.condition:
                condition_lower = rule.condition.lower()

                # Simple keyword matching
                if any(kw in question_lower for kw in condition_lower.split()):
                    applicable.append(rule)
                    continue

            # Check if rule mentions any tables
            rule_text = f"{rule.name} {rule.description} {rule.action}".lower()
            if any(table.lower() in rule_text for table in tables):
                applicable.append(rule)

        return applicable

    def get_metric_definition(self, metric_name: str) -> Optional[MetricDefinition]:
        """Get metric definition by name"""
        name_lower = metric_name.lower()

        # Direct match
        if name_lower in self._metrics:
            return self._metrics[name_lower]

        # Try synonyms
        for key, metric in self._metrics.items():
            if name_lower in metric.display_name.lower():
                return metric

        return None

    def expand_metric_to_sql(
        self,
        metric_name: str,
        dimensions: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        time_range: Optional[str] = None,
    ) -> Optional[str]:
        """Expand a metric reference to SQL"""
        metric = self.get_metric_definition(metric_name)
        if not metric:
            return None

        sql_parts = []

        # Build SELECT clause
        if dimensions:
            dim_str = ", ".join(dimensions)
            sql_parts.append(f"SELECT {dim_str},")
        else:
            sql_parts.append("SELECT")

        # Add aggregation
        if metric.sql_template:
            sql_parts.append(f"  {metric.aggregation}({metric.sql_template}) as {metric.name}")
        else:
            sql_parts.append(f"  {metric.aggregation}(*) as {metric.name}")

        # FROM clause
        sql_parts.append(f"FROM {metric.base_table}")

        # WHERE clause
        where_conditions = list(metric.required_filters)

        if filters:
            for col, val in filters.items():
                where_conditions.append(f"{col} = '{val}'")

        if time_range and metric.time_column:
            where_conditions.append(
                f"{metric.time_column} >= NOW() - INTERVAL '{time_range}'"
            )

        if where_conditions:
            sql_parts.append("WHERE " + " AND ".join(where_conditions))

        # GROUP BY
        if dimensions:
            sql_parts.append(f"GROUP BY {', '.join(dimensions)}")

        return "\n".join(sql_parts)

    def apply_rules_to_sql(
        self,
        sql: str,
        rules: List[BusinessRule],
    ) -> str:
        """Apply business rules to SQL"""
        modified_sql = sql

        for rule in rules:
            if rule.type == BusinessRuleType.DATA_FILTER:
                # Add filter to WHERE clause
                if rule.action:
                    if "WHERE" in modified_sql.upper():
                        modified_sql = modified_sql.replace(
                            "WHERE",
                            f"WHERE {rule.action} AND"
                        )
                    else:
                        # Add WHERE before ORDER BY, GROUP BY, or end
                        modified_sql = re.sub(
                            r'(ORDER BY|GROUP BY|LIMIT|$)',
                            f' WHERE {rule.action} \\1',
                            modified_sql,
                            count=1,
                            flags=re.IGNORECASE,
                        )

            elif rule.type == BusinessRuleType.CALCULATION:
                # Apply calculation transformations
                if rule.condition and rule.action:
                    modified_sql = modified_sql.replace(rule.condition, rule.action)

        return modified_sql

    def get_context_for_query(self, question: str) -> str:
        """Get business context for a question"""
        context_parts = []
        question_lower = question.lower()

        # Find relevant terminology
        for term, definition in self._terminology.items():
            if term in question_lower:
                context_parts.append(f"'{term}' means: {definition}")

        # Find relevant metrics
        for name, metric in self._metrics.items():
            if name in question_lower or metric.display_name.lower() in question_lower:
                context_parts.append(
                    f"Metric '{metric.display_name}': {metric.description}"
                )
                if metric.sql_template:
                    context_parts.append(f"  Calculation: {metric.sql_template}")

        # Find potentially applicable rules
        for rule in self._rules.values():
            if rule.name.lower() in question_lower:
                context_parts.append(
                    f"Rule '{rule.name}': {rule.description}"
                )

        return "\n".join(context_parts)

    def validate_result(
        self,
        question: str,
        sql: str,
        result: List[Dict],
    ) -> Tuple[bool, List[str]]:
        """Validate query result against business rules"""
        issues = []

        # Get applicable validation rules
        validation_rules = [
            r for r in self._rules.values()
            if r.type == BusinessRuleType.VALIDATION
        ]

        for rule in validation_rules:
            # Check rule conditions
            if rule.condition and rule.condition.lower() in question.lower():
                # Apply validation
                if rule.parameters:
                    min_val = rule.parameters.get("min_value")
                    max_val = rule.parameters.get("max_value")

                    for row in result:
                        for col, val in row.items():
                            if isinstance(val, (int, float)):
                                if min_val and val < min_val:
                                    issues.append(
                                        f"Value {val} in {col} below minimum {min_val}"
                                    )
                                if max_val and val > max_val:
                                    issues.append(
                                        f"Value {val} in {col} above maximum {max_val}"
                                    )

        return len(issues) == 0, issues

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            "total_rules": len(self._rules),
            "verified_rules": len([r for r in self._rules.values() if r.verified]),
            "total_metrics": len(self._metrics),
            "terminology_count": len(self._terminology),
            "rules_by_type": {
                t.value: len([r for r in self._rules.values() if r.type == t])
                for t in BusinessRuleType
            },
        }


class BusinessResearcher:
    """
    Researches and discovers business logic from:
    - Existing queries
    - Data patterns
    - Documentation
    - User behavior
    """

    def __init__(
        self,
        llm_client: Any,
        db_executor: Callable,
        business_learner: BusinessLogicLearner,
    ):
        self.llm = llm_client
        self.db_executor = db_executor
        self.learner = business_learner

    async def research_table_business_meaning(
        self,
        table_name: str,
        sample_limit: int = 100,
    ) -> Dict[str, Any]:
        """Research what a table represents in business terms"""
        # Get sample data
        try:
            sample = await self.db_executor(
                f"SELECT * FROM {table_name} LIMIT {sample_limit}"
            )
        except:
            sample = []

        # Get column info
        try:
            columns = await self.db_executor(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
            """)
        except:
            columns = []

        if not self.llm:
            return {"table": table_name, "meaning": "Unknown"}

        prompt = f"""
        Analyze this database table and explain its business purpose:

        Table: {table_name}
        Columns: {json.dumps(columns[:20])}
        Sample data: {json.dumps(sample[:5])}

        What does this table represent in business terms?
        What are the key business concepts?
        What metrics could be derived from this table?

        Return as JSON:
        {{
            "business_name": "human readable name",
            "purpose": "what the table is for",
            "key_concepts": ["concept1", "concept2"],
            "possible_metrics": ["metric1", "metric2"],
            "relationships": ["related to X", "feeds into Y"]
        }}

        JSON:
        """

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=500)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {"table": table_name, "meaning": "Unknown"}

    async def discover_implicit_metrics(
        self,
        table_name: str,
    ) -> List[MetricDefinition]:
        """Discover common metrics from table structure"""
        metrics = []

        # Get numeric columns
        try:
            columns = await self.db_executor(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                AND data_type IN ('integer', 'numeric', 'decimal', 'float', 'double precision')
            """)
        except:
            columns = []

        # Common metric patterns
        metric_patterns = {
            "amount": ("Total Amount", "SUM"),
            "quantity": ("Total Quantity", "SUM"),
            "count": ("Count", "SUM"),
            "price": ("Total Price", "SUM"),
            "revenue": ("Total Revenue", "SUM"),
            "cost": ("Total Cost", "SUM"),
            "rate": ("Average Rate", "AVG"),
            "score": ("Average Score", "AVG"),
        }

        for col in columns:
            col_name = col.get("column_name", "")
            col_lower = col_name.lower()

            for pattern, (display, agg) in metric_patterns.items():
                if pattern in col_lower:
                    metric = MetricDefinition(
                        name=f"{table_name}_{col_name}",
                        display_name=f"{display} ({col_name})",
                        description=f"{agg} of {col_name} from {table_name}",
                        sql_template=col_name,
                        base_table=table_name,
                        aggregation=agg,
                    )
                    metrics.append(metric)
                    break

        return metrics

    async def research_data_patterns(
        self,
        table_name: str,
    ) -> Dict[str, Any]:
        """Research data patterns that might indicate business rules"""
        patterns = {}

        # Check for status/type columns
        try:
            columns = await self.db_executor(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                AND (column_name LIKE '%status%' OR column_name LIKE '%type%' OR column_name LIKE '%state%')
            """)

            for col in columns:
                col_name = col.get("column_name")
                values = await self.db_executor(f"""
                    SELECT DISTINCT "{col_name}", COUNT(*) as cnt
                    FROM {table_name}
                    GROUP BY "{col_name}"
                    ORDER BY cnt DESC
                    LIMIT 20
                """)
                patterns[col_name] = [v.get(col_name) for v in values]
        except:
            pass

        return patterns
