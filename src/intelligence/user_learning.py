"""
User Preference Learning System
Learns individual user patterns, preferences, and behaviors
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PreferenceType(str, Enum):
    """Types of user preferences"""
    TERMINOLOGY = "terminology"  # Words/phrases user uses
    QUERY_STYLE = "query_style"  # How they ask questions
    DATA_FORMAT = "data_format"  # Preferred result format
    VISUALIZATION = "visualization"  # Chart preferences
    TABLES = "tables"  # Frequently accessed tables
    COLUMNS = "columns"  # Frequently used columns
    FILTERS = "filters"  # Common filter conditions
    TIME_RANGE = "time_range"  # Preferred time ranges
    AGGREGATION = "aggregation"  # Preferred aggregations


@dataclass
class UserPreference:
    """Single user preference"""
    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    type: PreferenceType = PreferenceType.TERMINOLOGY
    key: str = ""  # What the preference is about
    value: Any = None  # The preference value
    confidence: float = 1.0
    occurrence_count: int = 1
    last_used: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type.value,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "occurrence_count": self.occurrence_count,
        }


@dataclass
class UserProfile:
    """Complete user profile with all learned preferences"""
    user_id: str
    preferences: Dict[str, UserPreference] = field(default_factory=dict)

    # Behavioral patterns
    active_hours: List[int] = field(default_factory=list)  # Hours of day
    avg_query_complexity: float = 0.5
    preferred_result_limit: int = 100

    # Domain expertise
    familiar_tables: Set[str] = field(default_factory=set)
    familiar_terms: Set[str] = field(default_factory=set)
    expertise_level: str = "intermediate"  # beginner, intermediate, expert

    # Interaction history
    total_queries: int = 0
    successful_queries: int = 0
    corrections_made: int = 0

    # Timestamps
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)

    def get_success_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries


class TerminologyMapper:
    """
    Maps user terminology to database terms
    Learns synonyms and business terms
    """

    def __init__(self):
        # user_term -> database_term
        self._global_mappings: Dict[str, str] = {}
        # user_id -> {user_term -> database_term}
        self._user_mappings: Dict[str, Dict[str, str]] = {}
        # Track confidence
        self._mapping_confidence: Dict[str, float] = {}

    def add_mapping(
        self,
        user_term: str,
        db_term: str,
        user_id: Optional[str] = None,
        confidence: float = 1.0,
    ) -> None:
        """Add a terminology mapping"""
        user_term_lower = user_term.lower()

        if user_id:
            if user_id not in self._user_mappings:
                self._user_mappings[user_id] = {}
            self._user_mappings[user_id][user_term_lower] = db_term
        else:
            self._global_mappings[user_term_lower] = db_term

        key = f"{user_id or 'global'}:{user_term_lower}"
        self._mapping_confidence[key] = confidence

    def translate(
        self,
        text: str,
        user_id: Optional[str] = None,
    ) -> str:
        """Translate user terminology to database terms"""
        result = text

        # Apply user-specific mappings first
        if user_id and user_id in self._user_mappings:
            for user_term, db_term in self._user_mappings[user_id].items():
                result = result.replace(user_term, db_term)

        # Then global mappings
        for user_term, db_term in self._global_mappings.items():
            result = result.replace(user_term, db_term)

        return result

    def get_suggestions(
        self,
        term: str,
        user_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Get possible translations for a term"""
        term_lower = term.lower()
        suggestions = []

        # Check user mappings
        if user_id and user_id in self._user_mappings:
            if term_lower in self._user_mappings[user_id]:
                key = f"{user_id}:{term_lower}"
                conf = self._mapping_confidence.get(key, 1.0)
                suggestions.append((self._user_mappings[user_id][term_lower], conf))

        # Check global mappings
        if term_lower in self._global_mappings:
            key = f"global:{term_lower}"
            conf = self._mapping_confidence.get(key, 0.8)
            suggestions.append((self._global_mappings[term_lower], conf))

        return suggestions


class UserLearningEngine:
    """
    Learns user preferences and patterns from interactions

    Learns:
    - Terminology: What words user uses vs database terms
    - Query patterns: Common query structures
    - Data preferences: Preferred formats, limits, aggregations
    - Time patterns: Preferred date ranges
    - Table/column access: Frequently used data
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        persistence_store: Optional[Any] = None,
    ):
        self.llm = llm_client
        self.store = persistence_store

        self._profiles: Dict[str, UserProfile] = {}
        self._terminology = TerminologyMapper()
        self._query_patterns: Dict[str, List[Dict]] = {}  # user_id -> patterns

    async def learn_from_interaction(
        self,
        user_id: str,
        question: str,
        generated_sql: str,
        executed_sql: str,
        success: bool,
        user_correction: Optional[str] = None,
        result_data: Optional[List[Dict]] = None,
    ) -> None:
        """Learn from a user interaction"""
        profile = self._get_or_create_profile(user_id)

        # Update basic stats
        profile.total_queries += 1
        if success:
            profile.successful_queries += 1
        profile.last_active = datetime.utcnow()

        # Learn terminology
        await self._learn_terminology(user_id, question, executed_sql)

        # Learn from correction
        if user_correction:
            profile.corrections_made += 1
            await self._learn_from_correction(
                user_id, question, generated_sql, user_correction
            )

        # Learn query patterns
        await self._learn_query_pattern(user_id, question, executed_sql)

        # Learn table/column preferences
        self._learn_table_preferences(user_id, executed_sql)

        # Learn time preferences
        self._learn_time_preferences(user_id, question, executed_sql)

        # Update expertise level
        self._update_expertise_level(profile)

        logger.debug(f"Learned from interaction for user {user_id}")

    def _get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
        return self._profiles[user_id]

    async def _learn_terminology(
        self,
        user_id: str,
        question: str,
        sql: str,
    ) -> None:
        """Learn terminology mappings from question to SQL"""
        if not self.llm:
            return

        prompt = f"""
        Analyze this question and SQL to identify terminology mappings.

        Question: {question}
        SQL: {sql}

        Find words/phrases in the question that map to database terms in SQL.
        Return as JSON array of objects: {{"user_term": "...", "db_term": "..."}}

        JSON:
        """

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=300)

            # Parse JSON
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                mappings = json.loads(json_match.group())

                for mapping in mappings:
                    self._terminology.add_mapping(
                        user_term=mapping.get("user_term", ""),
                        db_term=mapping.get("db_term", ""),
                        user_id=user_id,
                        confidence=0.8,
                    )

                    # Also add as preference
                    pref = UserPreference(
                        user_id=user_id,
                        type=PreferenceType.TERMINOLOGY,
                        key=mapping.get("user_term", ""),
                        value=mapping.get("db_term", ""),
                    )
                    profile = self._get_or_create_profile(user_id)
                    profile.preferences[pref.id] = pref
        except Exception as e:
            logger.warning(f"Terminology learning failed: {e}")

    async def _learn_from_correction(
        self,
        user_id: str,
        question: str,
        wrong_sql: str,
        correct_sql: str,
    ) -> None:
        """Learn from user corrections"""
        if not self.llm:
            return

        prompt = f"""
        A user corrected a SQL query. Analyze what was wrong and learn:

        Question: {question}
        Wrong SQL: {wrong_sql}
        Correct SQL: {correct_sql}

        What should we learn from this correction?
        Return as JSON with:
        - terminology: [{{"user_term": "...", "correct_db_term": "..."}}]
        - patterns: ["what patterns to remember"]
        - rules: ["business rules implied"]

        JSON:
        """

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=500)

            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                learnings = json.loads(json_match.group())

                # Apply terminology learnings
                for term_map in learnings.get("terminology", []):
                    self._terminology.add_mapping(
                        user_term=term_map.get("user_term", ""),
                        db_term=term_map.get("correct_db_term", ""),
                        user_id=user_id,
                        confidence=1.0,  # High confidence from correction
                    )

                # Store patterns
                profile = self._get_or_create_profile(user_id)
                for pattern in learnings.get("patterns", []):
                    pref = UserPreference(
                        user_id=user_id,
                        type=PreferenceType.QUERY_STYLE,
                        key="pattern",
                        value=pattern,
                        confidence=1.0,
                    )
                    profile.preferences[pref.id] = pref

        except Exception as e:
            logger.warning(f"Correction learning failed: {e}")

    async def _learn_query_pattern(
        self,
        user_id: str,
        question: str,
        sql: str,
    ) -> None:
        """Learn query patterns user tends to use"""
        pattern = {
            "question_template": self._extract_template(question),
            "sql_template": self._extract_sql_template(sql),
            "timestamp": datetime.utcnow().isoformat(),
        }

        if user_id not in self._query_patterns:
            self._query_patterns[user_id] = []

        self._query_patterns[user_id].append(pattern)

        # Keep only recent patterns
        if len(self._query_patterns[user_id]) > 100:
            self._query_patterns[user_id] = self._query_patterns[user_id][-100:]

    def _extract_template(self, question: str) -> str:
        """Extract a template from question by removing specifics"""
        import re

        # Replace numbers with placeholder
        template = re.sub(r'\d+', '{NUMBER}', question)

        # Replace quoted strings
        template = re.sub(r'"[^"]*"', '{STRING}', template)
        template = re.sub(r"'[^']*'", '{STRING}', template)

        # Replace dates
        template = re.sub(r'\d{4}-\d{2}-\d{2}', '{DATE}', template)

        return template

    def _extract_sql_template(self, sql: str) -> str:
        """Extract SQL template"""
        import re

        # Replace literal values
        template = re.sub(r"'[^']*'", "'{VALUE}'", sql)
        template = re.sub(r'\b\d+\b', '{N}', template)

        return template

    def _learn_table_preferences(self, user_id: str, sql: str) -> None:
        """Learn which tables user frequently accesses"""
        import re

        profile = self._get_or_create_profile(user_id)

        # Extract table names
        tables = re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE)

        for table in tables:
            profile.familiar_tables.add(table.lower())

            # Update preference count
            pref_key = f"table:{table.lower()}"
            existing = None
            for p in profile.preferences.values():
                if p.type == PreferenceType.TABLES and p.key == table.lower():
                    existing = p
                    break

            if existing:
                existing.occurrence_count += 1
                existing.last_used = datetime.utcnow()
            else:
                pref = UserPreference(
                    user_id=user_id,
                    type=PreferenceType.TABLES,
                    key=table.lower(),
                    value=table,
                    occurrence_count=1,
                )
                profile.preferences[pref.id] = pref

    def _learn_time_preferences(
        self,
        user_id: str,
        question: str,
        sql: str,
    ) -> None:
        """Learn time range preferences"""
        import re

        profile = self._get_or_create_profile(user_id)

        # Common time patterns
        time_patterns = {
            "last week": 7,
            "last month": 30,
            "last quarter": 90,
            "last year": 365,
            "this week": 7,
            "this month": 30,
            "ytd": 365,
            "mtd": 30,
        }

        question_lower = question.lower()
        for pattern, days in time_patterns.items():
            if pattern in question_lower:
                pref = UserPreference(
                    user_id=user_id,
                    type=PreferenceType.TIME_RANGE,
                    key=pattern,
                    value=days,
                )
                profile.preferences[pref.id] = pref
                break

    def _update_expertise_level(self, profile: UserProfile) -> None:
        """Update user expertise level based on behavior"""
        # Factors: query complexity, success rate, familiar tables

        complexity_score = profile.avg_query_complexity
        success_rate = profile.get_success_rate()
        familiarity = len(profile.familiar_tables) / 20  # Normalize

        score = (complexity_score + success_rate + min(familiarity, 1.0)) / 3

        if score > 0.8:
            profile.expertise_level = "expert"
        elif score > 0.5:
            profile.expertise_level = "intermediate"
        else:
            profile.expertise_level = "beginner"

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context for query processing"""
        profile = self._get_or_create_profile(user_id)

        # Get frequently used tables
        table_prefs = [
            p for p in profile.preferences.values()
            if p.type == PreferenceType.TABLES
        ]
        table_prefs.sort(key=lambda x: x.occurrence_count, reverse=True)
        frequent_tables = [p.key for p in table_prefs[:10]]

        # Get terminology mappings
        user_terminology = {}
        for p in profile.preferences.values():
            if p.type == PreferenceType.TERMINOLOGY:
                user_terminology[p.key] = p.value

        return {
            "user_id": user_id,
            "expertise_level": profile.expertise_level,
            "frequent_tables": frequent_tables,
            "terminology": user_terminology,
            "success_rate": profile.get_success_rate(),
            "total_queries": profile.total_queries,
        }

    def translate_question(
        self,
        question: str,
        user_id: Optional[str] = None,
    ) -> str:
        """Translate question using learned terminology"""
        return self._terminology.translate(question, user_id)

    def get_similar_patterns(
        self,
        question: str,
        user_id: str,
        limit: int = 5,
    ) -> List[Dict]:
        """Get similar query patterns from user history"""
        if user_id not in self._query_patterns:
            return []

        question_template = self._extract_template(question)

        # Simple matching
        similar = []
        for pattern in self._query_patterns[user_id]:
            if self._template_similarity(question_template, pattern["question_template"]) > 0.5:
                similar.append(pattern)

        return similar[:limit]

    def _template_similarity(self, t1: str, t2: str) -> float:
        """Calculate similarity between templates"""
        words1 = set(t1.lower().split())
        words2 = set(t2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)
