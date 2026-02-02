"""
Intelligent System - End-to-end intelligent text-to-SQL system
Combines all intelligent components for best-in-class performance
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from .auto_discovery import SchemaDiscovery, RelationshipInference, DataProfiler
from .auto_trainer import AutoTrainer, TrainingPipeline, LearningFeedback, FeedbackType
from .self_healing import SelfHealingEngine, ErrorCorrector, QueryOptimizer
from .deep_reasoner import DeepReasoner, ReasoningStrategy
from .knowledge_base import KnowledgeBase, KnowledgeType
from .user_learning import UserLearningEngine
from .business_learning import BusinessLogicLearner, BusinessResearcher
from .research_agent import ResearchAgent, SelfImplementer

logger = logging.getLogger(__name__)


class SystemState(str, Enum):
    """System states"""
    INITIALIZING = "initializing"
    DISCOVERING = "discovering"
    READY = "ready"
    PROCESSING = "processing"
    LEARNING = "learning"
    ERROR = "error"


class IntelligenceLevel(str, Enum):
    """Intelligence levels for different query complexities"""
    SIMPLE = "simple"  # Direct pattern match
    MODERATE = "moderate"  # Chain-of-thought
    COMPLEX = "complex"  # Tree-of-thought with verification
    EXPERT = "expert"  # Multi-agent with deep reasoning


@dataclass
class QueryResult:
    """Complete result from intelligent query processing"""
    success: bool
    question: str
    sql: Optional[str] = None
    data: Optional[List[Dict]] = None
    columns: Optional[List[str]] = None
    row_count: int = 0

    # Intelligence metadata
    intelligence_level: IntelligenceLevel = IntelligenceLevel.SIMPLE
    reasoning_steps: int = 0
    confidence: float = 0.0
    was_healed: bool = False
    used_patterns: int = 0

    # Explanation
    explanation: Optional[str] = None
    reasoning_chain: Optional[str] = None

    # Performance
    processing_time_ms: float = 0.0
    tokens_used: int = 0

    # Error info
    error: Optional[str] = None
    error_corrected: bool = False


class IntelligentSystemConfig(BaseModel):
    """Configuration for intelligent system"""
    # Auto-discovery
    auto_discover_on_connect: bool = True
    schema_refresh_interval_hours: int = 24

    # Learning
    enable_auto_learning: bool = True
    learning_batch_size: int = 10
    min_confidence_for_pattern: float = 0.8

    # User preference learning
    enable_user_learning: bool = True
    track_user_terminology: bool = True
    track_user_patterns: bool = True

    # Business logic learning
    enable_business_learning: bool = True
    auto_discover_metrics: bool = True
    learn_from_documentation: bool = True

    # Research agent
    enable_research_agent: bool = True
    research_interval_minutes: int = 60
    auto_implement_findings: bool = True

    # Reasoning
    default_reasoning_strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    max_reasoning_depth: int = 10
    enable_deep_reasoning: bool = True

    # Self-healing
    enable_self_healing: bool = True
    max_healing_attempts: int = 3

    # Performance
    query_timeout_seconds: float = 60.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


class IntelligentSystem:
    """
    End-to-End Intelligent Text-to-SQL System

    Features:
    1. AUTO-DISCOVERY: Automatically discovers and understands database schema
    2. AUTO-TRAINING: Learns from successful queries and user feedback
    3. SELF-HEALING: Automatically corrects errors and adapts to changes
    4. DEEP REASONING: Uses advanced reasoning for complex queries
    5. KNOWLEDGE INTEGRATION: Combines all learned knowledge
    6. REAL-TIME SCALING: Handles 100+ tables across multiple databases

    Usage:
        system = IntelligentSystem(config, llm_client)
        await system.connect(connection_configs)
        result = await system.query("What are the top 10 customers by revenue?")
    """

    def __init__(
        self,
        config: IntelligentSystemConfig,
        llm_client: Any,
        embedding_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.llm = llm_client
        self.embedding_fn = embedding_fn

        # State
        self.state = SystemState.INITIALIZING
        self._databases: Dict[str, Any] = {}
        self._schema_profiles: Dict[str, Any] = {}

        # Initialize core components
        self.knowledge_base = KnowledgeBase(embedding_fn, llm_client)

        self.training_pipeline = TrainingPipeline(
            memory_manager=None,  # Will be set after memory init
            embedding_fn=embedding_fn,
            min_confidence_threshold=config.min_confidence_for_pattern,
        )

        self.auto_trainer = AutoTrainer(
            pipeline=self.training_pipeline,
            llm_client=llm_client,
            batch_size=config.learning_batch_size,
        )

        self.deep_reasoner = DeepReasoner(
            llm_client=llm_client,
            knowledge_base=self.knowledge_base,
            max_depth=config.max_reasoning_depth,
        )

        # User preference learning
        self.user_learning = UserLearningEngine(
            llm_client=llm_client,
        ) if config.enable_user_learning else None

        # Business logic learning
        self.business_learning = BusinessLogicLearner(
            llm_client=llm_client,
            knowledge_base=self.knowledge_base,
        ) if config.enable_business_learning else None

        # Research agent (autonomous improvement)
        self.research_agent = ResearchAgent(
            llm_client=llm_client,
            knowledge_base=self.knowledge_base,
            user_learning=self.user_learning,
            business_learning=self.business_learning,
        ) if config.enable_research_agent else None

        # Self-implementer for autonomous improvements
        self.self_implementer = SelfImplementer(
            knowledge_base=self.knowledge_base,
            business_learning=self.business_learning,
            user_learning=self.user_learning,
        ) if config.auto_implement_findings else None

        # These will be initialized after connection
        self.schema_discovery: Optional[SchemaDiscovery] = None
        self.self_healing: Optional[SelfHealingEngine] = None
        self.business_researcher: Optional[BusinessResearcher] = None

        # Statistics
        self._query_count = 0
        self._success_count = 0
        self._heal_count = 0
        self._learn_count = 0
        self._start_time: Optional[datetime] = None

    async def connect(
        self,
        db_configs: List[Dict[str, Any]],
    ) -> Dict[str, bool]:
        """
        Connect to databases and auto-discover schemas

        Returns dict of db_name -> success
        """
        self.state = SystemState.DISCOVERING
        self._start_time = datetime.utcnow()

        results = {}

        for config in db_configs:
            db_name = config.get("name", f"db_{len(self._databases)}")

            try:
                # Create database adapter
                from ..database.multi_db import (
                    DatabaseAdapter,
                    PostgreSQLAdapter,
                    MySQLAdapter,
                    SQLiteAdapter,
                    MSSQLAdapter,
                    ConnectionConfig,
                    DatabaseType,
                )

                conn_config = ConnectionConfig(
                    name=db_name,
                    db_type=DatabaseType(config.get("type", "postgresql")),
                    host=config.get("host", "localhost"),
                    port=config.get("port", 5432),
                    database=config.get("database", ""),
                    username=config.get("username", ""),
                    password=config.get("password", ""),
                )

                # Get appropriate adapter
                adapter_map = {
                    DatabaseType.POSTGRESQL: PostgreSQLAdapter,
                    DatabaseType.MYSQL: MySQLAdapter,
                    DatabaseType.SQLITE: SQLiteAdapter,
                    DatabaseType.MSSQL: MSSQLAdapter,
                }

                adapter_class = adapter_map.get(conn_config.db_type)
                if adapter_class:
                    adapter = adapter_class(conn_config)
                    await adapter.connect()
                    self._databases[db_name] = adapter

                    # Auto-discover schema
                    if self.config.auto_discover_on_connect:
                        await self._discover_database(db_name, adapter)

                    results[db_name] = True
                    logger.info(f"Connected and discovered: {db_name}")
                else:
                    results[db_name] = False
                    logger.warning(f"Unsupported database type: {conn_config.db_type}")

            except Exception as e:
                logger.exception(f"Failed to connect to {db_name}: {e}")
                results[db_name] = False

        # Initialize self-healing with discovered schema
        self._init_healing_engine()

        self.state = SystemState.READY
        logger.info(f"System ready with {len(self._databases)} databases")

        return results

    async def _discover_database(
        self,
        db_name: str,
        adapter: Any,
    ) -> None:
        """Discover and profile a database"""
        # Create discovery engine
        discovery = SchemaDiscovery(
            db_executor=adapter.execute,
            llm_client=self.llm,
        )

        # Discover schema
        profiles = await discovery.discover(
            include_stats=True,
            include_samples=True,
            max_tables=100,
        )

        self._schema_profiles[db_name] = profiles

        # Infer relationships
        inference = RelationshipInference(profiles)
        relationship_graph = inference.build_relationship_graph()

        # Ingest into knowledge base
        await self.knowledge_base.ingest_schema(profiles)

        logger.info(
            f"Discovered {len(profiles)} tables in {db_name} "
            f"with {len(relationship_graph)} relationships"
        )

    def _init_healing_engine(self) -> None:
        """Initialize self-healing engine with discovered schema"""
        if not self._schema_profiles:
            return

        # Combine all schemas
        combined_schema = {}
        for db_name, profiles in self._schema_profiles.items():
            for table_name, profile in profiles.items():
                combined_schema[f"{db_name}.{table_name}"] = profile

        # Get first database executor
        if self._databases:
            first_db = next(iter(self._databases.values()))

            self.self_healing = SelfHealingEngine(
                schema_profiles=combined_schema,
                db_executor=first_db.execute,
                llm_client=self.llm,
                memory_manager=None,
            )

    async def query(
        self,
        question: str,
        user_id: Optional[str] = None,
        force_reasoning: Optional[ReasoningStrategy] = None,
    ) -> QueryResult:
        """
        Process a natural language query with full intelligence pipeline

        1. Translate user terminology
        2. Apply business logic
        3. Analyze complexity
        4. Retrieve knowledge + user patterns
        5. Apply reasoning strategy
        6. Generate and validate SQL
        7. Execute with self-healing
        8. Learn from result
        """
        start_time = datetime.utcnow()
        self.state = SystemState.PROCESSING
        self._query_count += 1

        result = QueryResult(
            success=False,
            question=question,
        )

        try:
            # 1. Translate user terminology (personalized)
            translated_question = question
            if self.user_learning and user_id:
                translated_question = self.user_learning.translate_question(
                    question, user_id
                )
                if translated_question != question:
                    logger.debug(f"Translated: {question} -> {translated_question}")

            # 2. Get user context
            user_context = {}
            if self.user_learning and user_id:
                user_context = self.user_learning.get_user_context(user_id)

            # 3. Get business context
            business_context = ""
            if self.business_learning:
                business_context = self.business_learning.get_context_for_query(
                    translated_question
                )

            # 4. Determine intelligence level
            intelligence_level = await self._assess_complexity(translated_question)
            result.intelligence_level = intelligence_level

            # 5. Get relevant knowledge
            relevant_knowledge = await self.knowledge_base.search(
                translated_question,
                include_related=True,
                limit=10,
            )
            result.used_patterns = len(relevant_knowledge)

            # 6. Get similar examples (global + user-specific)
            examples = await self.auto_trainer.get_relevant_examples(
                translated_question, limit=5
            )

            # Also get user-specific patterns
            if self.user_learning and user_id:
                user_patterns = self.user_learning.get_similar_patterns(
                    translated_question, user_id, limit=3
                )
                # Add to examples
                for pattern in user_patterns:
                    examples.append({
                        "question": pattern.get("question_template", ""),
                        "sql": pattern.get("sql_template", ""),
                    })

            # 7. Build enhanced schema context
            schema_context = self._build_schema_context(translated_question)

            # Add business context
            if business_context:
                schema_context = f"{schema_context}\n\nBusiness Context:\n{business_context}"

            # Add user context
            if user_context:
                frequent_tables = user_context.get("frequent_tables", [])
                if frequent_tables:
                    schema_context += f"\n\nUser frequently uses: {', '.join(frequent_tables)}"

            # 8. Apply reasoning strategy
            reasoning_strategy = force_reasoning or self._select_strategy(intelligence_level)

            reasoning_chain = await self.deep_reasoner.reason(
                question=translated_question,
                schema_context=schema_context,
                strategy=reasoning_strategy,
                examples=[
                    {"question": e.question, "sql": e.sql} if hasattr(e, 'question')
                    else e
                    for e in examples
                ],
            )

            result.reasoning_steps = len(reasoning_chain.steps)
            result.reasoning_chain = reasoning_chain.get_chain_text()
            result.confidence = reasoning_chain.total_confidence

            sql = reasoning_chain.sql_result

            if not sql:
                result.error = "Failed to generate SQL"
                return result

            # 9. Apply business rules to SQL
            if self.business_learning:
                applicable_rules = self.business_learning.get_applicable_rules(
                    translated_question,
                    self._extract_tables_from_sql(sql),
                )
                if applicable_rules:
                    sql = self.business_learning.apply_rules_to_sql(sql, applicable_rules)

            result.sql = sql

            # 10. Execute with self-healing
            if self.config.enable_self_healing and self.self_healing:
                data, final_sql, healed = await self.self_healing.execute_with_healing(
                    sql,
                    max_retries=self.config.max_healing_attempts,
                )
                result.was_healed = healed
                result.sql = final_sql
                if healed:
                    self._heal_count += 1
            else:
                # Direct execution
                db = next(iter(self._databases.values()))
                data = await db.execute(sql)

            # 11. Process results
            result.data = data
            result.row_count = len(data) if data else 0
            result.columns = list(data[0].keys()) if data else []
            result.success = True
            self._success_count += 1

            # 12. Validate against business rules
            if self.business_learning and data:
                valid, issues = self.business_learning.validate_result(
                    translated_question, result.sql, data
                )
                if not valid:
                    logger.warning(f"Business validation issues: {issues}")

            # 13. Generate explanation
            result.explanation = await self._generate_explanation(
                question,
                result.sql,
                result.row_count,
            )

            # 14. Learn from success
            self._learn_count += 1

            # Learn to auto-trainer
            if self.config.enable_auto_learning:
                feedback = LearningFeedback(
                    query_id=str(uuid4()),
                    original_question=question,
                    generated_sql=sql,
                    executed_sql=result.sql,
                    feedback_type=FeedbackType.SUCCESS,
                    success=True,
                    result_count=result.row_count,
                )
                self.auto_trainer.add_feedback(feedback)

            # Learn user preferences
            if self.user_learning and user_id:
                await self.user_learning.learn_from_interaction(
                    user_id=user_id,
                    question=question,
                    generated_sql=sql,
                    executed_sql=result.sql,
                    success=True,
                    result_data=data,
                )

        except Exception as e:
            logger.exception(f"Query failed: {e}")
            result.error = str(e)
            result.success = False

            # Learn from failure
            if self.config.enable_auto_learning and result.sql:
                feedback = LearningFeedback(
                    query_id=str(uuid4()),
                    original_question=question,
                    generated_sql=result.sql or "",
                    executed_sql=result.sql or "",
                    feedback_type=FeedbackType.REJECTION,
                    success=False,
                    error_message=str(e),
                )
                self.auto_trainer.add_feedback(feedback)

            # Learn from user failures too
            if self.user_learning and user_id:
                await self.user_learning.learn_from_interaction(
                    user_id=user_id,
                    question=question,
                    generated_sql=result.sql or "",
                    executed_sql=result.sql or "",
                    success=False,
                )

        finally:
            result.processing_time_ms = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000
            self.state = SystemState.READY

        return result

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL"""
        import re
        tables = re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE)
        return list(set(t.lower() for t in tables))

    async def _assess_complexity(self, question: str) -> IntelligenceLevel:
        """Assess question complexity to determine intelligence level"""
        question_lower = question.lower()

        # Simple indicators
        simple_patterns = ["how many", "list all", "show me", "what is the"]

        # Complex indicators
        complex_patterns = [
            "compare", "trend", "correlation", "percentage",
            "relative to", "over time", "by each", "grouped by",
        ]

        # Expert indicators
        expert_patterns = [
            "optimal", "predict", "forecast", "anomaly",
            "why", "explain", "recommend",
        ]

        if any(p in question_lower for p in expert_patterns):
            return IntelligenceLevel.EXPERT
        elif any(p in question_lower for p in complex_patterns):
            return IntelligenceLevel.COMPLEX
        elif len(question.split()) > 20:
            return IntelligenceLevel.MODERATE
        elif any(p in question_lower for p in simple_patterns):
            return IntelligenceLevel.SIMPLE

        return IntelligenceLevel.MODERATE

    def _select_strategy(
        self,
        intelligence_level: IntelligenceLevel,
    ) -> ReasoningStrategy:
        """Select reasoning strategy based on intelligence level"""
        strategy_map = {
            IntelligenceLevel.SIMPLE: ReasoningStrategy.CHAIN_OF_THOUGHT,
            IntelligenceLevel.MODERATE: ReasoningStrategy.CHAIN_OF_THOUGHT,
            IntelligenceLevel.COMPLEX: ReasoningStrategy.TREE_OF_THOUGHT,
            IntelligenceLevel.EXPERT: ReasoningStrategy.DECOMPOSITION,
        }

        return strategy_map.get(
            intelligence_level,
            self.config.default_reasoning_strategy,
        )

    def _build_schema_context(self, question: str) -> str:
        """Build relevant schema context for question"""
        # Get all table descriptions
        context_parts = []

        for db_name, profiles in self._schema_profiles.items():
            for table_name, profile in profiles.items():
                desc = f"Table {table_name}"
                if hasattr(profile, 'description') and profile.description:
                    desc += f": {profile.description}"

                columns = []
                if hasattr(profile, 'columns'):
                    for col in profile.columns[:10]:  # Limit columns
                        col_name = col.name if hasattr(col, 'name') else str(col)
                        col_type = col.data_type if hasattr(col, 'data_type') else ""
                        columns.append(f"{col_name} ({col_type})")

                if columns:
                    desc += f"\n  Columns: {', '.join(columns)}"

                context_parts.append(desc)

        return "\n".join(context_parts)

    async def _generate_explanation(
        self,
        question: str,
        sql: str,
        row_count: int,
    ) -> str:
        """Generate natural language explanation of result"""
        prompt = f"""
        Explain this SQL query result in plain English:

        Question: {question}
        SQL: {sql}
        Rows returned: {row_count}

        Provide a brief explanation (2-3 sentences):
        """

        try:
            return await self.llm.generate(prompt=prompt, max_tokens=150)
        except:
            return ""

    async def provide_feedback(
        self,
        query_id: str,
        feedback_type: FeedbackType,
        original_question: str = "",
        generated_sql: str = "",
        correction: Optional[str] = None,
        rating: Optional[float] = None,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Provide feedback for a query to improve learning

        This is how the system learns from:
        - User corrections
        - Positive/negative ratings
        - Comments explaining what was wrong
        """
        feedback = LearningFeedback(
            query_id=query_id,
            original_question=original_question,
            generated_sql=generated_sql,
            executed_sql=generated_sql,
            feedback_type=feedback_type,
            user_correction=correction,
            user_rating=rating,
            user_comment=comment,
        )

        learned_items = 0

        # Process through training pipeline
        if self.config.enable_auto_learning:
            await self.training_pipeline.process_feedback(feedback)
            learned_items += 1

        # If correction provided, learn business rules
        if correction and self.business_learning:
            rules = await self.business_learning.learn_from_correction(
                question=original_question,
                wrong_sql=generated_sql,
                correct_sql=correction,
                user_explanation=comment,
            )
            learned_items += len(rules)

        # Learn user preferences from correction
        if correction and self.user_learning and user_id:
            await self.user_learning.learn_from_interaction(
                user_id=user_id,
                question=original_question,
                generated_sql=generated_sql,
                executed_sql=correction,
                success=True,
                user_correction=correction,
            )
            learned_items += 1

        self._learn_count += learned_items

        return {
            "processed": True,
            "items_learned": learned_items,
            "feedback_type": feedback_type.value,
        }

    async def add_business_knowledge(
        self,
        knowledge_type: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Add business knowledge to the system

        Types:
        - terminology: Define business terms
        - metric: Define how metrics are calculated
        - rule: Business rules to apply
        - documentation: General documentation
        """
        try:
            if knowledge_type == "terminology" and self.business_learning:
                term = metadata.get("term", "") if metadata else ""
                synonyms = metadata.get("synonyms", []) if metadata else []
                self.business_learning.add_terminology(term, content, synonyms)

            elif knowledge_type == "metric" and self.business_learning:
                from .business_learning import MetricDefinition
                metric = MetricDefinition(
                    name=metadata.get("name", "") if metadata else "",
                    display_name=metadata.get("display_name", "") if metadata else "",
                    description=content,
                    sql_template=metadata.get("calculation", "") if metadata else "",
                    base_table=metadata.get("table", "") if metadata else "",
                )
                self.business_learning.add_metric(metric)

            elif knowledge_type == "rule" and self.business_learning:
                from .business_learning import BusinessRule, BusinessRuleType, RuleConfidence
                rule = BusinessRule(
                    type=BusinessRuleType.DATA_FILTER,
                    name=metadata.get("name", "") if metadata else "",
                    description=content,
                    action=metadata.get("sql", "") if metadata else "",
                    condition=metadata.get("applies_when", "") if metadata else "",
                    confidence=RuleConfidence.EXPLICIT,
                    verified=True,
                )
                self.business_learning.add_rule(rule)

            elif knowledge_type == "documentation":
                if self.business_learning:
                    count = await self.business_learning.learn_from_documentation(
                        content, source="user_provided"
                    )
                    return count > 0

            return True

        except Exception as e:
            logger.warning(f"Failed to add business knowledge: {e}")
            return False

    async def start_research_loop(self) -> None:
        """Start the autonomous research and improvement loop"""
        if self.research_agent and self.config.enable_research_agent:
            asyncio.create_task(
                self.research_agent.continuous_improvement_loop(
                    get_stats_fn=self.get_stats,
                    interval_minutes=self.config.research_interval_minutes,
                )
            )
            logger.info("Research agent started")

    async def refresh_knowledge(self) -> None:
        """Refresh all knowledge (schemas, patterns, etc.)"""
        self.state = SystemState.DISCOVERING

        # Re-discover schemas
        for db_name, adapter in self._databases.items():
            await self._discover_database(db_name, adapter)

        # Refresh self-healing
        self._init_healing_engine()

        self.state = SystemState.READY
        logger.info("Knowledge refresh complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        uptime = (
            (datetime.utcnow() - self._start_time).total_seconds()
            if self._start_time else 0
        )

        stats = {
            "state": self.state.value,
            "uptime_seconds": uptime,
            "databases_connected": len(self._databases),
            "tables_discovered": sum(
                len(p) for p in self._schema_profiles.values()
            ),
            "queries_processed": self._query_count,
            "success_rate": (
                self._success_count / self._query_count
                if self._query_count > 0 else 0
            ),
            "queries_healed": self._heal_count,
            "learning_events": self._learn_count,
            "knowledge_items": self.knowledge_base.get_stats(),
            "training_examples": len(self.training_pipeline._examples),
        }

        # Add user learning stats
        if self.user_learning:
            stats["users_tracked"] = len(self.user_learning._profiles)
            stats["terminology_mappings"] = len(
                self.user_learning._terminology._global_mappings
            )

        # Add business learning stats
        if self.business_learning:
            stats["business_stats"] = self.business_learning.get_stats()

        # Add research agent stats
        if self.research_agent:
            stats["research_tasks"] = len(self.research_agent._tasks)
            stats["research_findings"] = len(self.research_agent._findings)
            stats["improvements_made"] = self.research_agent._improvements_made

        return stats

    async def shutdown(self) -> None:
        """Gracefully shutdown the system"""
        logger.info("Shutting down intelligent system...")

        # Stop auto-trainer
        await self.auto_trainer.stop()

        # Disconnect databases
        for db_name, adapter in self._databases.items():
            try:
                await adapter.disconnect()
            except:
                pass

        self._databases.clear()
        self.state = SystemState.INITIALIZING

        logger.info("Shutdown complete")
