"""
Auto-Training System - Self-learning from interactions
Learns from successful queries, user corrections, and patterns
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of learning feedback"""
    SUCCESS = "success"  # Query executed successfully
    CORRECTION = "correction"  # User corrected the SQL
    REJECTION = "rejection"  # User rejected the result
    REFINEMENT = "refinement"  # User asked for refinement
    POSITIVE = "positive"  # Explicit positive feedback
    NEGATIVE = "negative"  # Explicit negative feedback


class TrainingExampleType(str, Enum):
    """Types of training examples"""
    QUESTION_SQL = "question_sql"  # NL question -> SQL pair
    SCHEMA_CONTEXT = "schema_context"  # Schema understanding
    ERROR_PATTERN = "error_pattern"  # Error to avoid
    QUERY_TEMPLATE = "query_template"  # Reusable patterns
    BUSINESS_RULE = "business_rule"  # Domain knowledge


@dataclass
class TrainingExample:
    """A single training example"""
    id: str = field(default_factory=lambda: str(uuid4()))
    type: TrainingExampleType = TrainingExampleType.QUESTION_SQL

    # Content
    question: str = ""
    sql: str = ""
    schema_context: str = ""
    tables_used: List[str] = field(default_factory=list)

    # Metadata
    success: bool = True
    feedback_score: float = 1.0
    use_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Embedding for similarity search
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "question": self.question,
            "sql": self.sql,
            "schema_context": self.schema_context,
            "tables_used": self.tables_used,
            "success": self.success,
            "feedback_score": self.feedback_score,
            "use_count": self.use_count,
        }


@dataclass
class LearningFeedback:
    """Feedback from a query execution"""
    query_id: str
    original_question: str
    generated_sql: str
    executed_sql: str
    feedback_type: FeedbackType

    # Result info
    success: bool = True
    error_message: Optional[str] = None
    result_count: int = 0
    execution_time_ms: float = 0.0

    # User feedback
    user_rating: Optional[float] = None  # 1-5
    user_correction: Optional[str] = None
    user_comment: Optional[str] = None

    timestamp: datetime = field(default_factory=datetime.utcnow)


class TrainingPipeline:
    """
    Training pipeline for processing feedback and updating knowledge
    """

    def __init__(
        self,
        memory_manager: Any,
        embedding_fn: Optional[Callable] = None,
        min_confidence_threshold: float = 0.7,
    ):
        self.memory = memory_manager
        self.embedding_fn = embedding_fn
        self.min_confidence = min_confidence_threshold
        self._pending_feedback: List[LearningFeedback] = []
        self._examples: Dict[str, TrainingExample] = {}

    async def process_feedback(self, feedback: LearningFeedback) -> None:
        """Process a single feedback item"""
        logger.info(f"Processing feedback: {feedback.feedback_type.value}")

        if feedback.feedback_type == FeedbackType.SUCCESS:
            await self._learn_from_success(feedback)
        elif feedback.feedback_type == FeedbackType.CORRECTION:
            await self._learn_from_correction(feedback)
        elif feedback.feedback_type == FeedbackType.REJECTION:
            await self._learn_from_rejection(feedback)
        elif feedback.feedback_type in [FeedbackType.POSITIVE, FeedbackType.NEGATIVE]:
            await self._update_confidence(feedback)

    async def _learn_from_success(self, feedback: LearningFeedback) -> None:
        """Learn from a successful query"""
        # Create training example
        example = TrainingExample(
            type=TrainingExampleType.QUESTION_SQL,
            question=feedback.original_question,
            sql=feedback.executed_sql,
            success=True,
            feedback_score=1.0,
        )

        # Generate embedding
        if self.embedding_fn:
            example.embedding = await self._generate_embedding(
                f"{example.question}\n{example.sql}"
            )

        # Check for similar existing examples
        similar = await self._find_similar_examples(example)

        if similar:
            # Update existing example's confidence
            similar[0].use_count += 1
            similar[0].feedback_score = min(1.0, similar[0].feedback_score + 0.1)
            similar[0].updated_at = datetime.utcnow()
        else:
            # Store new example
            self._examples[example.id] = example

            # Also store in memory system
            if self.memory:
                from ..memory.manager import MemoryType, MemoryPriority
                await self.memory.ingest(
                    content=json.dumps(example.to_dict()),
                    memory_type=MemoryType.QUERY_PATTERN,
                    priority=MemoryPriority.HIGH,
                )

    async def _learn_from_correction(self, feedback: LearningFeedback) -> None:
        """Learn from a user correction"""
        if not feedback.user_correction:
            return

        # Store corrected example as high-value
        example = TrainingExample(
            type=TrainingExampleType.QUESTION_SQL,
            question=feedback.original_question,
            sql=feedback.user_correction,  # Use corrected SQL
            success=True,
            feedback_score=1.5,  # Higher score for corrections
        )

        # Store the error pattern to avoid
        error_example = TrainingExample(
            type=TrainingExampleType.ERROR_PATTERN,
            question=feedback.original_question,
            sql=feedback.generated_sql,  # Original wrong SQL
            success=False,
            feedback_score=0.0,
        )

        self._examples[example.id] = example
        self._examples[error_example.id] = error_example

        logger.info(f"Learned correction: {feedback.original_question[:50]}...")

    async def _learn_from_rejection(self, feedback: LearningFeedback) -> None:
        """Learn from a rejected query"""
        # Decrease confidence in this pattern
        example_id = self._find_example_by_sql(feedback.generated_sql)

        if example_id and example_id in self._examples:
            self._examples[example_id].feedback_score *= 0.5
            self._examples[example_id].updated_at = datetime.utcnow()

        # Store as error pattern
        error_example = TrainingExample(
            type=TrainingExampleType.ERROR_PATTERN,
            question=feedback.original_question,
            sql=feedback.generated_sql,
            success=False,
            feedback_score=0.0,
        )
        self._examples[error_example.id] = error_example

    async def _update_confidence(self, feedback: LearningFeedback) -> None:
        """Update confidence based on explicit feedback"""
        example_id = self._find_example_by_sql(feedback.executed_sql)

        if example_id and example_id in self._examples:
            if feedback.feedback_type == FeedbackType.POSITIVE:
                self._examples[example_id].feedback_score = min(
                    2.0, self._examples[example_id].feedback_score + 0.2
                )
            else:
                self._examples[example_id].feedback_score *= 0.8

    async def _find_similar_examples(
        self,
        example: TrainingExample,
        threshold: float = 0.9,
    ) -> List[TrainingExample]:
        """Find similar existing examples"""
        similar = []

        if example.embedding:
            for existing in self._examples.values():
                if existing.embedding:
                    similarity = self._cosine_similarity(
                        example.embedding,
                        existing.embedding,
                    )
                    if similarity > threshold:
                        similar.append(existing)

        return similar

    def _find_example_by_sql(self, sql: str) -> Optional[str]:
        """Find example by SQL hash"""
        sql_hash = hashlib.sha256(sql.encode()).hexdigest()

        for example_id, example in self._examples.items():
            if hashlib.sha256(example.sql.encode()).hexdigest() == sql_hash:
                return example_id

        return None

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if asyncio.iscoroutinefunction(self.embedding_fn):
            return await self.embedding_fn(text)
        return self.embedding_fn(text)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_training_examples(
        self,
        min_score: float = 0.5,
        example_type: Optional[TrainingExampleType] = None,
    ) -> List[TrainingExample]:
        """Get high-quality training examples"""
        examples = list(self._examples.values())

        # Filter by score
        examples = [e for e in examples if e.feedback_score >= min_score]

        # Filter by type
        if example_type:
            examples = [e for e in examples if e.type == example_type]

        # Sort by score
        examples.sort(key=lambda e: e.feedback_score, reverse=True)

        return examples


class AutoTrainer:
    """
    Auto-training orchestrator

    Features:
    - Continuous learning from interactions
    - Pattern extraction from successful queries
    - Schema understanding improvement
    - Query template generation
    """

    def __init__(
        self,
        pipeline: TrainingPipeline,
        llm_client: Optional[Any] = None,
        batch_size: int = 10,
        training_interval_seconds: int = 300,
    ):
        self.pipeline = pipeline
        self.llm = llm_client
        self.batch_size = batch_size
        self.training_interval = training_interval_seconds
        self._running = False
        self._feedback_buffer: List[LearningFeedback] = []

    async def start(self) -> None:
        """Start background training loop"""
        self._running = True
        logger.info("Auto-trainer started")

        while self._running:
            await self._training_cycle()
            await asyncio.sleep(self.training_interval)

    async def stop(self) -> None:
        """Stop training loop"""
        self._running = False
        logger.info("Auto-trainer stopped")

    def add_feedback(self, feedback: LearningFeedback) -> None:
        """Add feedback to buffer"""
        self._feedback_buffer.append(feedback)

        # Process immediately if buffer is full
        if len(self._feedback_buffer) >= self.batch_size:
            asyncio.create_task(self._process_buffer())

    async def _training_cycle(self) -> None:
        """Single training cycle"""
        if self._feedback_buffer:
            await self._process_buffer()

        # Extract patterns from accumulated examples
        await self._extract_patterns()

        # Prune low-quality examples
        self._prune_examples()

    async def _process_buffer(self) -> None:
        """Process buffered feedback"""
        buffer = self._feedback_buffer.copy()
        self._feedback_buffer.clear()

        for feedback in buffer:
            await self.pipeline.process_feedback(feedback)

        logger.info(f"Processed {len(buffer)} feedback items")

    async def _extract_patterns(self) -> None:
        """Extract reusable patterns from examples"""
        if not self.llm:
            return

        examples = self.pipeline.get_training_examples(min_score=1.0)

        if len(examples) < 5:
            return

        # Group by tables used
        by_tables: Dict[str, List[TrainingExample]] = {}
        for ex in examples:
            key = ",".join(sorted(ex.tables_used))
            if key not in by_tables:
                by_tables[key] = []
            by_tables[key].append(ex)

        # Extract patterns for groups with multiple examples
        for tables, group in by_tables.items():
            if len(group) >= 3:
                await self._extract_group_pattern(tables, group)

    async def _extract_group_pattern(
        self,
        tables: str,
        examples: List[TrainingExample],
    ) -> None:
        """Extract pattern from a group of similar examples"""
        prompt = f"""
        Analyze these successful SQL queries for the same tables and extract a reusable pattern:

        Tables: {tables}

        Examples:
        {chr(10).join(f"Q: {e.question}\\nSQL: {e.sql}" for e in examples[:5])}

        Generate a query template that captures the common pattern.
        Use placeholders like {{table}}, {{column}}, {{condition}} for variable parts.

        Template:
        """

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=500)

            template = TrainingExample(
                type=TrainingExampleType.QUERY_TEMPLATE,
                question=f"Pattern for {tables}",
                sql=response.strip(),
                tables_used=tables.split(","),
                feedback_score=1.0,
            )

            self.pipeline._examples[template.id] = template
            logger.info(f"Extracted pattern for tables: {tables}")
        except Exception as e:
            logger.warning(f"Pattern extraction failed: {e}")

    def _prune_examples(self) -> None:
        """Remove low-quality examples"""
        to_remove = []

        for example_id, example in self.pipeline._examples.items():
            # Remove old low-score examples
            age_days = (datetime.utcnow() - example.updated_at).days

            if example.feedback_score < 0.3 and age_days > 7:
                to_remove.append(example_id)
            elif example.feedback_score < 0.1:
                to_remove.append(example_id)

        for example_id in to_remove:
            del self.pipeline._examples[example_id]

        if to_remove:
            logger.info(f"Pruned {len(to_remove)} low-quality examples")

    async def get_relevant_examples(
        self,
        question: str,
        limit: int = 5,
    ) -> List[TrainingExample]:
        """Get relevant examples for a question"""
        # Get all good examples
        all_examples = self.pipeline.get_training_examples(min_score=0.7)

        if not all_examples:
            return []

        # Simple keyword matching for now
        # In production, use embedding similarity
        question_words = set(question.lower().split())

        scored = []
        for example in all_examples:
            example_words = set(example.question.lower().split())
            overlap = len(question_words & example_words)
            score = overlap * example.feedback_score
            scored.append((score, example))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [ex for _, ex in scored[:limit]]
