# Intelligence Module - Auto-discovery, Auto-learning, Self-healing
from .auto_discovery import SchemaDiscovery, RelationshipInference, DataProfiler
from .auto_trainer import AutoTrainer, TrainingPipeline, LearningFeedback
from .self_healing import SelfHealingEngine, ErrorCorrector, QueryOptimizer
from .deep_reasoner import DeepReasoner, ReasoningChain, KnowledgeSynthesis
from .knowledge_base import KnowledgeBase, KnowledgeGraph, SemanticIndex
from .user_learning import UserLearningEngine, UserProfile, TerminologyMapper
from .business_learning import BusinessLogicLearner, BusinessRule, MetricDefinition
from .research_agent import ResearchAgent, ThinkingEngine, SelfImplementer
from .intelligent_system import IntelligentSystem, IntelligentSystemConfig

__all__ = [
    # Auto-discovery
    "SchemaDiscovery",
    "RelationshipInference",
    "DataProfiler",
    # Auto-training
    "AutoTrainer",
    "TrainingPipeline",
    "LearningFeedback",
    # Self-healing
    "SelfHealingEngine",
    "ErrorCorrector",
    "QueryOptimizer",
    # Deep reasoning
    "DeepReasoner",
    "ReasoningChain",
    "KnowledgeSynthesis",
    # Knowledge
    "KnowledgeBase",
    "KnowledgeGraph",
    "SemanticIndex",
    # User learning
    "UserLearningEngine",
    "UserProfile",
    "TerminologyMapper",
    # Business learning
    "BusinessLogicLearner",
    "BusinessRule",
    "MetricDefinition",
    # Research & Implementation
    "ResearchAgent",
    "ThinkingEngine",
    "SelfImplementer",
    # Main system
    "IntelligentSystem",
    "IntelligentSystemConfig",
]
