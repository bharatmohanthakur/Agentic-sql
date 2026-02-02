"""
Research Agent - Actively researches, thinks, and implements improvements
Autonomous agent for continuous system improvement
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

logger = logging.getLogger(__name__)


class ResearchGoal(str, Enum):
    """Types of research goals"""
    IMPROVE_ACCURACY = "improve_accuracy"  # Better SQL generation
    LEARN_DOMAIN = "learn_domain"  # Understand business domain
    OPTIMIZE_PERFORMANCE = "optimize_performance"  # Faster queries
    EXPAND_COVERAGE = "expand_coverage"  # Handle more query types
    REDUCE_ERRORS = "reduce_errors"  # Fix common errors
    USER_SATISFACTION = "user_satisfaction"  # Better user experience


class ResearchStatus(str, Enum):
    """Research task status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    IMPLEMENTED = "implemented"


@dataclass
class ResearchFinding:
    """A single research finding"""
    id: str = field(default_factory=lambda: str(uuid4()))
    topic: str = ""
    finding: str = ""
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    actionable: bool = False
    action_suggested: str = ""
    implemented: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResearchTask:
    """A research task"""
    id: str = field(default_factory=lambda: str(uuid4()))
    goal: ResearchGoal = ResearchGoal.IMPROVE_ACCURACY
    question: str = ""  # What to research
    hypothesis: str = ""  # Expected finding
    status: ResearchStatus = ResearchStatus.PENDING
    findings: List[ResearchFinding] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class ThinkingEngine:
    """
    Deep thinking engine for analysis and reasoning
    Uses multiple reasoning strategies
    """

    def __init__(self, llm_client: Any):
        self.llm = llm_client

    async def analyze_problem(
        self,
        problem: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deeply analyze a problem"""
        prompt = f"""
        Analyze this problem thoroughly:

        Problem: {problem}

        Context:
        {json.dumps(context, indent=2)}

        Think step by step:
        1. What is the core issue?
        2. What are the root causes?
        3. What are possible solutions?
        4. What are the trade-offs?
        5. What is the best approach?

        Return as JSON:
        {{
            "core_issue": "...",
            "root_causes": ["cause1", "cause2"],
            "solutions": [
                {{"solution": "...", "pros": [...], "cons": [...], "effort": "low|medium|high"}}
            ],
            "recommended_solution": "...",
            "implementation_steps": ["step1", "step2"],
            "confidence": 0.0-1.0
        }}

        JSON:
        """

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=1000)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {"error": "Analysis failed"}

    async def generate_hypotheses(
        self,
        observation: str,
        domain_context: str,
    ) -> List[Dict[str, Any]]:
        """Generate hypotheses from observations"""
        prompt = f"""
        Given this observation, generate hypotheses:

        Observation: {observation}

        Domain Context:
        {domain_context}

        Generate 3-5 hypotheses that could explain this observation.
        For each hypothesis, include how to test it.

        Return as JSON array:
        [
            {{
                "hypothesis": "...",
                "reasoning": "why this might be true",
                "test_method": "how to verify",
                "confidence": 0.0-1.0
            }}
        ]

        JSON:
        """

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=800)
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return []

    async def reason_about_improvement(
        self,
        current_state: Dict[str, Any],
        goal: str,
    ) -> Dict[str, Any]:
        """Reason about how to improve from current state to goal"""
        prompt = f"""
        Reason about how to improve the system:

        Current State:
        {json.dumps(current_state, indent=2)}

        Goal: {goal}

        Think deeply about:
        1. Gap between current state and goal
        2. What's blocking progress
        3. Quick wins available
        4. Long-term improvements needed

        Return as JSON:
        {{
            "gap_analysis": "...",
            "blockers": ["blocker1", "blocker2"],
            "quick_wins": [
                {{"action": "...", "impact": "high|medium|low", "effort": "low|medium|high"}}
            ],
            "long_term_improvements": [
                {{"action": "...", "timeline": "...", "dependencies": [...]}}
            ],
            "priority_order": ["action1", "action2"]
        }}

        JSON:
        """

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=1000)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {"error": "Reasoning failed"}


class ResearchAgent:
    """
    Autonomous research agent that:
    1. Identifies areas for improvement
    2. Researches solutions
    3. Tests hypotheses
    4. Implements improvements
    5. Validates results
    """

    def __init__(
        self,
        llm_client: Any,
        db_executor: Optional[Callable] = None,
        knowledge_base: Optional[Any] = None,
        user_learning: Optional[Any] = None,
        business_learning: Optional[Any] = None,
    ):
        self.llm = llm_client
        self.db_executor = db_executor
        self.knowledge = knowledge_base
        self.user_learning = user_learning
        self.business_learning = business_learning

        self.thinking = ThinkingEngine(llm_client)
        self._tasks: Dict[str, ResearchTask] = {}
        self._findings: List[ResearchFinding] = []
        self._improvements_made: int = 0

    async def identify_improvement_opportunities(
        self,
        system_stats: Dict[str, Any],
    ) -> List[ResearchTask]:
        """Identify areas that need improvement"""
        tasks = []

        # Check success rate
        success_rate = system_stats.get("success_rate", 1.0)
        if success_rate < 0.9:
            task = ResearchTask(
                goal=ResearchGoal.REDUCE_ERRORS,
                question="Why are queries failing? What patterns cause failures?",
                hypothesis="Common query patterns are not well supported",
            )
            tasks.append(task)
            self._tasks[task.id] = task

        # Check if learning is happening
        training_examples = system_stats.get("training_examples", 0)
        if training_examples < 10:
            task = ResearchTask(
                goal=ResearchGoal.LEARN_DOMAIN,
                question="What domain knowledge is missing?",
                hypothesis="System needs more training examples",
            )
            tasks.append(task)
            self._tasks[task.id] = task

        # Check knowledge coverage
        knowledge_items = system_stats.get("knowledge_items", {}).get("total_items", 0)
        if knowledge_items < 50:
            task = ResearchTask(
                goal=ResearchGoal.EXPAND_COVERAGE,
                question="What knowledge gaps exist in the system?",
                hypothesis="Schema and business rules are not fully understood",
            )
            tasks.append(task)
            self._tasks[task.id] = task

        return tasks

    async def research(self, task: ResearchTask) -> List[ResearchFinding]:
        """Conduct research for a task"""
        task.status = ResearchStatus.IN_PROGRESS
        findings = []

        logger.info(f"Starting research: {task.question}")

        if task.goal == ResearchGoal.REDUCE_ERRORS:
            findings = await self._research_error_patterns()

        elif task.goal == ResearchGoal.LEARN_DOMAIN:
            findings = await self._research_domain_knowledge()

        elif task.goal == ResearchGoal.EXPAND_COVERAGE:
            findings = await self._research_coverage_gaps()

        elif task.goal == ResearchGoal.IMPROVE_ACCURACY:
            findings = await self._research_accuracy_improvements()

        task.findings = findings
        task.status = ResearchStatus.COMPLETED
        task.completed_at = datetime.utcnow()

        self._findings.extend(findings)

        logger.info(f"Research completed: {len(findings)} findings")

        return findings

    async def _research_error_patterns(self) -> List[ResearchFinding]:
        """Research common error patterns"""
        findings = []

        # Analyze what types of errors occur
        analysis = await self.thinking.analyze_problem(
            problem="Queries are failing at a rate above acceptable threshold",
            context={
                "goal": "Reduce query failure rate",
                "available_data": "error logs, failed queries, user feedback",
            },
        )

        if analysis.get("root_causes"):
            finding = ResearchFinding(
                topic="Error Root Causes",
                finding=f"Main causes: {analysis['root_causes']}",
                evidence=analysis.get("solutions", []),
                confidence=analysis.get("confidence", 0.5),
                actionable=True,
                action_suggested=analysis.get("recommended_solution", ""),
            )
            findings.append(finding)

        return findings

    async def _research_domain_knowledge(self) -> List[ResearchFinding]:
        """Research missing domain knowledge"""
        findings = []

        if self.db_executor:
            # Get list of tables to research
            try:
                tables = await self.db_executor("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    LIMIT 20
                """)

                for table in tables:
                    table_name = table.get("table_name")

                    # Research business meaning
                    meaning = await self._research_table_meaning(table_name)

                    finding = ResearchFinding(
                        topic=f"Table: {table_name}",
                        finding=meaning.get("purpose", "Unknown purpose"),
                        evidence=meaning.get("key_concepts", []),
                        confidence=0.7,
                        actionable=True,
                        action_suggested=f"Add business context for {table_name}",
                    )
                    findings.append(finding)
            except:
                pass

        return findings

    async def _research_table_meaning(self, table_name: str) -> Dict[str, Any]:
        """Research what a table means in business context"""
        if not self.db_executor:
            return {}

        try:
            # Get sample data
            sample = await self.db_executor(
                f"SELECT * FROM {table_name} LIMIT 5"
            )

            prompt = f"""
            What is the business purpose of this database table?

            Table: {table_name}
            Sample data: {json.dumps(sample)}

            Return JSON:
            {{
                "purpose": "business purpose",
                "key_concepts": ["concept1", "concept2"],
                "common_queries": ["what users might ask about this"]
            }}

            JSON:
            """

            response = await self.llm.generate(prompt=prompt, max_tokens=300)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {}

    async def _research_coverage_gaps(self) -> List[ResearchFinding]:
        """Research what knowledge is missing"""
        findings = []

        # Generate hypotheses about gaps
        hypotheses = await self.thinking.generate_hypotheses(
            observation="System has limited knowledge coverage",
            domain_context="Text-to-SQL system needs schema, business rules, and query patterns",
        )

        for hyp in hypotheses:
            finding = ResearchFinding(
                topic="Coverage Gap",
                finding=hyp.get("hypothesis", ""),
                evidence=[hyp.get("reasoning", "")],
                confidence=hyp.get("confidence", 0.5),
                actionable=True,
                action_suggested=hyp.get("test_method", ""),
            )
            findings.append(finding)

        return findings

    async def _research_accuracy_improvements(self) -> List[ResearchFinding]:
        """Research how to improve accuracy"""
        findings = []

        # Think about improvements
        improvement_plan = await self.thinking.reason_about_improvement(
            current_state={
                "current_accuracy": "unknown",
                "known_issues": ["complex joins", "aggregations", "time filters"],
            },
            goal="Achieve 95%+ query accuracy",
        )

        if improvement_plan.get("quick_wins"):
            for win in improvement_plan["quick_wins"]:
                finding = ResearchFinding(
                    topic="Quick Win",
                    finding=win.get("action", ""),
                    evidence=[f"Impact: {win.get('impact')}", f"Effort: {win.get('effort')}"],
                    confidence=0.8,
                    actionable=True,
                    action_suggested=win.get("action", ""),
                )
                findings.append(finding)

        return findings

    async def implement_finding(
        self,
        finding: ResearchFinding,
    ) -> bool:
        """Implement an actionable finding"""
        if not finding.actionable:
            return False

        logger.info(f"Implementing finding: {finding.topic}")

        try:
            # Different implementation strategies based on topic
            if "Error" in finding.topic:
                # Add error handling pattern
                if self.business_learning:
                    from .business_learning import BusinessRule, BusinessRuleType, RuleConfidence

                    rule = BusinessRule(
                        type=BusinessRuleType.VALIDATION,
                        name=f"Error Prevention: {finding.topic}",
                        description=finding.finding,
                        action=finding.action_suggested,
                        confidence=RuleConfidence.INFERRED,
                        source="research_agent",
                    )
                    self.business_learning.add_rule(rule)

            elif "Table" in finding.topic:
                # Add domain knowledge
                if self.knowledge:
                    from .knowledge_base import KnowledgeType

                    await self.knowledge.add(
                        content=finding.finding,
                        knowledge_type=KnowledgeType.SCHEMA,
                        tags=finding.evidence,
                        source="research_agent",
                    )

            elif "Coverage" in finding.topic or "Quick Win" in finding.topic:
                # Log for manual implementation
                logger.info(f"Suggested improvement: {finding.action_suggested}")

            finding.implemented = True
            self._improvements_made += 1

            return True

        except Exception as e:
            logger.warning(f"Implementation failed: {e}")
            return False

    async def run_research_cycle(
        self,
        system_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a complete research cycle"""
        logger.info("Starting research cycle...")

        # 1. Identify opportunities
        tasks = await self.identify_improvement_opportunities(system_stats)
        logger.info(f"Identified {len(tasks)} research opportunities")

        # 2. Conduct research
        all_findings = []
        for task in tasks:
            findings = await self.research(task)
            all_findings.extend(findings)

        # 3. Implement actionable findings
        implemented = 0
        for finding in all_findings:
            if finding.actionable and finding.confidence > 0.6:
                success = await self.implement_finding(finding)
                if success:
                    implemented += 1

        logger.info(
            f"Research cycle complete: {len(all_findings)} findings, "
            f"{implemented} implemented"
        )

        return {
            "tasks_created": len(tasks),
            "findings": len(all_findings),
            "implemented": implemented,
            "total_improvements": self._improvements_made,
        }

    async def continuous_improvement_loop(
        self,
        get_stats_fn: Callable,
        interval_minutes: int = 60,
    ) -> None:
        """Run continuous improvement in background"""
        logger.info("Starting continuous improvement loop")

        while True:
            try:
                stats = await get_stats_fn()
                await self.run_research_cycle(stats)
            except Exception as e:
                logger.warning(f"Research cycle failed: {e}")

            await asyncio.sleep(interval_minutes * 60)


class SelfImplementer:
    """
    Implements improvements automatically
    Can modify system behavior based on research findings
    """

    def __init__(
        self,
        knowledge_base: Any,
        business_learning: Any,
        user_learning: Any,
    ):
        self.knowledge = knowledge_base
        self.business = business_learning
        self.users = user_learning

    async def implement_terminology_mapping(
        self,
        user_term: str,
        db_term: str,
        scope: str = "global",  # global or user_id
    ) -> bool:
        """Implement a terminology mapping"""
        try:
            self.business.add_terminology(user_term, f"Maps to: {db_term}")
            return True
        except:
            return False

    async def implement_business_rule(
        self,
        rule_name: str,
        rule_description: str,
        sql_pattern: str,
        applies_when: str,
    ) -> bool:
        """Implement a new business rule"""
        try:
            from .business_learning import BusinessRule, BusinessRuleType, RuleConfidence

            rule = BusinessRule(
                type=BusinessRuleType.DATA_FILTER,
                name=rule_name,
                description=rule_description,
                action=sql_pattern,
                condition=applies_when,
                confidence=RuleConfidence.INFERRED,
                source="self_implementer",
            )
            self.business.add_rule(rule)
            return True
        except:
            return False

    async def implement_metric(
        self,
        name: str,
        display_name: str,
        calculation: str,
        base_table: str,
    ) -> bool:
        """Implement a new metric definition"""
        try:
            from .business_learning import MetricDefinition

            metric = MetricDefinition(
                name=name,
                display_name=display_name,
                sql_template=calculation,
                base_table=base_table,
            )
            self.business.add_metric(metric)
            return True
        except:
            return False
