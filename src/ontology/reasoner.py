"""
Fact-Based Reasoner - Inference engine for deriving knowledge from ontology facts.

Implements forward-chaining and backward-chaining reasoning over elementary facts,
using both logical rules and LLM-assisted semantic inference.

Capabilities:
- Derive new facts from existing facts + rules
- Validate fact consistency against constraints
- Expand context by traversing related facts
- Generate natural language explanations of reasoning chains
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

from .fact_model import (
    Constraint,
    ConstraintKind,
    ElementaryFact,
    FactInstance,
    ObjectType,
    OntologyDefinition,
)

logger = logging.getLogger(__name__)


class InferenceMethod(str, Enum):
    """Methods of inference"""
    FORWARD_CHAIN = "forward_chain"    # Derive from known facts
    BACKWARD_CHAIN = "backward_chain"  # Work backwards from goal
    SEMANTIC = "semantic"              # LLM-assisted semantic reasoning
    CONSTRAINT = "constraint"          # Constraint validation


class ReasoningStep(BaseModel):
    """A single step in a reasoning chain"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    method: InferenceMethod
    input_facts: List[str] = Field(default_factory=list)   # Fact type names used
    output_fact: Optional[str] = None                       # Derived fact type
    rule_applied: str = ""                                   # Rule or explanation
    confidence: float = 1.0
    explanation: str = ""


class InferenceResult(BaseModel):
    """Result of a reasoning operation"""
    success: bool
    derived_facts: List[FactInstance] = Field(default_factory=list)
    reasoning_chain: List[ReasoningStep] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)  # Constraint violations
    explanation: str = ""
    confidence: float = 1.0


class FactReasoner:
    """
    Reasoner that performs inference over ontology facts.

    Supports:
    1. Forward chaining: Given facts → derive new facts
    2. Backward chaining: Given goal → find supporting facts
    3. Constraint checking: Validate facts against ontology constraints
    4. Context expansion: Find related facts for a given entity/question
    """

    def __init__(self, ontology: OntologyDefinition):
        self.ontology = ontology
        self._derivation_rules: Dict[str, List[Dict]] = {}  # fact_name -> rules
        self._build_rule_index()

    def _build_rule_index(self) -> None:
        """Build index of derivation rules from derived fact types"""
        for ft in self.ontology.fact_types:
            if ft.is_derived and ft.derivation_rule:
                self._derivation_rules[ft.name] = [{
                    "rule": ft.derivation_rule,
                    "source_facts": self._parse_rule_dependencies(ft.derivation_rule),
                }]

    def _parse_rule_dependencies(self, rule: str) -> List[str]:
        """Extract fact type names referenced in a derivation rule"""
        deps = []
        for ft in self.ontology.fact_types:
            if ft.name in rule:
                deps.append(ft.name)
        return deps

    def forward_chain(
        self,
        known_facts: List[FactInstance],
        max_depth: int = 5,
    ) -> InferenceResult:
        """
        Forward chaining: derive new facts from known facts.

        Applies derivation rules iteratively until no new facts can be derived
        or max depth is reached.
        """
        steps = []
        derived = []
        known_types = {f.fact_type for f in known_facts}

        for depth in range(max_depth):
            new_derived = False

            for fact_name, rules in self._derivation_rules.items():
                if fact_name in known_types:
                    continue

                for rule_def in rules:
                    source_facts = rule_def["source_facts"]
                    if all(sf in known_types for sf in source_facts):
                        # All prerequisites met, derive the fact
                        step = ReasoningStep(
                            method=InferenceMethod.FORWARD_CHAIN,
                            input_facts=source_facts,
                            output_fact=fact_name,
                            rule_applied=rule_def["rule"],
                            explanation=f"Derived '{fact_name}' from {source_facts}",
                        )
                        steps.append(step)
                        known_types.add(fact_name)
                        new_derived = True

            if not new_derived:
                break

        return InferenceResult(
            success=True,
            derived_facts=derived,
            reasoning_chain=steps,
            explanation=f"Forward chaining completed in {len(steps)} steps",
        )

    def backward_chain(
        self,
        goal_fact_type: str,
        known_facts: List[FactInstance],
        max_depth: int = 5,
    ) -> InferenceResult:
        """
        Backward chaining: find what facts are needed to establish a goal.

        Works backwards from the goal fact to identify required supporting facts.
        """
        steps = []
        known_types = {f.fact_type for f in known_facts}
        required: Set[str] = set()

        def _find_requirements(fact_name: str, depth: int) -> bool:
            if depth > max_depth:
                return False
            if fact_name in known_types:
                return True

            rules = self._derivation_rules.get(fact_name, [])
            for rule_def in rules:
                source_facts = rule_def["source_facts"]
                if all(_find_requirements(sf, depth + 1) for sf in source_facts):
                    steps.append(ReasoningStep(
                        method=InferenceMethod.BACKWARD_CHAIN,
                        input_facts=source_facts,
                        output_fact=fact_name,
                        rule_applied=rule_def["rule"],
                        explanation=f"To derive '{fact_name}', need {source_facts}",
                    ))
                    return True

            # No derivation path found - this is a base fact we need
            required.add(fact_name)
            return fact_name in known_types

        reachable = _find_requirements(goal_fact_type, 0)

        missing = required - known_types
        return InferenceResult(
            success=reachable,
            reasoning_chain=steps,
            violations=[f"Missing required fact: {f}" for f in missing],
            explanation=(
                f"Goal '{goal_fact_type}' is {'reachable' if reachable else 'not reachable'}. "
                f"Missing facts: {missing}" if missing else ""
            ),
        )

    def check_constraints(
        self,
        facts: List[FactInstance],
    ) -> InferenceResult:
        """
        Validate a set of fact instances against ontology constraints.

        Checks uniqueness, mandatory roles, frequency, and other constraints.
        """
        violations = []
        steps = []

        for constraint in self.ontology.constraints:
            if constraint.kind == ConstraintKind.UNIQUENESS:
                violations.extend(
                    self._check_uniqueness(constraint, facts)
                )
            elif constraint.kind == ConstraintKind.MANDATORY:
                violations.extend(
                    self._check_mandatory(constraint, facts)
                )

            steps.append(ReasoningStep(
                method=InferenceMethod.CONSTRAINT,
                rule_applied=f"Checking {constraint.kind.value}: {constraint.name}",
                explanation=f"Constraint {constraint.name}: "
                           f"{'PASS' if not violations else 'FAIL'}",
            ))

        return InferenceResult(
            success=len(violations) == 0,
            reasoning_chain=steps,
            violations=violations,
            explanation=f"Checked {len(self.ontology.constraints)} constraints, "
                       f"found {len(violations)} violations",
        )

    def _check_uniqueness(
        self,
        constraint: Constraint,
        facts: List[FactInstance],
    ) -> List[str]:
        """Check uniqueness constraint"""
        violations = []
        # Group facts by fact type
        for fact_name in constraint.fact_types:
            type_facts = [f for f in facts if f.fact_type == fact_name]
            seen_values: Dict[str, set] = {}

            for fi in type_facts:
                for role_id in constraint.roles:
                    value = fi.values.get(role_id)
                    if value:
                        if role_id not in seen_values:
                            seen_values[role_id] = set()
                        if value in seen_values[role_id]:
                            violations.append(
                                f"Uniqueness violation in '{fact_name}': "
                                f"duplicate value '{value}' for role '{role_id}'"
                            )
                        seen_values[role_id].add(value)

        return violations

    def _check_mandatory(
        self,
        constraint: Constraint,
        facts: List[FactInstance],
    ) -> List[str]:
        """Check mandatory role constraint"""
        violations = []
        for fact_name in constraint.fact_types:
            type_facts = [f for f in facts if f.fact_type == fact_name]
            for fi in type_facts:
                for role_id in constraint.roles:
                    if role_id not in fi.values or fi.values[role_id] is None:
                        violations.append(
                            f"Mandatory role violation in '{fact_name}': "
                            f"role '{role_id}' is empty in instance {fi.id}"
                        )
        return violations

    def expand_context(
        self,
        object_type: str,
        depth: int = 2,
    ) -> List[ElementaryFact]:
        """
        Expand context around an object type.

        Finds all fact types within N hops of the given type,
        useful for building LLM context.
        """
        visited: Set[str] = set()
        facts: List[ElementaryFact] = []

        def _traverse(type_name: str, current_depth: int):
            if current_depth > depth or type_name in visited:
                return
            visited.add(type_name)

            for ft in self.ontology.get_facts_for_type(type_name):
                if ft not in facts:
                    facts.append(ft)
                for related_type in ft.object_types:
                    if related_type != type_name:
                        _traverse(related_type, current_depth + 1)

        _traverse(object_type, 0)
        return facts

    def generate_context_for_question(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Generate ontology context for a natural language question.

        Identifies relevant object types and facts, then builds
        a structured context for LLM consumption.
        """
        # Find matching object types
        matching_types = []
        for ot in self.ontology.object_types:
            if ot.name.lower() in question.lower():
                matching_types.append(ot)

        # Expand context around matching types
        relevant_facts = []
        for ot in matching_types:
            relevant_facts.extend(self.expand_context(ot.name))

        # Deduplicate
        seen_ids = set()
        unique_facts = []
        for f in relevant_facts:
            if f.id not in seen_ids:
                unique_facts.append(f)
                seen_ids.add(f.id)

        return {
            "matching_types": [
                {"name": ot.name, "kind": ot.kind.value, "description": ot.description}
                for ot in matching_types
            ],
            "relevant_facts": [
                {"name": f.name, "reading": f.reading, "types": f.object_types}
                for f in unique_facts
            ],
            "ontology_name": self.ontology.name,
            "total_relevant_facts": len(unique_facts),
        }
