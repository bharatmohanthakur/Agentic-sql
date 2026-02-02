"""
Deep Reasoner - Advanced multi-step reasoning for complex queries
Implements Chain-of-Thought, Tree-of-Thought, and ReAct patterns
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class ReasoningStrategy(str, Enum):
    """Reasoning strategies"""
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Linear step-by-step
    TREE_OF_THOUGHT = "tree_of_thought"  # Explore multiple paths
    REACT = "react"  # Reason-Act-Observe loop
    DECOMPOSITION = "decomposition"  # Break into sub-problems
    ANALOGY = "analogy"  # Use similar solved problems
    VERIFICATION = "verification"  # Self-verify each step


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain"""
    id: str = field(default_factory=lambda: str(uuid4()))
    step_number: int = 0
    type: str = "thought"  # thought, action, observation, conclusion

    content: str = ""
    confidence: float = 1.0

    # For tree-of-thought
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    score: float = 0.0  # Evaluation score

    # Metadata
    tokens_used: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReasoningChain:
    """Complete reasoning chain"""
    id: str = field(default_factory=lambda: str(uuid4()))
    question: str = ""
    strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT

    steps: List[ReasoningStep] = field(default_factory=list)
    conclusion: Optional[str] = None
    sql_result: Optional[str] = None

    # Quality metrics
    total_confidence: float = 0.0
    verification_passed: bool = False
    alternative_paths: int = 0

    # Performance
    total_tokens: int = 0
    reasoning_time_ms: float = 0.0

    def add_step(self, step: ReasoningStep) -> None:
        step.step_number = len(self.steps) + 1
        self.steps.append(step)

    def get_chain_text(self) -> str:
        """Get formatted reasoning chain"""
        lines = []
        for step in self.steps:
            lines.append(f"[Step {step.step_number}] ({step.type}): {step.content}")
        return "\n".join(lines)


class DeepReasoner:
    """
    Deep reasoning engine for complex query understanding

    Features:
    - Chain-of-Thought: Step-by-step reasoning
    - Tree-of-Thought: Explore multiple solution paths
    - Self-verification: Validate reasoning steps
    - Knowledge integration: Use learned patterns
    - Confidence scoring: Track reasoning quality
    """

    def __init__(
        self,
        llm_client: Any,
        knowledge_base: Optional[Any] = None,
        max_depth: int = 10,
        beam_width: int = 3,  # For tree-of-thought
        confidence_threshold: float = 0.7,
    ):
        self.llm = llm_client
        self.knowledge = knowledge_base
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.min_confidence = confidence_threshold

    async def reason(
        self,
        question: str,
        schema_context: str,
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
        examples: Optional[List[Dict]] = None,
    ) -> ReasoningChain:
        """
        Perform deep reasoning on a question

        Returns complete reasoning chain with SQL conclusion
        """
        chain = ReasoningChain(
            question=question,
            strategy=strategy,
        )

        start_time = datetime.utcnow()

        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            await self._chain_of_thought(chain, schema_context, examples)
        elif strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            await self._tree_of_thought(chain, schema_context, examples)
        elif strategy == ReasoningStrategy.DECOMPOSITION:
            await self._decomposition(chain, schema_context, examples)
        elif strategy == ReasoningStrategy.REACT:
            await self._react_loop(chain, schema_context, examples)

        # Verification pass
        if chain.sql_result:
            chain.verification_passed = await self._verify_reasoning(chain)

        # Calculate metrics
        chain.reasoning_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        chain.total_confidence = self._calculate_confidence(chain)

        return chain

    async def _chain_of_thought(
        self,
        chain: ReasoningChain,
        schema_context: str,
        examples: Optional[List[Dict]],
    ) -> None:
        """Linear chain-of-thought reasoning"""
        prompt = self._build_cot_prompt(chain.question, schema_context, examples)

        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=2000,
        )

        # Parse steps from response
        steps = self._parse_cot_response(response)

        for step_content in steps:
            step = ReasoningStep(
                type="thought" if "STEP" in step_content.upper() else "conclusion",
                content=step_content,
            )
            chain.add_step(step)

        # Extract SQL from conclusion
        chain.sql_result = self._extract_sql(response)
        chain.conclusion = steps[-1] if steps else ""

    async def _tree_of_thought(
        self,
        chain: ReasoningChain,
        schema_context: str,
        examples: Optional[List[Dict]],
    ) -> None:
        """Tree-of-thought: explore multiple paths"""
        # Generate initial thoughts
        initial_thoughts = await self._generate_thoughts(
            chain.question,
            schema_context,
            n=self.beam_width,
        )

        # Track best path
        best_path: List[ReasoningStep] = []
        best_score = 0.0

        for thought in initial_thoughts:
            root = ReasoningStep(type="thought", content=thought)
            chain.add_step(root)

            # Expand this path
            path, score = await self._expand_path(
                root,
                chain.question,
                schema_context,
                depth=0,
            )

            if score > best_score:
                best_score = score
                best_path = path

        # Use best path
        chain.steps = best_path
        chain.alternative_paths = len(initial_thoughts)

        # Generate SQL from best path
        chain.sql_result = await self._generate_sql_from_path(
            chain.question,
            best_path,
            schema_context,
        )

    async def _decomposition(
        self,
        chain: ReasoningChain,
        schema_context: str,
        examples: Optional[List[Dict]],
    ) -> None:
        """Decompose complex question into sub-problems"""
        # Step 1: Decompose
        decompose_prompt = f"""
        Break down this complex data question into simpler sub-questions:

        Question: {chain.question}

        Schema:
        {schema_context}

        List 2-5 simpler questions that, when answered, will answer the main question:
        """

        response = await self.llm.generate(prompt=decompose_prompt, max_tokens=500)

        sub_questions = self._parse_numbered_list(response)

        chain.add_step(ReasoningStep(
            type="decomposition",
            content=f"Sub-questions: {sub_questions}",
        ))

        # Step 2: Solve each sub-question
        sub_results = []
        for i, sub_q in enumerate(sub_questions):
            sub_prompt = f"""
            Answer this sub-question to help solve the main question.

            Main question: {chain.question}
            Sub-question {i+1}: {sub_q}

            Schema:
            {schema_context}

            Provide the SQL for this sub-question:
            """

            sub_response = await self.llm.generate(prompt=sub_prompt, max_tokens=500)
            sub_sql = self._extract_sql(sub_response)

            chain.add_step(ReasoningStep(
                type="sub-solution",
                content=f"Sub-Q{i+1}: {sub_q}\nSQL: {sub_sql}",
            ))

            sub_results.append((sub_q, sub_sql))

        # Step 3: Combine into final solution
        combine_prompt = f"""
        Combine these sub-solutions into a final SQL query:

        Main question: {chain.question}

        Sub-solutions:
        {chr(10).join(f"{i+1}. {q}: {sql}" for i, (q, sql) in enumerate(sub_results))}

        Provide the final combined SQL:
        """

        final_response = await self.llm.generate(prompt=combine_prompt, max_tokens=500)
        chain.sql_result = self._extract_sql(final_response)

        chain.add_step(ReasoningStep(
            type="conclusion",
            content=f"Final SQL: {chain.sql_result}",
        ))

    async def _react_loop(
        self,
        chain: ReasoningChain,
        schema_context: str,
        examples: Optional[List[Dict]],
    ) -> None:
        """ReAct: Reason-Act-Observe loop"""
        context = f"Question: {chain.question}\n\nSchema:\n{schema_context}"
        iteration = 0

        while iteration < self.max_depth:
            iteration += 1

            # REASON
            reason_prompt = f"""
            {context}

            Previous steps:
            {chain.get_chain_text()}

            Think about what you need to do next to answer the question.
            If you have enough information, generate the final SQL.

            Thought:
            """

            thought = await self.llm.generate(prompt=reason_prompt, max_tokens=300)

            chain.add_step(ReasoningStep(
                type="thought",
                content=thought.strip(),
            ))

            # Check if done
            if "FINAL SQL" in thought.upper() or "ANSWER" in thought.upper():
                chain.sql_result = self._extract_sql(thought)
                break

            # ACT
            act_prompt = f"""
            Based on your thought: {thought}

            What action should you take?
            Available actions:
            - LOOKUP: Find relevant table/column
            - ANALYZE: Understand data patterns
            - GENERATE: Create SQL query
            - VERIFY: Check logic

            Action:
            """

            action = await self.llm.generate(prompt=act_prompt, max_tokens=100)

            chain.add_step(ReasoningStep(
                type="action",
                content=action.strip(),
            ))

            # OBSERVE (simulated)
            observe_prompt = f"""
            Action taken: {action}

            Observation/Result:
            """

            observation = await self.llm.generate(prompt=observe_prompt, max_tokens=200)

            chain.add_step(ReasoningStep(
                type="observation",
                content=observation.strip(),
            ))

        # Final SQL generation if not done
        if not chain.sql_result:
            final_prompt = f"""
            Based on this reasoning:
            {chain.get_chain_text()}

            Generate the final SQL query for: {chain.question}
            """

            final = await self.llm.generate(prompt=final_prompt, max_tokens=500)
            chain.sql_result = self._extract_sql(final)

    async def _generate_thoughts(
        self,
        question: str,
        schema_context: str,
        n: int = 3,
    ) -> List[str]:
        """Generate multiple initial thoughts"""
        prompt = f"""
        Generate {n} different approaches to solve this question:

        Question: {question}

        Schema:
        {schema_context}

        List {n} different solution approaches:
        """

        response = await self.llm.generate(prompt=prompt, max_tokens=500)
        return self._parse_numbered_list(response)[:n]

    async def _expand_path(
        self,
        node: ReasoningStep,
        question: str,
        schema_context: str,
        depth: int,
    ) -> Tuple[List[ReasoningStep], float]:
        """Expand a reasoning path"""
        if depth >= self.max_depth:
            score = await self._evaluate_path([node])
            return [node], score

        # Generate next steps
        prompt = f"""
        Continue this reasoning:

        Question: {question}
        Current thought: {node.content}

        What is the next logical step?
        """

        response = await self.llm.generate(prompt=prompt, max_tokens=200)

        next_step = ReasoningStep(
            type="thought",
            content=response.strip(),
            parent_id=node.id,
        )

        # Recursively expand
        path, score = await self._expand_path(
            next_step,
            question,
            schema_context,
            depth + 1,
        )

        return [node] + path, score

    async def _evaluate_path(self, path: List[ReasoningStep]) -> float:
        """Evaluate quality of a reasoning path"""
        if not path:
            return 0.0

        prompt = f"""
        Rate the quality of this reasoning (0-10):

        {chr(10).join(s.content for s in path)}

        Score (just the number):
        """

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=10)
            score = float(response.strip().split()[0])
            return min(10, max(0, score)) / 10
        except:
            return 0.5

    async def _verify_reasoning(self, chain: ReasoningChain) -> bool:
        """Verify the reasoning chain and SQL"""
        prompt = f"""
        Verify this reasoning and SQL:

        Question: {chain.question}

        Reasoning:
        {chain.get_chain_text()}

        SQL: {chain.sql_result}

        Is the reasoning valid and does the SQL correctly answer the question?
        Answer YES or NO with explanation:
        """

        response = await self.llm.generate(prompt=prompt, max_tokens=200)
        return "YES" in response.upper()

    async def _generate_sql_from_path(
        self,
        question: str,
        path: List[ReasoningStep],
        schema_context: str,
    ) -> str:
        """Generate SQL from reasoning path"""
        prompt = f"""
        Based on this reasoning, generate SQL:

        Question: {question}

        Reasoning:
        {chr(10).join(s.content for s in path)}

        Schema:
        {schema_context}

        SQL:
        """

        response = await self.llm.generate(prompt=prompt, max_tokens=500)
        return self._extract_sql(response)

    def _build_cot_prompt(
        self,
        question: str,
        schema_context: str,
        examples: Optional[List[Dict]],
    ) -> str:
        """Build chain-of-thought prompt"""
        examples_text = ""
        if examples:
            examples_text = "\n\nExamples:\n"
            for ex in examples[:3]:
                examples_text += f"Q: {ex['question']}\nSQL: {ex['sql']}\n\n"

        return f"""
        Solve this step by step:

        Question: {question}

        Database Schema:
        {schema_context}
        {examples_text}

        Think through this step-by-step:
        STEP 1: Identify what data is needed
        STEP 2: Identify relevant tables
        STEP 3: Determine how to join tables
        STEP 4: Identify filters and conditions
        STEP 5: Determine aggregations if needed
        STEP 6: Write the final SQL

        Let's solve this:
        """

    def _parse_cot_response(self, response: str) -> List[str]:
        """Parse chain-of-thought response into steps"""
        import re
        steps = re.split(r'STEP \d+:', response)
        return [s.strip() for s in steps if s.strip()]

    def _parse_numbered_list(self, response: str) -> List[str]:
        """Parse numbered list from response"""
        import re
        items = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', response, re.DOTALL)
        return [item.strip() for item in items if item.strip()]

    def _extract_sql(self, response: str) -> str:
        """Extract SQL from response"""
        import re

        # Look for SQL in code blocks
        match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Look for SELECT statements
        match = re.search(r'(SELECT\s+.*?)(;|$)', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return ""

    def _calculate_confidence(self, chain: ReasoningChain) -> float:
        """Calculate overall confidence score"""
        if not chain.steps:
            return 0.0

        # Average step confidence
        avg_confidence = sum(s.confidence for s in chain.steps) / len(chain.steps)

        # Boost for verification
        if chain.verification_passed:
            avg_confidence = min(1.0, avg_confidence * 1.2)

        return avg_confidence


class KnowledgeSynthesis:
    """
    Synthesizes knowledge from multiple sources for better reasoning
    """

    def __init__(
        self,
        knowledge_base: Any,
        llm_client: Any,
    ):
        self.knowledge = knowledge_base
        self.llm = llm_client

    async def synthesize_context(
        self,
        question: str,
        schema_context: str,
    ) -> str:
        """
        Synthesize relevant knowledge for a question

        Combines:
        - Similar past queries
        - Domain knowledge
        - Schema understanding
        - Business rules
        """
        contexts = []

        # Get similar queries
        if self.knowledge:
            similar = await self.knowledge.search(question, limit=3)
            if similar:
                contexts.append("Similar queries:\n" + "\n".join(
                    f"- {s.content}" for s in similar
                ))

        # Get domain knowledge
        domain = await self._infer_domain(question)
        if domain:
            contexts.append(f"Domain: {domain}")

        # Combine with schema
        contexts.append(f"Schema:\n{schema_context}")

        return "\n\n".join(contexts)

    async def _infer_domain(self, question: str) -> str:
        """Infer the business domain of a question"""
        prompt = f"""
        What business domain does this question relate to?
        Question: {question}

        Domain (one word):
        """

        response = await self.llm.generate(prompt=prompt, max_tokens=20)
        return response.strip().lower()
