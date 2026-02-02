"""
Base Agent Architecture with ReAct + Reflection Patterns
Implements best-in-class agentic design patterns for 2026
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class ThoughtType(str, Enum):
    """Types of agent thoughts in ReAct pattern"""
    OBSERVATION = "observation"
    REASONING = "reasoning"
    PLANNING = "planning"
    REFLECTION = "reflection"
    DECISION = "decision"


@dataclass
class Thought:
    """Represents a single thought in the agent's reasoning chain"""
    id: UUID = field(default_factory=uuid4)
    type: ThoughtType = ThoughtType.REASONING
    content: str = ""
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Represents an action the agent can take"""
    id: UUID = field(default_factory=uuid4)
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    thought: Optional[Thought] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class UserContext(BaseModel):
    """User identity and permissions context - propagated through entire pipeline"""
    user_id: str
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    roles: List[str] = Field(default_factory=list)
    permissions: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions or "admin" in self.roles

    def get_sql_filters(self) -> Dict[str, Any]:
        """Get row-level security filters for this user"""
        return self.permissions.get("sql_filters", {})


class AgentConfig(BaseModel):
    """Configuration for an agent instance"""
    name: str = "agent"
    description: str = ""
    max_iterations: int = 10
    max_reflection_depth: int = 3
    thinking_budget: int = 5000  # tokens for reasoning
    temperature: float = 0.7
    enable_reflection: bool = True
    enable_planning: bool = True
    enable_memory: bool = True
    timeout_seconds: float = 300.0
    retry_attempts: int = 3

    class Config:
        extra = "allow"


class AgentContext(BaseModel):
    """Runtime context for agent execution"""
    agent_id: UUID = Field(default_factory=uuid4)
    user: Optional[UserContext] = None
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_context: Optional[AgentContext] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    thoughts: List[Thought] = Field(default_factory=list)
    actions: List[Action] = Field(default_factory=list)
    state: AgentState = AgentState.IDLE
    iteration: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True

    def add_thought(self, thought: Thought) -> None:
        self.thoughts.append(thought)

    def add_action(self, action: Action) -> None:
        self.actions.append(action)

    def get_reasoning_chain(self) -> str:
        """Get formatted reasoning chain for reflection"""
        chain = []
        for thought in self.thoughts:
            chain.append(f"[{thought.type.value}] {thought.content}")
        return "\n".join(chain)


T = TypeVar("T")


class AgentResult(BaseModel, Generic[T]):
    """Result from agent execution"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    context: Optional[AgentContext] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class BaseAgent(ABC):
    """
    Base Agent implementing ReAct + Reflection patterns

    Key patterns implemented:
    1. ReAct (Reason + Act): Alternates between thinking and acting
    2. Reflection: Self-evaluates and corrects outputs
    3. Planning: Breaks complex tasks into steps
    4. Tool Use: Extends capabilities via tools
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[Dict[str, Callable]] = None,
        memory: Optional[Any] = None,  # MemoryManager type
    ):
        self.config = config
        self.tools = tools or {}
        self.memory = memory
        self._hooks: Dict[str, List[Callable]] = {
            "pre_think": [],
            "post_think": [],
            "pre_act": [],
            "post_act": [],
            "pre_reflect": [],
            "post_reflect": [],
        }

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register lifecycle hook"""
        if event in self._hooks:
            self._hooks[event].append(callback)

    async def _run_hooks(self, event: str, context: AgentContext, **kwargs) -> None:
        """Execute registered hooks"""
        for hook in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(context, **kwargs)
                else:
                    hook(context, **kwargs)
            except Exception as e:
                logger.warning(f"Hook {event} failed: {e}")

    @abstractmethod
    async def think(self, context: AgentContext, input_data: Any) -> Thought:
        """
        ReAct THINK phase: Reason about the current state and decide next action
        """
        pass

    @abstractmethod
    async def act(self, context: AgentContext, thought: Thought) -> Action:
        """
        ReAct ACT phase: Execute the decided action using tools
        """
        pass

    @abstractmethod
    async def reflect(self, context: AgentContext) -> Thought:
        """
        REFLECTION phase: Self-evaluate the reasoning chain and outputs
        Returns a reflection thought with critique and suggestions
        """
        pass

    async def plan(self, context: AgentContext, goal: str) -> List[Thought]:
        """
        PLANNING phase: Break down complex goals into actionable steps
        Override for custom planning logic
        """
        planning_thought = Thought(
            type=ThoughtType.PLANNING,
            content=f"Planning approach for: {goal}",
        )
        context.add_thought(planning_thought)
        return [planning_thought]

    async def execute(
        self,
        input_data: Any,
        user_context: Optional[UserContext] = None,
    ) -> AgentResult:
        """
        Main execution loop implementing ReAct + Reflection pattern

        Loop:
        1. THINK: Reason about current state
        2. ACT: Execute decided action
        3. OBSERVE: Record action result
        4. REFLECT: Self-evaluate (if enabled)
        5. Repeat or terminate
        """
        context = AgentContext(
            user=user_context,
            start_time=datetime.utcnow(),
        )

        try:
            context.state = AgentState.THINKING

            # Optional planning phase for complex tasks
            if self.config.enable_planning:
                await self.plan(context, str(input_data))

            # Main ReAct loop
            while context.iteration < self.config.max_iterations:
                context.iteration += 1

                # THINK phase
                await self._run_hooks("pre_think", context)
                context.state = AgentState.THINKING
                thought = await self.think(context, input_data)
                context.add_thought(thought)
                await self._run_hooks("post_think", context, thought=thought)

                # Check for completion
                if thought.type == ThoughtType.DECISION and "DONE" in thought.content:
                    break

                # ACT phase
                await self._run_hooks("pre_act", context)
                context.state = AgentState.ACTING
                action = await self.act(context, thought)
                context.add_action(action)
                await self._run_hooks("post_act", context, action=action)

                # OBSERVE - record result as thought
                observation = Thought(
                    type=ThoughtType.OBSERVATION,
                    content=str(action.result) if action.result else str(action.error),
                    metadata={"action_id": str(action.id)},
                )
                context.add_thought(observation)

                # REFLECT phase (optional)
                if self.config.enable_reflection and context.iteration % 2 == 0:
                    await self._run_hooks("pre_reflect", context)
                    context.state = AgentState.REFLECTING
                    reflection = await self.reflect(context)
                    context.add_thought(reflection)
                    await self._run_hooks("post_reflect", context, reflection=reflection)

                    # Handle reflection feedback
                    if reflection.confidence < 0.5:
                        # Low confidence - need to reconsider
                        input_data = f"[REFLECTION]: {reflection.content}\n[ORIGINAL]: {input_data}"

            context.state = AgentState.COMPLETED
            context.end_time = datetime.utcnow()

            # Store in memory if enabled
            if self.config.enable_memory and self.memory:
                await self.memory.store_interaction(context)

            return AgentResult(
                success=True,
                data=self._extract_result(context),
                context=context,
            )

        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            context.state = AgentState.FAILED
            context.end_time = datetime.utcnow()
            return AgentResult(
                success=False,
                error=str(e),
                context=context,
            )

    def _extract_result(self, context: AgentContext) -> Any:
        """Extract final result from context - override for custom extraction"""
        if context.actions:
            last_action = context.actions[-1]
            return last_action.result
        return None


class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents for complex workflows
    Implements the Multi-Agent Collaboration pattern
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, List[str]] = {}

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register a specialized agent"""
        self.agents[name] = agent

    def define_workflow(self, name: str, agent_sequence: List[str]) -> None:
        """Define a workflow as a sequence of agents"""
        self.workflows[name] = agent_sequence

    async def execute_workflow(
        self,
        workflow_name: str,
        input_data: Any,
        user_context: Optional[UserContext] = None,
    ) -> AgentResult:
        """Execute a defined workflow"""
        if workflow_name not in self.workflows:
            return AgentResult(success=False, error=f"Unknown workflow: {workflow_name}")

        current_data = input_data
        contexts = []

        for agent_name in self.workflows[workflow_name]:
            if agent_name not in self.agents:
                return AgentResult(success=False, error=f"Unknown agent: {agent_name}")

            agent = self.agents[agent_name]
            result = await agent.execute(current_data, user_context)

            if not result.success:
                return result

            current_data = result.data
            if result.context:
                contexts.append(result.context)

        return AgentResult(
            success=True,
            data=current_data,
            metadata={"workflow": workflow_name, "agent_count": len(contexts)},
        )
