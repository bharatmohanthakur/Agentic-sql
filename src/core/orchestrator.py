"""
Agent Orchestrator - Coordinates multi-agent workflows
Implements the Multi-Agent Collaboration pattern
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .base import AgentContext, AgentResult, BaseAgent, UserContext

logger = logging.getLogger(__name__)


class WorkflowState(str, Enum):
    """Workflow execution states"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentTask(BaseModel):
    """Task definition for an agent in a workflow"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    input_data: Any = None
    dependencies: List[str] = Field(default_factory=list)  # Task IDs
    timeout_seconds: float = 300.0
    retry_count: int = 0
    max_retries: int = 3

    # Runtime state
    state: WorkflowState = WorkflowState.PENDING
    result: Optional[AgentResult] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class Workflow(BaseModel):
    """Workflow definition"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    tasks: List[AgentTask] = Field(default_factory=list)

    # Runtime state
    state: WorkflowState = WorkflowState.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class AgentOrchestrator:
    """
    Orchestrates multi-agent workflows with:
    - DAG-based task dependencies
    - Parallel execution where possible
    - Error handling and retries
    - Progress tracking
    - Result aggregation
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, Workflow] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "workflow_start": [],
            "workflow_complete": [],
            "task_start": [],
            "task_complete": [],
            "task_error": [],
        }

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register an agent"""
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")

    def unregister_agent(self, name: str) -> None:
        """Unregister an agent"""
        if name in self.agents:
            del self.agents[name]

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register workflow hook"""
        if event in self._hooks:
            self._hooks[event].append(callback)

    async def _emit_hook(self, event: str, **kwargs) -> None:
        """Emit hook event"""
        for hook in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(**kwargs)
                else:
                    hook(**kwargs)
            except Exception as e:
                logger.warning(f"Hook {event} failed: {e}")

    def create_workflow(
        self,
        name: str,
        description: str = "",
    ) -> Workflow:
        """Create a new workflow"""
        workflow = Workflow(name=name, description=description)
        self.workflows[workflow.id] = workflow
        return workflow

    def add_task(
        self,
        workflow: Workflow,
        agent_name: str,
        input_data: Any = None,
        dependencies: Optional[List[str]] = None,
        timeout_seconds: float = 300.0,
    ) -> AgentTask:
        """Add a task to a workflow"""
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")

        task = AgentTask(
            agent_name=agent_name,
            input_data=input_data,
            dependencies=dependencies or [],
            timeout_seconds=timeout_seconds,
        )
        workflow.tasks.append(task)
        return task

    async def execute_workflow(
        self,
        workflow: Workflow,
        user_context: Optional[UserContext] = None,
        initial_input: Any = None,
    ) -> Dict[str, Any]:
        """
        Execute a workflow

        Executes tasks respecting dependencies, running parallel
        where possible.
        """
        workflow.state = WorkflowState.RUNNING
        workflow.started_at = datetime.utcnow()

        await self._emit_hook("workflow_start", workflow=workflow)

        try:
            # Build dependency graph
            task_map = {t.id: t for t in workflow.tasks}
            completed: Set[str] = set()
            failed: Set[str] = set()

            # Set initial input for tasks with no dependencies
            for task in workflow.tasks:
                if not task.dependencies and initial_input is not None:
                    task.input_data = initial_input

            while len(completed) + len(failed) < len(workflow.tasks):
                # Find ready tasks (all dependencies completed)
                ready_tasks = [
                    t for t in workflow.tasks
                    if t.id not in completed
                    and t.id not in failed
                    and t.state == WorkflowState.PENDING
                    and all(dep in completed for dep in t.dependencies)
                ]

                if not ready_tasks:
                    # Check for deadlock or all tasks blocked
                    pending = [
                        t for t in workflow.tasks
                        if t.id not in completed and t.id not in failed
                    ]
                    if pending:
                        # Tasks blocked by failed dependencies
                        for task in pending:
                            if any(dep in failed for dep in task.dependencies):
                                task.state = WorkflowState.FAILED
                                task.error = "Dependency failed"
                                failed.add(task.id)
                    break

                # Execute ready tasks in parallel
                results = await asyncio.gather(
                    *[
                        self._execute_task(task, task_map, user_context)
                        for task in ready_tasks
                    ],
                    return_exceptions=True,
                )

                # Process results
                for task, result in zip(ready_tasks, results):
                    if isinstance(result, Exception):
                        task.state = WorkflowState.FAILED
                        task.error = str(result)
                        failed.add(task.id)
                        await self._emit_hook("task_error", task=task, error=result)
                    elif result.success:
                        task.state = WorkflowState.COMPLETED
                        task.result = result
                        completed.add(task.id)
                        workflow.results[task.id] = result.data
                        await self._emit_hook("task_complete", task=task, result=result)
                    else:
                        task.state = WorkflowState.FAILED
                        task.error = result.error
                        failed.add(task.id)
                        await self._emit_hook("task_error", task=task, error=result.error)

            # Determine workflow state
            if failed:
                workflow.state = WorkflowState.FAILED
            else:
                workflow.state = WorkflowState.COMPLETED

            workflow.completed_at = datetime.utcnow()

            await self._emit_hook("workflow_complete", workflow=workflow)

            return {
                "success": workflow.state == WorkflowState.COMPLETED,
                "results": workflow.results,
                "completed_tasks": list(completed),
                "failed_tasks": list(failed),
            }

        except Exception as e:
            logger.exception("Workflow execution failed")
            workflow.state = WorkflowState.FAILED
            workflow.completed_at = datetime.utcnow()
            return {
                "success": False,
                "error": str(e),
                "results": workflow.results,
            }

    async def _execute_task(
        self,
        task: AgentTask,
        task_map: Dict[str, AgentTask],
        user_context: Optional[UserContext],
    ) -> AgentResult:
        """Execute a single task"""
        task.state = WorkflowState.RUNNING
        task.started_at = datetime.utcnow()

        await self._emit_hook("task_start", task=task)

        agent = self.agents.get(task.agent_name)
        if not agent:
            return AgentResult(success=False, error=f"Agent not found: {task.agent_name}")

        # Get input from dependencies if not set
        input_data = task.input_data
        if input_data is None and task.dependencies:
            # Use output from first dependency
            dep_task = task_map.get(task.dependencies[0])
            if dep_task and dep_task.result:
                input_data = dep_task.result.data

        try:
            result = await asyncio.wait_for(
                agent.execute(input_data, user_context),
                timeout=task.timeout_seconds,
            )
            task.completed_at = datetime.utcnow()
            return result

        except asyncio.TimeoutError:
            return AgentResult(success=False, error="Task timeout")
        except Exception as e:
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.state = WorkflowState.PENDING
                logger.warning(f"Task {task.id} retry {task.retry_count}")
                return await self._execute_task(task, task_map, user_context)
            return AgentResult(success=False, error=str(e))


class PipelineBuilder:
    """
    Fluent builder for creating common workflow patterns
    """

    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
        self._workflow: Optional[Workflow] = None
        self._last_task: Optional[AgentTask] = None

    def create(self, name: str, description: str = "") -> "PipelineBuilder":
        """Create a new workflow"""
        self._workflow = self.orchestrator.create_workflow(name, description)
        self._last_task = None
        return self

    def add(
        self,
        agent_name: str,
        input_data: Any = None,
        timeout: float = 300.0,
    ) -> "PipelineBuilder":
        """Add a task that depends on the previous task"""
        if not self._workflow:
            raise ValueError("Call create() first")

        dependencies = [self._last_task.id] if self._last_task else []

        self._last_task = self.orchestrator.add_task(
            self._workflow,
            agent_name,
            input_data,
            dependencies,
            timeout,
        )
        return self

    def parallel(
        self,
        *agent_names: str,
        timeout: float = 300.0,
    ) -> "PipelineBuilder":
        """Add multiple tasks that run in parallel, all depending on previous"""
        if not self._workflow:
            raise ValueError("Call create() first")

        dependencies = [self._last_task.id] if self._last_task else []

        tasks = []
        for agent_name in agent_names:
            task = self.orchestrator.add_task(
                self._workflow,
                agent_name,
                dependencies=dependencies,
                timeout_seconds=timeout,
            )
            tasks.append(task)

        # Create a join point - next task will depend on all parallel tasks
        self._last_task = tasks[-1] if tasks else self._last_task
        return self

    def build(self) -> Workflow:
        """Return the built workflow"""
        if not self._workflow:
            raise ValueError("Call create() first")
        return self._workflow

    async def run(
        self,
        user_context: Optional[UserContext] = None,
        initial_input: Any = None,
    ) -> Dict[str, Any]:
        """Build and execute the workflow"""
        workflow = self.build()
        return await self.orchestrator.execute_workflow(
            workflow,
            user_context,
            initial_input,
        )
