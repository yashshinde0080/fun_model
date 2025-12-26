"""
Orchestrator Module
"""

import logging
import uuid
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

from orchestrator.config import get_orchestration_config, get_agent_config
from orchestrator.llm_client import OpenRouterClient as LLMClient, LLMError
from orchestrator.agents import (
    CEOAgent, PMAgent, ResearchAgent, CoderAgent, QAAgent, DocsAgent
)
from orchestrator.tools.sandbox_runner import SandboxRunner

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_CLARIFICATION = "needs_clarification"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowError(Exception):
    """Custom exception for workflow errors."""
    pass


@dataclass
class WorkflowContext:
    workflow_id: str
    user_id: str
    user_request: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    iteration: int = 0
    clarified_intent: str = ""
    project_spec: Optional[Dict[str, Any]] = None
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    codebase: Dict[str, str] = field(default_factory=dict) # path -> content
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


class Orchestrator:
    """
    Main Orchestrator for the Multi-Agent System.
    """

    def __init__(self, supabase_auth=None, llm_client=None, notifier=None):
        self.config = get_orchestration_config()
        self.supabase_auth = supabase_auth
        self.llm_client = llm_client or LLMClient()
        self.notifier = notifier
        self.sandbox = SandboxRunner(use_docker=False) # Fallback to local execution if Docker missing

        # Initialize Agents
        self.agents = {
            'ceo': CEOAgent(self.llm_client),
            'pm': PMAgent(self.llm_client),
            'research': ResearchAgent(self.llm_client),
            'coder': CoderAgent(self.llm_client),
            'qa': QAAgent(self.llm_client),
            'docs': DocsAgent(self.llm_client)
        }

    def execute(
        self,
        workflow_id: str,
        user_id: str,
        user_request: str,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a complete workflow following the Agile strict process."""
        context = WorkflowContext(
            workflow_id=workflow_id,
            user_id=user_id,
            user_request=user_request
        )

        try:
            context.status = WorkflowStatus.RUNNING
            self._log_event(context, 'workflow', 'started', {'request': user_request})
            # --- LANGGRAPH EXECUTION ---
            from orchestrator.graph import build_workflow_graph
            # Assuming NB_ITERATIONS is defined elsewhere or should be a config value
            NB_ITERATIONS = self.config.get('max_iterations_per_workflow', 5)

            logger.info(f"[{workflow_id}] Starting LangGraph Workflow")
            
            # Define logging callback for graph nodes
            def event_callback(agent_name: str, event_type: str, payload: Dict[str, Any]):
                self._log_event(context, agent_name, event_type, payload)

            # Initialize Graph
            app = build_workflow_graph(event_callback=event_callback)
            
            # Initial State
            initial_state = {
                "user_request": user_request,
                "user_id": user_id,
                "workspace_id": workflow_id,
                "clarifications": [],
                "intent": "",
                "specification": {},
                "project_plan": {},
                "backlog": [],
                "codebase": {},
                "qa_feedback": [],
                "qa_status": "PENDING",
                "iteration_count": 0,
                "max_iterations": NB_ITERATIONS,
                "documentation": {},
                "final_decision": "PENDING",
                "messages": []
            }
            
            # Execute
            final_state = app.invoke(initial_state)
            
            # Unpack results into context for backward compatibility w/ UI
            context.clarified_intent = final_state.get('intent', '')
            context.project_spec = final_state.get('specification', {})
            context.codebase = final_state.get('codebase', {})
            
            # Map LangGraph state to agent_outputs for compat
            context.agent_outputs['docs'] = {'payload': {'readme': final_state.get('documentation', {}).get('README.md', '')}}
            context.agent_outputs['final_signoff'] = {'payload': {'final_summary': 'Workflow Completed Successfully.'}}
            
            # Handle stops
            if final_state.get('clarifications'):
                 # Ambiguous
                 context.status = WorkflowStatus.NEEDS_CLARIFICATION
                 self._log_event(context, 'system', 'clarification_needed', {'questions': final_state['clarifications']})
                 return {
                    'success': False, 
                    'error': 'Clarification Needed',
                    'questions': final_state['clarifications']
                 }
                 
            if final_state.get('qa_status') == 'FAIL':
                raise WorkflowError("QA Logic failed to converge.")

            context.status = WorkflowStatus.COMPLETED
            result = self._compile_result(context, time.time() - context.start_time)
            
            return {
                'success': True,
                'workflow_id': workflow_id,
                'data': result,
                'duration': time.time() - context.start_time
            }

        except Exception as e:
            logger.exception(f"[{workflow_id}] Workflow error: {e}")
            context.status = WorkflowStatus.FAILED

            return {
                'success': False,
                'workflow_id': workflow_id,
                'error': str(e),
                'duration': time.time() - context.start_time
            }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _compile_result(self, context: WorkflowContext, elapsed: float) -> Dict[str, Any]:
        """Compile final workflow result with strict /final structure."""
        final_signoff = context.agent_outputs.get('final_signoff', {}).get('payload', {})
        
        # Build strict structure
        final_pkg = {
            'src': [],
            'tests': [],
            'root': []
        }
        
        files_flat = []
        
        for path, content in context.codebase.items():
            file_obj = {'path': path, 'content': content}
            files_flat.append(file_obj)
            
            if 'test' in path or 'tests/' in path:
                final_pkg['tests'].append(file_obj)
            elif '/' in path and 'src' in path: 
                final_pkg['src'].append(file_obj)
            else:
                final_pkg['root'].append(file_obj)
                
        # Ensure README and requirements are in root
        readme = context.agent_outputs.get('docs', {}).get('payload', {}).get('readme')
        if readme:
            final_pkg['root'].append({'path': 'README.md', 'content': readme})
            
        return {
            'success': True,
            'workflow_id': context.workflow_id,
            'duration': elapsed,
            'project_spec': context.project_spec,
            'final_output': { # Use this key for the contract
                'final': final_pkg
            },
            'deliverables': {'files': files_flat},
            'final_summary': final_signoff.get('final_summary', ''),
            'events': context.events
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _execute_agent_task(
        self,
        context: WorkflowContext,
        agent_name: str,
        task_id: str,
        phase: str,
        inputs: Dict[str, Any],
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """Execute a single agent task."""
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Unknown agent: {agent_name}")

        try:
            self._log_event(context, agent_name, 'task_started', {'task_id': task_id, 'phase': phase})

            result = agent.execute(task_id=task_id, phase=phase, inputs=inputs, timeout=timeout)

            self._log_event(context, agent_name, 'task_completed', {
                'task_id': task_id,
                'status': result.get('status'),
                'output': result.get('payload')
            })

            return result
        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {e}")
            self._log_event(context, agent_name, 'task_failed', {'error': str(e)})
            return {'status': 'failed', 'payload': {'error': str(e)}}

    def _log_event(self, context: WorkflowContext, agent_name: str, event_type: str, data: Dict[str, Any]):
        """Log an event."""
        event = {
            'workflow_id': context.workflow_id,
            'agent_name': agent_name,
            'event_type': event_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'payload': data
        }
        
        # Log to terminal/file as well
        logger.info(f"[{agent_name}] {event_type}")
        
        context.events.append(event)
        try:
            self.supabase_auth.log_event(event)
        except Exception:
            pass # Non-critical