"""
Main Orchestrator - Workflow Coordination Engine
"""

import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from orchestrator.config import get_orchestration_config, get_agent_config
from orchestrator.llm_client import OpenRouterClient, LLMError

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    id: str
    name: str
    assigned_to: str
    description: str
    acceptance_criteria: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    retry_count: int = 0


@dataclass
class WorkflowContext:
    workflow_id: str
    user_id: str
    user_request: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    iteration: int = 0
    project_spec: Optional[Dict[str, Any]] = None
    tasks: List[Task] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


class WorkflowError(Exception):
    """Exception raised for workflow-level errors."""
    pass


class Orchestrator:
    """Main orchestrator that coordinates multi-agent workflows."""
    
    def __init__(self, llm_client: OpenRouterClient, supabase_auth, notifier):
        self.llm_client = llm_client
        self.supabase_auth = supabase_auth
        self.notifier = notifier
        self.config = get_orchestration_config()
        
        # Initialize agents
        from orchestrator.agents import (
            CEOAgent, PMAgent, ResearchAgent, 
            CoderAgent, QAAgent, DocsAgent
        )
        
        self.agents = {
            'ceo': CEOAgent(llm_client),
            'pm': PMAgent(llm_client),
            'research': ResearchAgent(llm_client),
            'coder': CoderAgent(llm_client),
            'qa': QAAgent(llm_client),
            'docs': DocsAgent(llm_client)
        }
        
        logger.info("Orchestrator initialized")
    
    def execute(
        self,
        workflow_id: str,
        user_id: str,
        user_request: str,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a complete workflow."""
        context = WorkflowContext(
            workflow_id=workflow_id,
            user_id=user_id,
            user_request=user_request
        )
        
        try:
            context.status = WorkflowStatus.RUNNING
            self._log_event(context, 'workflow', 'started', {'request': user_request})
            
            # Phase 1: CEO Specification
            logger.info(f"[{workflow_id}] Phase 1: CEO Specification")
            ceo_result = self._execute_agent_task(
                context, 'ceo', 'ceo-spec', 'specification',
                {'user_request': user_request}
            )
            
            if ceo_result['status'] != 'success':
                raise WorkflowError("CEO failed to create specification")
            
            context.project_spec = ceo_result['payload'].get('project_spec', {})
            context.agent_outputs['ceo_spec'] = ceo_result
            
            # Phase 2: PM Planning
            logger.info(f"[{workflow_id}] Phase 2: PM Planning")
            pm_result = self._execute_agent_task(
                context, 'pm', 'pm-planning', 'planning',
                {
                    'project_spec': context.project_spec,
                    'objectives': ceo_result['payload'].get('objectives', []),
                    'constraints': ceo_result['payload'].get('constraints', [])
                }
            )
            
            if pm_result['status'] != 'success':
                raise WorkflowError("PM failed to create plan")
            
            context.agent_outputs['pm'] = pm_result
            
            # Phase 3: Execute tasks
            logger.info(f"[{workflow_id}] Phase 3: Task Execution")
            execution_order = pm_result['payload'].get('execution_order', [])
            tasks = pm_result['payload'].get('tasks', [])
            
            for task_data in tasks:
                task_id = task_data.get('id', str(uuid.uuid4()))
                agent_name = task_data.get('assigned_to', 'coder')
                
                context.iteration += 1
                if context.iteration > self.config.get('max_iterations_per_workflow', 6):
                    break
                
                result = self._execute_agent_task(
                    context, agent_name, task_id, 'execution',
                    {
                        'task_description': task_data.get('description', ''),
                        'acceptance_criteria': task_data.get('acceptance_criteria', []),
                        'project_spec': context.project_spec,
                        'project_context': context.project_spec
                    }
                )
                context.agent_outputs[f"{agent_name}_{task_id}"] = result
            
            # Phase 4: QA Validation
            if self.config.get('require_qa_approval', True):
                logger.info(f"[{workflow_id}] Phase 4: QA Validation")
                coder_output = self._get_agent_output(context, 'coder')
                
                qa_result = self._execute_agent_task(
                    context, 'qa', 'qa-validation', 'validation',
                    {
                        'task_description': 'Validate all deliverables',
                        'coder_output': coder_output,
                        'project_requirements': context.project_spec,
                        'success_criteria': ceo_result['payload'].get('success_criteria', [])
                    }
                )
                context.agent_outputs['qa'] = qa_result
            
            # Phase 5: Documentation
            logger.info(f"[{workflow_id}] Phase 5: Documentation")
            docs_result = self._execute_agent_task(
                context, 'docs', 'docs-generation', 'documentation',
                {
                    'task_description': 'Create documentation',
                    'project_spec': context.project_spec,
                    'coder_output': self._get_agent_output(context, 'coder'),
                    'qa_report': context.agent_outputs.get('qa', {}).get('payload', {})
                }
            )
            context.agent_outputs['docs'] = docs_result
            
            # Phase 6: CEO Finalization
            if self.config.get('require_ceo_finalization', True):
                logger.info(f"[{workflow_id}] Phase 6: CEO Finalization")
                final_result = self._execute_agent_task(
                    context, 'ceo', 'ceo-finalize', 'finalization',
                    {
                        'user_request': user_request,
                        'project_spec': context.project_spec,
                        'all_outputs': self._summarize_outputs(context),
                        'qa_report': context.agent_outputs.get('qa', {}).get('payload', {})
                    }
                )
                context.agent_outputs['ceo_final'] = final_result
            
            context.status = WorkflowStatus.COMPLETED
            elapsed = time.time() - context.start_time
            
            return self._compile_result(context, elapsed)
            
        except Exception as e:
            logger.exception(f"[{workflow_id}] Workflow error: {e}")
            context.status = WorkflowStatus.FAILED
            
            return {
                'success': False,
                'workflow_id': workflow_id,
                'error': str(e),
                'iterations': context.iteration,
                'duration': time.time() - context.start_time
            }
    
    def _execute_agent_task(
        self,
        context: WorkflowContext,
        agent_name: str,
        task_id: str,
        phase: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single agent task."""
        agent = self.agents.get(agent_name)
        if not agent:
            return self._error_response(task_id, agent_name, f"Unknown agent: {agent_name}")
        
        try:
            self._log_event(context, agent_name, 'task_started', {'task_id': task_id, 'phase': phase})
            
            result = agent.execute(task_id=task_id, phase=phase, inputs=inputs)
            
            self._log_event(context, agent_name, 'task_completed', {
                'task_id': task_id,
                'status': result.get('status'),
                'confidence': result.get('confidence'),
                'output': result.get('payload')
            })
            
            return result
        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {e}")
            return self._error_response(task_id, agent_name, str(e))
    
    def _error_response(self, task_id: str, agent_name: str, error: str) -> Dict[str, Any]:
        return {
            'status': 'failed',
            'task_id': task_id,
            'agent': agent_name,
            'payload': {'error': error},
            'confidence': 'low',
            'meta': {'elapsed': 0}
        }
    
    def _get_agent_output(self, context: WorkflowContext, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get output from a specific agent."""
        for key, value in context.agent_outputs.items():
            if key.startswith(agent_name):
                return value.get('payload', {})
        return {}
    
    def _summarize_outputs(self, context: WorkflowContext) -> Dict[str, Any]:
        """Summarize all agent outputs."""
        return {k: {'status': v.get('status')} for k, v in context.agent_outputs.items()}
    
    def _compile_result(self, context: WorkflowContext, elapsed: float) -> Dict[str, Any]:
        """Compile final workflow result."""
        coder_output = self._get_agent_output(context, 'coder') or {}
        docs_output = self._get_agent_output(context, 'docs') or {}
        qa_output = context.agent_outputs.get('qa', {}).get('payload', {})
        ceo_final = context.agent_outputs.get('ceo_final', {}).get('payload', {})
        
        return {
            'success': True,
            'workflow_id': context.workflow_id,
            'iterations': context.iteration,
            'duration': elapsed,
            'project_spec': context.project_spec,
            'deliverables': {
                'files': coder_output.get('files', []),
                'run_instructions': coder_output.get('run_instructions', {}),
                'dependencies': coder_output.get('dependencies', [])
            },
            'documentation': {
                'readme': docs_output.get('readme', ''),
                'summary': docs_output.get('summary', ''),
                'release_notes': docs_output.get('release_notes', '')
            },
            'quality_report': {
                'approval_status': qa_output.get('approval_status', 'unknown'),
                'validation_results': qa_output.get('validation_results', [])
            },
            'final_summary': ceo_final.get('final_summary', ''),
            'agent_outputs': context.agent_outputs,
            'events': context.events
        }
    
    def _log_event(self, context: WorkflowContext, agent_name: str, event_type: str, data: Dict[str, Any]):
        """Log an event."""
        event = {
            'workflow_id': context.workflow_id,
            'agent_name': agent_name,
            'task_id': data.get('task_id', ''),
            'event_type': event_type,
            'status': data.get('status', event_type),
            'payload': data
        }
        context.events.append(event)
        
        try:
            self.supabase_auth.log_event(event)
        except Exception as e:
            logger.warning(f"Failed to log event: {e}")
