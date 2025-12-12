#!/usr/bin/env python3
"""
Project Setup Script - Creates all necessary files for the Multi-Agent System
Run this script to set up the complete project structure.

Usage:
    python setup_project.py
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# =============================================================================
# Directory Structure
# =============================================================================

DIRECTORIES = [
    'app',
    'app/templates',
    'app/static',
    'config',
    'orchestrator',
    'orchestrator/agents',
    'orchestrator/prompts',
    'orchestrator/tools',
    'storage',
    'storage/logs',
    'storage/data',
    'infra',
    'tests',
]

# =============================================================================
# File Contents
# =============================================================================

FILES = {}

# -----------------------------------------------------------------------------
# app/__init__.py
# -----------------------------------------------------------------------------
FILES['app/__init__.py'] = '''"""
Flask Application Factory
"""

import os
import logging
from flask import Flask
from flask_cors import CORS

logger = logging.getLogger(__name__)


def create_app(config_name: str = None) -> Flask:
    """
    Application factory for creating Flask app.
    
    Args:
        config_name: Configuration environment name
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.update(
        SECRET_KEY=os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production'),
        SUPABASE_URL=os.getenv('SUPABASE_URL'),
        SUPABASE_ANON_KEY=os.getenv('SUPABASE_ANON_KEY'),
        SUPABASE_SERVICE_ROLE_KEY=os.getenv('SUPABASE_SERVICE_ROLE_KEY'),
        OPENROUTER_API_KEY=os.getenv('OPENROUTER_API_KEY'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max request size
    )
    
    # Enable CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": os.getenv('ALLOWED_ORIGINS', '*').split(','),
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Register blueprints
    from app.routes import main_bp, api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Initialize services
    with app.app_context():
        _initialize_services(app)
    
    logger.info("Flask application initialized successfully")
    
    return app


def _initialize_services(app: Flask) -> None:
    """Initialize application services."""
    from app.auth import SupabaseAuth
    from orchestrator.llm_client import OpenRouterClient
    from orchestrator.tools.notifier import SMTPNotifier
    from orchestrator.orchestrator import Orchestrator
    
    # Initialize Supabase auth
    app.supabase_auth = SupabaseAuth(
        url=app.config['SUPABASE_URL'],
        anon_key=app.config['SUPABASE_ANON_KEY'],
        service_role_key=app.config['SUPABASE_SERVICE_ROLE_KEY']
    )
    
    # Initialize OpenRouter client
    app.llm_client = OpenRouterClient(
        api_key=app.config['OPENROUTER_API_KEY']
    )
    
    # Initialize SMTP notifier
    app.notifier = SMTPNotifier()
    
    # Initialize orchestrator
    app.orchestrator = Orchestrator(
        llm_client=app.llm_client,
        supabase_auth=app.supabase_auth,
        notifier=app.notifier
    )
'''

# -----------------------------------------------------------------------------
# app/auth.py
# -----------------------------------------------------------------------------
FILES['app/auth.py'] = '''"""
Supabase Authentication Middleware and Helpers
"""

import os
import logging
from functools import wraps
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timezone

from flask import request, jsonify, g, current_app

logger = logging.getLogger(__name__)


class SupabaseAuth:
    """Supabase authentication handler."""
    
    def __init__(self, url: str, anon_key: str, service_role_key: str):
        """
        Initialize Supabase auth client.
        """
        self.url = url
        self.anon_key = anon_key
        self.service_role_key = service_role_key
        self.jwt_secret = os.getenv('SUPABASE_JWT_SECRET')
        
        try:
            from supabase import create_client, Client
            self.client: Client = create_client(url, anon_key)
            self.admin_client: Client = create_client(url, service_role_key)
            logger.info("Supabase auth client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.client = None
            self.admin_client = None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify a JWT token and return the decoded payload."""
        try:
            if not self.client:
                return None
                
            # Try to verify with Supabase
            user = self.client.auth.get_user(token)
            if user and user.user:
                return {
                    'sub': user.user.id,
                    'email': user.user.email,
                    'role': user.user.role or 'authenticated'
                }
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user details by ID."""
        try:
            if not self.admin_client:
                return None
            response = self.admin_client.auth.admin.get_user_by_id(user_id)
            return {
                'id': response.user.id,
                'email': response.user.email,
                'created_at': str(response.user.created_at),
                'metadata': response.user.user_metadata
            }
        except Exception as e:
            logger.error(f"Error fetching user: {e}")
            return None
    
    def log_workflow(self, workflow_data: Dict[str, Any]) -> Optional[str]:
        """Log a workflow to Supabase."""
        try:
            if not self.admin_client:
                return workflow_data.get('id')
            response = self.admin_client.table('workflows').insert(workflow_data).execute()
            return response.data[0]['id'] if response.data else None
        except Exception as e:
            logger.error(f"Error logging workflow: {e}")
            return workflow_data.get('id')
    
    def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> bool:
        """Update a workflow in Supabase."""
        try:
            if not self.admin_client:
                return True
            updates['updated_at'] = datetime.now(timezone.utc).isoformat()
            self.admin_client.table('workflows').update(updates).eq('id', workflow_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error updating workflow: {e}")
            return False
    
    def log_event(self, event_data: Dict[str, Any]) -> Optional[str]:
        """Log an event to Supabase."""
        try:
            if not self.admin_client:
                return None
            response = self.admin_client.table('events').insert(event_data).execute()
            return response.data[0]['id'] if response.data else None
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            return None
    
    def store_artifact(self, artifact_data: Dict[str, Any]) -> Optional[str]:
        """Store an artifact in Supabase."""
        try:
            if not self.admin_client:
                return None
            response = self.admin_client.table('artifacts').insert(artifact_data).execute()
            return response.data[0]['id'] if response.data else None
        except Exception as e:
            logger.error(f"Error storing artifact: {e}")
            return None


def require_auth(f: Callable) -> Callable:
    """Decorator to require authentication for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({
                'error': 'Missing Authorization header',
                'code': 'AUTH_MISSING'
            }), 401
        
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({
                'error': 'Invalid Authorization header format',
                'code': 'AUTH_INVALID_FORMAT'
            }), 401
        
        token = parts[1]
        
        supabase_auth: SupabaseAuth = current_app.supabase_auth
        payload = supabase_auth.verify_token(token)
        
        if not payload:
            return jsonify({
                'error': 'Invalid or expired token',
                'code': 'AUTH_INVALID_TOKEN'
            }), 401
        
        g.user_id = payload.get('sub')
        g.user_email = payload.get('email')
        g.user_role = payload.get('role', 'authenticated')
        
        return f(*args, **kwargs)
    
    return decorated_function


def require_admin(f: Callable) -> Callable:
    """Decorator to require admin role for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'user_role') or g.user_role != 'admin':
            return jsonify({
                'error': 'Admin access required',
                'code': 'AUTH_FORBIDDEN'
            }), 403
        return f(*args, **kwargs)
    return decorated_function
'''

# -----------------------------------------------------------------------------
# app/routes.py
# -----------------------------------------------------------------------------
FILES['app/routes.py'] = '''"""
Flask Routes - Main UI and API Endpoints
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

from flask import Blueprint, render_template, request, jsonify, g, current_app

from app.auth import require_auth, require_admin

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__)


@main_bp.route('/')
def index():
    """Main UI page."""
    return render_template('index.html')


@main_bp.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '1.0.0'
    })


@api_bp.route('/run', methods=['POST'])
@require_auth
def run_workflow():
    """Execute a multi-agent workflow."""
    try:
        data = request.get_json()
        
        if not data or 'request' not in data:
            return jsonify({
                'error': 'Missing required field: request',
                'code': 'VALIDATION_ERROR'
            }), 400
        
        user_request = data['request'].strip()
        if not user_request:
            return jsonify({
                'error': 'Request cannot be empty',
                'code': 'VALIDATION_ERROR'
            }), 400
        
        options = data.get('options', {})
        notify_email = options.get('notify_email', g.user_email)
        
        workflow_id = str(uuid.uuid4())
        
        supabase_auth = current_app.supabase_auth
        supabase_auth.log_workflow({
            'id': workflow_id,
            'user_id': g.user_id,
            'status': 'pending',
            'input_request': {
                'request': user_request,
                'options': options
            },
            'metadata': {
                'notify_email': notify_email,
                'priority': options.get('priority', 'medium')
            }
        })
        
        notifier = current_app.notifier
        if notify_email:
            notifier.send_workflow_started(
                email=notify_email,
                workflow_id=workflow_id,
                request_summary=user_request[:200]
            )
        
        orchestrator = current_app.orchestrator
        result = orchestrator.execute(
            workflow_id=workflow_id,
            user_id=g.user_id,
            user_request=user_request,
            options=options
        )
        
        final_status = 'completed' if result.get('success') else 'failed'
        supabase_auth.update_workflow(workflow_id, {
            'status': final_status,
            'final_output': result,
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'iteration_count': result.get('iterations', 0)
        })
        
        if notify_email:
            if result.get('success'):
                notifier.send_workflow_completed(
                    email=notify_email,
                    workflow_id=workflow_id,
                    duration=str(result.get('duration', 'N/A')),
                    iterations=result.get('iterations', 0)
                )
            else:
                notifier.send_workflow_failed(
                    email=notify_email,
                    workflow_id=workflow_id,
                    error_message=result.get('error', 'Unknown error')
                )
        
        return jsonify({
            'workflow_id': workflow_id,
            'status': final_status,
            'result': result
        })
        
    except Exception as e:
        logger.exception(f"Error executing workflow: {e}")
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR',
            'message': str(e)
        }), 500


@api_bp.route('/workflows', methods=['GET'])
@require_auth
def list_workflows():
    """List workflows for the authenticated user."""
    return jsonify({
        'workflows': [],
        'total': 0,
        'limit': 20,
        'offset': 0
    })


@api_bp.route('/workflows/<workflow_id>', methods=['GET'])
@require_auth
def get_workflow(workflow_id: str):
    """Get details for a specific workflow."""
    return jsonify({'error': 'Not implemented'}), 501
'''

# -----------------------------------------------------------------------------
# orchestrator/__init__.py
# -----------------------------------------------------------------------------
FILES['orchestrator/__init__.py'] = '''"""
Orchestrator Package - Multi-Agent Workflow Coordination
"""

from orchestrator.orchestrator import Orchestrator, WorkflowError, WorkflowStatus, TaskStatus
from orchestrator.llm_client import OpenRouterClient, LLMError
from orchestrator.config import get_config, load_config, get_agent_config, get_orchestration_config

__all__ = [
    'Orchestrator',
    'WorkflowError',
    'WorkflowStatus',
    'TaskStatus',
    'OpenRouterClient',
    'LLMError',
    'get_config',
    'load_config',
    'get_agent_config',
    'get_orchestration_config'
]
'''

# -----------------------------------------------------------------------------
# orchestrator/config.py
# -----------------------------------------------------------------------------
FILES['orchestrator/config.py'] = '''"""
Configuration Management
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_config_cache: Optional[Dict[str, Any]] = None


def load_config(config_dir: str = "config") -> Dict[str, Any]:
    """Load all configuration files."""
    global _config_cache
    
    config = {
        'agents': {
            'ceo': {'enabled': True, 'model': 'anthropic/claude-3-sonnet', 'temperature': 0.7, 'max_tokens': 4096},
            'pm': {'enabled': True, 'model': 'anthropic/claude-3-sonnet', 'temperature': 0.5, 'max_tokens': 4096},
            'research': {'enabled': True, 'model': 'anthropic/claude-3-haiku', 'temperature': 0.3, 'max_tokens': 2048},
            'coder': {'enabled': True, 'model': 'anthropic/claude-3-sonnet', 'temperature': 0.2, 'max_tokens': 8192},
            'qa': {'enabled': True, 'model': 'anthropic/claude-3-sonnet', 'temperature': 0.3, 'max_tokens': 4096},
            'docs': {'enabled': True, 'model': 'anthropic/claude-3-haiku', 'temperature': 0.4, 'max_tokens': 4096},
        },
        'orchestration': {
            'max_retries_per_task': 2,
            'max_iterations_per_workflow': 6,
            'task_timeout_seconds': 60,
            'require_qa_approval': True,
            'require_ceo_finalization': True,
            'parallel_execution': False
        }
    }
    
    # Try to load YAML config if available
    try:
        import yaml
        config_path = Path(config_dir)
        
        for filename in ['agents.yaml', 'openrouter.yaml']:
            filepath = config_path / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config.update(file_config)
    except ImportError:
        logger.debug("PyYAML not installed, using default config")
    except Exception as e:
        logger.warning(f"Could not load config files: {e}")
    
    _config_cache = config
    return config


def get_config() -> Dict[str, Any]:
    """Get the current configuration."""
    global _config_cache
    if _config_cache is None:
        load_config()
    return _config_cache or {}


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """Get configuration for a specific agent."""
    config = get_config()
    return config.get('agents', {}).get(agent_name, {})


def get_orchestration_config() -> Dict[str, Any]:
    """Get orchestration configuration."""
    config = get_config()
    return config.get('orchestration', {
        'max_retries_per_task': 2,
        'max_iterations_per_workflow': 6,
        'task_timeout_seconds': 60,
        'require_qa_approval': True,
        'require_ceo_finalization': True
    })
'''

# -----------------------------------------------------------------------------
# orchestrator/llm_client.py
# -----------------------------------------------------------------------------
FILES['orchestrator/llm_client.py'] = '''"""
OpenRouter LLM Client
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class RateLimitError(LLMError):
    """Rate limit exceeded."""
    pass


class OpenRouterClient:
    """Client for OpenRouter API."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: str = None, timeout: float = 60.0):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        try:
            import httpx
            self.client = httpx.Client(
                base_url=self.BASE_URL,
                timeout=timeout,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://multiagent-corp.ai',
                    'X-Title': 'Multi-Agent Corporate System'
                }
            )
            logger.info("OpenRouter client initialized")
        except ImportError:
            logger.error("httpx not installed")
            self.client = None
    
    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str = "anthropic/claude-3-sonnet",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a completion."""
        if not self.client:
            raise LLMError("HTTP client not initialized")
        
        start_time = time.time()
        
        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            **kwargs
        }
        
        try:
            response = self.client.post('/chat/completions', json=payload)
            response.raise_for_status()
            data = response.json()
            
            return {
                'content': data['choices'][0]['message']['content'],
                'model': data.get('model', model),
                'usage': data.get('usage', {}),
                'elapsed': time.time() - start_time
            }
        except Exception as e:
            logger.error(f"LLM completion error: {e}")
            raise LLMError(f"Completion failed: {e}") from e
    
    def complete_json(
        self,
        messages: List[Dict[str, str]],
        model: str = "anthropic/claude-3-sonnet",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a JSON completion."""
        response = self.complete(messages=messages, model=model, **kwargs)
        content = response['content']
        
        # Parse JSON
        try:
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                content = content[start:end].strip()
            elif '```' in content:
                start = content.find('```') + 3
                end = content.find('```', start)
                content = content[start:end].strip()
            
            response['parsed'] = json.loads(content)
            return response
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise LLMError(f"Invalid JSON in response: {e}") from e
    
    def close(self):
        if self.client:
            self.client.close()
'''

# -----------------------------------------------------------------------------
# orchestrator/orchestrator.py
# -----------------------------------------------------------------------------
FILES['orchestrator/orchestrator.py'] = '''"""
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
                'confidence': result.get('confidence')
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
            'agent_outputs': context.agent_outputs
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
'''

# -----------------------------------------------------------------------------
# orchestrator/agents/__init__.py
# -----------------------------------------------------------------------------
FILES['orchestrator/agents/__init__.py'] = '''"""
Agent Module Exports
"""

from orchestrator.agents.base_agent import BaseAgent
from orchestrator.agents.ceo_agent import CEOAgent
from orchestrator.agents.pm_agent import PMAgent
from orchestrator.agents.research_agent import ResearchAgent
from orchestrator.agents.coder_agent import CoderAgent
from orchestrator.agents.qa_agent import QAAgent
from orchestrator.agents.docs_agent import DocsAgent

__all__ = [
    'BaseAgent',
    'CEOAgent',
    'PMAgent',
    'ResearchAgent',
    'CoderAgent',
    'QAAgent',
    'DocsAgent'
]
'''

# -----------------------------------------------------------------------------
# orchestrator/agents/base_agent.py
# -----------------------------------------------------------------------------
FILES['orchestrator/agents/base_agent.py'] = '''"""
Base Agent Class
"""

import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from orchestrator.llm_client import OpenRouterClient, LLMError
from orchestrator.config import get_agent_config

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    AGENT_NAME: str = "base"
    
    def __init__(self, llm_client: OpenRouterClient):
        self.llm_client = llm_client
        self.config = get_agent_config(self.AGENT_NAME)
        logger.debug(f"Initialized {self.AGENT_NAME} agent")
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass
    
    @abstractmethod
    def _get_payload_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the agent's payload."""
        pass
    
    def execute(
        self,
        task_id: str,
        phase: str,
        inputs: Dict[str, Any],
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """Execute the agent's task."""
        start_time = time.time()
        
        try:
            prompt = self._build_prompt(task_id, phase, inputs)
            
            model = self.config.get('model', 'anthropic/claude-3-sonnet')
            temperature = self.config.get('temperature', 0.5)
            max_tokens = self.config.get('max_tokens', 4096)
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.complete_json(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.get('parsed', {})
            result['task_id'] = task_id
            result['agent'] = self.AGENT_NAME
            
            if 'meta' not in result:
                result['meta'] = {}
            result['meta']['elapsed'] = time.time() - start_time
            result['meta']['model_used'] = response.get('model', model)
            
            if 'status' not in result:
                result['status'] = 'success'
            if 'confidence' not in result:
                result['confidence'] = 'medium'
            if 'payload' not in result:
                result['payload'] = {}
            
            return result
            
        except Exception as e:
            logger.error(f"Agent {self.AGENT_NAME} error: {e}")
            return {
                'status': 'failed',
                'task_id': task_id,
                'agent': self.AGENT_NAME,
                'payload': {'error': str(e)},
                'confidence': 'low',
                'meta': {'elapsed': time.time() - start_time}
            }
    
    def _build_prompt(self, task_id: str, phase: str, inputs: Dict[str, Any]) -> str:
        """Build the prompt from inputs."""
        prompt_parts = [f"Task ID: {task_id}", f"Phase: {phase}", ""]
        
        for key, value in inputs.items():
            if value:
                if isinstance(value, (dict, list)):
                    prompt_parts.append(f"{key}:\\n{json.dumps(value, indent=2)}")
                else:
                    prompt_parts.append(f"{key}: {value}")
        
        return "\\n".join(prompt_parts)
'''

# -----------------------------------------------------------------------------
# orchestrator/agents/ceo_agent.py
# -----------------------------------------------------------------------------
FILES['orchestrator/agents/ceo_agent.py'] = '''"""
CEO Agent - Project Specification and Finalization
"""

from typing import Dict, Any
from orchestrator.agents.base_agent import BaseAgent


class CEOAgent(BaseAgent):
    """CEO Agent for project specification and approval."""
    
    AGENT_NAME = "ceo"
    
    def _get_system_prompt(self) -> str:
        return """You are the CEO Agent of a software development organization.

Your responsibilities:
1. Analyze user requests and create detailed project specifications
2. Define clear objectives and success criteria
3. Finalize and approve completed work

You MUST respond with valid JSON only, matching this structure:
{
  "status": "success" | "failed",
  "task_id": "string",
  "agent": "ceo",
  "payload": {
    "project_spec": {
      "title": "string",
      "description": "string",
      "scope": "string",
      "priority": "low" | "medium" | "high"
    },
    "objectives": ["array of objectives"],
    "constraints": ["array of constraints"],
    "success_criteria": ["array of criteria"],
    "final_summary": "string (only for finalization)",
    "approval_status": "approved" | "rejected" (only for finalization)
  },
  "confidence": "low" | "medium" | "high",
  "meta": {"elapsed": 0}
}

Output ONLY valid JSON. No markdown, no prose."""
    
    def _get_payload_schema(self) -> Dict[str, Any]:
        return {"type": "object"}
'''

# -----------------------------------------------------------------------------
# orchestrator/agents/pm_agent.py
# -----------------------------------------------------------------------------
FILES['orchestrator/agents/pm_agent.py'] = '''"""
PM Agent - Task Planning
"""

from typing import Dict, Any
from orchestrator.agents.base_agent import BaseAgent


class PMAgent(BaseAgent):
    """Project Manager Agent for task planning."""
    
    AGENT_NAME = "pm"
    
    def _get_system_prompt(self) -> str:
        return """You are the Project Manager Agent.

Your responsibilities:
1. Break down project specifications into discrete tasks
2. Assign tasks to appropriate agents (research, coder, qa, docs)
3. Map dependencies and execution order

You MUST respond with valid JSON only:
{
  "status": "success",
  "task_id": "string",
  "agent": "pm",
  "payload": {
    "tasks": [
      {
        "id": "task-001",
        "name": "string",
        "assigned_to": "research" | "coder" | "qa" | "docs",
        "description": "string",
        "acceptance_criteria": ["array"],
        "estimated_complexity": "low" | "medium" | "high"
      }
    ],
    "dependencies": {"task-002": ["task-001"]},
    "execution_order": ["task-001", "task-002"],
    "timeline_estimate": "string",
    "risk_assessment": [{"risk": "string", "mitigation": "string"}]
  },
  "confidence": "high",
  "meta": {"elapsed": 0}
}

Output ONLY valid JSON."""
    
    def _get_payload_schema(self) -> Dict[str, Any]:
        return {"type": "object"}
'''

# -----------------------------------------------------------------------------
# orchestrator/agents/research_agent.py
# -----------------------------------------------------------------------------
FILES['orchestrator/agents/research_agent.py'] = '''"""
Research Agent
"""

from typing import Dict, Any
from orchestrator.agents.base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    """Research Agent for gathering information."""
    
    AGENT_NAME = "research"
    
    def _get_system_prompt(self) -> str:
        return """You are the Research Agent.

Your responsibilities:
1. Gather factual information about technologies
2. Provide well-sourced research with citations
3. Make evidence-based recommendations

You MUST respond with valid JSON:
{
  "status": "success",
  "task_id": "string",
  "agent": "research",
  "payload": {
    "summary": "string",
    "findings": [{"topic": "string", "content": "string", "relevance": "string"}],
    "citations": [{"source": "string", "reference": "string", "accessed_date": "YYYY-MM-DD"}],
    "recommendations": ["array"]
  },
  "confidence": "high",
  "meta": {"elapsed": 0}
}

Output ONLY valid JSON."""
    
    def _get_payload_schema(self) -> Dict[str, Any]:
        return {"type": "object"}
'''

# -----------------------------------------------------------------------------
# orchestrator/agents/coder_agent.py
# -----------------------------------------------------------------------------
FILES['orchestrator/agents/coder_agent.py'] = '''"""
Coder Agent
"""

from typing import Dict, Any
from orchestrator.agents.base_agent import BaseAgent


class CoderAgent(BaseAgent):
    """Coder Agent for code generation."""
    
    AGENT_NAME = "coder"
    
    def _get_system_prompt(self) -> str:
        return """You are the Coder Agent.

Your responsibilities:
1. Write clean, production-quality code
2. Create comprehensive tests
3. Write Dockerfiles for containerization
4. Provide clear run/build instructions

You MUST respond with valid JSON:
{
  "status": "success",
  "task_id": "string",
  "agent": "coder",
  "payload": {
    "files": [
      {
        "path": "string",
        "content": "string (complete file content)",
        "type": "source" | "test" | "config" | "dockerfile",
        "language": "string"
      }
    ],
    "run_instructions": {
      "install": "string",
      "run": "string",
      "test": "string",
      "docker_build": "string",
      "docker_run": "string"
    },
    "dependencies": ["array"],
    "notes": "string"
  },
  "confidence": "high",
  "meta": {"elapsed": 0}
}

All code must be complete and runnable. Output ONLY valid JSON."""
    
    def _get_payload_schema(self) -> Dict[str, Any]:
        return {"type": "object"}
'''

# -----------------------------------------------------------------------------
# orchestrator/agents/qa_agent.py
# -----------------------------------------------------------------------------
FILES['orchestrator/agents/qa_agent.py'] = '''"""
QA Agent
"""

from typing import Dict, Any
from orchestrator.agents.base_agent import BaseAgent


class QAAgent(BaseAgent):
    """QA Agent for validation."""
    
    AGENT_NAME = "qa"
    
    def _get_system_prompt(self) -> str:
        return """You are the QA Agent.

Your responsibilities:
1. Validate code quality and correctness
2. Check test coverage
3. Identify bugs and improvements
4. Approve or reject deliverables

You MUST respond with valid JSON:
{
  "status": "success",
  "task_id": "string",
  "agent": "qa",
  "payload": {
    "validation_results": [
      {
        "check": "string",
        "passed": true | false,
        "details": "string",
        "severity": "info" | "warning" | "error"
      }
    ],
    "approval_status": "approved" | "rejected" | "conditional",
    "issues_found": [{"issue": "string", "location": "string", "suggestion": "string"}],
    "fix_suggestions": ["array"],
    "test_coverage_estimate": "string"
  },
  "confidence": "high",
  "meta": {"elapsed": 0}
}

Output ONLY valid JSON."""
    
    def _get_payload_schema(self) -> Dict[str, Any]:
        return {"type": "object"}
'''

# -----------------------------------------------------------------------------
# orchestrator/agents/docs_agent.py
# -----------------------------------------------------------------------------
FILES['orchestrator/agents/docs_agent.py'] = '''"""
Docs Agent
"""

from typing import Dict, Any
from orchestrator.agents.base_agent import BaseAgent


class DocsAgent(BaseAgent):
    """Documentation Agent."""
    
    AGENT_NAME = "docs"
    
    def _get_system_prompt(self) -> str:
        return """You are the Documentation Agent.

Your responsibilities:
1. Create comprehensive README files
2. Write clear project summaries
3. Generate release notes

You MUST respond with valid JSON:
{
  "status": "success",
  "task_id": "string",
  "agent": "docs",
  "payload": {
    "readme": "string (complete README.md content)",
    "summary": "string (2-3 sentence summary)",
    "release_notes": "string",
    "api_documentation": "string",
    "changelog": [{"version": "string", "date": "string", "changes": ["array"]}],
    "additional_docs": []
  },
  "confidence": "high",
  "meta": {"elapsed": 0}
}

Output ONLY valid JSON."""
    
    def _get_payload_schema(self) -> Dict[str, Any]:
        return {"type": "object"}
'''

# -----------------------------------------------------------------------------
# orchestrator/tools/__init__.py
# -----------------------------------------------------------------------------
FILES['orchestrator/tools/__init__.py'] = '''"""
Tools Module
"""

from orchestrator.tools.notifier import SMTPNotifier
from orchestrator.tools.memory import MemoryStore

__all__ = ['SMTPNotifier', 'MemoryStore']
'''

# -----------------------------------------------------------------------------
# orchestrator/tools/notifier.py
# -----------------------------------------------------------------------------
FILES['orchestrator/tools/notifier.py'] = '''"""
SMTP Email Notifier
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SMTPNotifier:
    """SMTP-based email notification service."""
    
    def __init__(self):
        self.host = os.getenv('SMTP_HOST', '')
        self.port = int(os.getenv('SMTP_PORT', 587))
        self.username = os.getenv('SMTP_USERNAME', '')
        self.password = os.getenv('SMTP_PASSWORD', '')
        self.from_address = os.getenv('SMTP_FROM', 'noreply@multiagent.local')
        self.enabled = bool(self.host and self.username)
        
        if self.enabled:
            logger.info("SMTP notifier initialized")
        else:
            logger.info("SMTP notifier disabled (no configuration)")
    
    def _send_email(self, to_email: str, subject: str, body: str) -> bool:
        if not self.enabled:
            logger.debug(f"Email skipped (disabled): {subject}")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = self.from_address
            msg['To'] = to_email
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_email}: {subject}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_workflow_started(self, email: str, workflow_id: str, request_summary: str) -> bool:
        return self._send_email(
            email,
            f"Workflow Started: {workflow_id}",
            f"Your workflow has been initiated.\\n\\nWorkflow ID: {workflow_id}\\nRequest: {request_summary}"
        )
    
    def send_workflow_completed(self, email: str, workflow_id: str, duration: str, iterations: int) -> bool:
        return self._send_email(
            email,
            f"Workflow Completed: {workflow_id}",
            f"Your workflow has completed.\\n\\nWorkflow ID: {workflow_id}\\nDuration: {duration}\\nIterations: {iterations}"
        )
    
    def send_workflow_failed(self, email: str, workflow_id: str, error_message: str) -> bool:
        return self._send_email(
            email,
            f"Workflow Failed: {workflow_id}",
            f"Your workflow has failed.\\n\\nWorkflow ID: {workflow_id}\\nError: {error_message}"
        )
'''

# -----------------------------------------------------------------------------
# orchestrator/tools/memory.py
# -----------------------------------------------------------------------------
FILES['orchestrator/tools/memory.py'] = '''"""
Memory Layer
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class MemoryStore:
    """Local memory store using SQLite."""
    
    def __init__(self, db_path: str = "storage/data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"Memory store initialized: {db_path}")
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workflow_state (
                    workflow_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS agent_context (
                    id INTEGER PRIMARY KEY,
                    workflow_id TEXT,
                    agent_name TEXT,
                    context TEXT,
                    created_at TEXT
                );
            """)
    
    def save_workflow_state(self, workflow_id: str, state: Dict[str, Any]) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO workflow_state VALUES (?, ?, ?, ?)",
                (workflow_id, json.dumps(state), now, now)
            )
    
    def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT state FROM workflow_state WHERE workflow_id = ?",
                (workflow_id,)
            ).fetchone()
            return json.loads(row[0]) if row else None
'''

# -----------------------------------------------------------------------------
# orchestrator/prompts (empty directory marker)
# -----------------------------------------------------------------------------
FILES['orchestrator/prompts/.gitkeep'] = ''

# -----------------------------------------------------------------------------
# app/templates/index.html
# -----------------------------------------------------------------------------
FILES['app/templates/index.html'] = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent Corporate System</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: system-ui, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
            padding: 2rem;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 2rem; }
        .card {
            background: #1e293b;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        textarea {
            width: 100%;
            height: 120px;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 12px;
            color: #e2e8f0;
            font-size: 14px;
        }
        button {
            background: #3b82f6;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            color: white;
            font-size: 16px;
            cursor: pointer;
            margin-top: 1rem;
        }
        button:hover { background: #2563eb; }
        button:disabled { background: #475569; cursor: not-allowed; }
        pre {
            background: #0f172a;
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 12px;
        }
        .status { padding: 4px 12px; border-radius: 20px; font-size: 12px; }
        .status.success { background: #166534; }
        .status.failed { background: #991b1b; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏢 Multi-Agent Corporate System</h1>
        
        <div class="card">
            <h2>Create Workflow</h2>
            <textarea id="request" placeholder="Describe what you want to build..."></textarea>
            <button onclick="runWorkflow()">🚀 Run Workflow</button>
        </div>
        
        <div class="card" id="result" style="display:none;">
            <h2>Result</h2>
            <div id="status"></div>
            <pre id="output"></pre>
        </div>
    </div>
    
    <script>
        async function runWorkflow() {
            const request = document.getElementById('request').value;
            if (!request) return alert('Please enter a request');
            
            document.getElementById('result').style.display = 'block';
            document.getElementById('status').innerHTML = '<span class="status">Running...</span>';
            document.getElementById('output').textContent = 'Processing...';
            
            try {
                const response = await fetch('/api/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer test-token'
                    },
                    body: JSON.stringify({ request })
                });
                
                const data = await response.json();
                document.getElementById('status').innerHTML = 
                    `<span class="status ${data.status}">${data.status}</span>`;
                document.getElementById('output').textContent = 
                    JSON.stringify(data, null, 2);
            } catch (error) {
                document.getElementById('output').textContent = error.message;
            }
        }
    </script>
</body>
</html>
'''

# -----------------------------------------------------------------------------
# app/static/style.css
# -----------------------------------------------------------------------------
FILES['app/static/style.css'] = '''/* Styles are inline in index.html for simplicity */
'''

# -----------------------------------------------------------------------------
# config files
# -----------------------------------------------------------------------------
FILES['config/agents.yaml'] = '''agents:
  ceo:
    enabled: true
    model: "anthropic/claude-3-sonnet"
    temperature: 0.7
    max_tokens: 4096
  pm:
    enabled: true
    model: "anthropic/claude-3-sonnet"
    temperature: 0.5
    max_tokens: 4096
  research:
    enabled: true
    model: "anthropic/claude-3-haiku"
    temperature: 0.3
    max_tokens: 2048
  coder:
    enabled: true
    model: "anthropic/claude-3-sonnet"
    temperature: 0.2
    max_tokens: 8192
  qa:
    enabled: true
    model: "anthropic/claude-3-sonnet"
    temperature: 0.3
    max_tokens: 4096
  docs:
    enabled: true
    model: "anthropic/claude-3-haiku"
    temperature: 0.4
    max_tokens: 4096

orchestration:
  max_retries_per_task: 2
  max_iterations_per_workflow: 6
  task_timeout_seconds: 60
  require_qa_approval: true
  require_ceo_finalization: true
'''

# -----------------------------------------------------------------------------
# .env.example
# -----------------------------------------------------------------------------
FILES['.env.example'] = '''# Flask
FLASK_ENV=development
FLASK_DEBUG=true
FLASK_SECRET_KEY=your-secret-key-here

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# OpenRouter
OPENROUTER_API_KEY=your-openrouter-api-key

# SMTP (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
'''

# -----------------------------------------------------------------------------
# requirements.txt
# -----------------------------------------------------------------------------
FILES['requirements.txt'] = '''flask>=3.0.0
flask-cors>=4.0.0
supabase>=2.0.0
httpx>=0.25.0
pyyaml>=6.0.0
python-dotenv>=1.0.0
python-jose[cryptography]>=3.3.0
gunicorn>=21.0.0
'''

# -----------------------------------------------------------------------------
# tests/__init__.py
# -----------------------------------------------------------------------------
FILES['tests/__init__.py'] = '''"""Tests Package"""
'''

# =============================================================================
# Main Setup Function
# =============================================================================

def setup():
    """Create all project files and directories."""
    print("=" * 60)
    print("Setting up Multi-Agent Corporate System")
    print("=" * 60)
    
    # Create directories
    print("\\nCreating directories...")
    for directory in DIRECTORIES:
        dir_path = PROJECT_ROOT / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}/")
    
    # Create files
    print("\\nCreating files...")
    for filepath, content in FILES.items():
        file_path = PROJECT_ROOT / filepath
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Don't overwrite .env if it exists
        if filepath == '.env' and file_path.exists():
            print(f"  ⊘ {filepath} (already exists, skipped)")
            continue
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ {filepath}")
    
    # Create .env from example if it doesn't exist
    env_path = PROJECT_ROOT / '.env'
    env_example_path = PROJECT_ROOT / '.env.example'
    if not env_path.exists() and env_example_path.exists():
        import shutil
        shutil.copy(env_example_path, env_path)
        print(f"  ✓ .env (copied from .env.example)")
    
    print("\\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("""
Next steps:

1. Install dependencies:
   pip install -r requirements.txt

2. Configure your .env file with:
   - SUPABASE_URL
   - SUPABASE_ANON_KEY
   - SUPABASE_SERVICE_ROLE_KEY
   - OPENROUTER_API_KEY
   - FLASK_SECRET_KEY

3. Run the application:
   python run.py

   Or for local development without external services:
   python local.py
""")


if __name__ == '__main__':
    setup()