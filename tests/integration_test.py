"""
Integration Tests for Multi-Agent Corporate System
"""

import os
import json
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

# Test fixtures
@pytest.fixture
def mock_supabase_auth():
    """Create mock Supabase auth."""
    mock = Mock()
    mock.verify_token.return_value = {
        'sub': 'test-user-id',
        'email': 'test@example.com',
        'role': 'authenticated'
    }
    mock.log_workflow.return_value = 'test-workflow-id'
    mock.update_workflow.return_value = True
    mock.log_event.return_value = 'test-event-id'
    mock.store_artifact.return_value = 'test-artifact-id'
    return mock


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    mock = Mock()
    
    def mock_complete_json(messages, model, **kwargs):
        # Return appropriate response based on context
        content = messages[0]['content'] if messages else ''
        
        if 'CEO' in content and 'specification' in content.lower():
            return {
                'parsed': {
                    'status': 'success',
                    'task_id': 'ceo-spec',
                    'agent': 'ceo',
                    'payload': {
                        'project_spec': {
                            'title': 'Test Project',
                            'description': 'A test project',
                            'scope': 'Testing only',
                            'priority': 'medium'
                        },
                        'objectives': ['Create test endpoint'],
                        'constraints': ['Must be simple'],
                        'success_criteria': ['Endpoint works']
                    },
                    'confidence': 'high',
                    'meta': {'elapsed': 1.0}
                },
                'model': model,
                'usage': {'total_tokens': 100}
            }
        elif 'PM' in content or 'Project Manager' in content:
            return {
                'parsed': {
                    'status': 'success',
                    'task_id': 'pm-planning',
                    'agent': 'pm',
                    'payload': {
                        'tasks': [
                            {
                                'id': 'task-001',
                                'name': 'Implement endpoint',
                                'assigned_to': 'coder',
                                'description': 'Create the health endpoint',
                                'acceptance_criteria': ['Returns 200'],
                                'estimated_complexity': 'low'
                            }
                        ],
                        'dependencies': {},
                        'execution_order': ['task-001'],
                        'timeline_estimate': '1 hour',
                        'risk_assessment': []
                    },
                    'confidence': 'high',
                    'meta': {'elapsed': 1.0}
                },
                'model': model,
                'usage': {'total_tokens': 100}
            }
        elif 'Coder' in content:
            return {
                'parsed': {
                    'status': 'success',
                    'task_id': 'task-001',
                    'agent': 'coder',
                    'payload': {
                        'files': [
                            {
                                'path': 'app.py',
                                'content': 'from flask import Flask\\napp = Flask(__name__)\\n@app.route("/health")\\ndef health():\\n    return {"status": "ok"}',
                                'type': 'source',
                                'language': 'python'
                            },
                            {
                                'path': 'test_app.py',
                                'content': 'def test_health():\\n    assert True',
                                'type': 'test',
                                'language': 'python'
                            },
                            {
                                'path': 'Dockerfile',
                                'content': 'FROM python:3.11\\nCOPY . .\\nCMD ["python", "app.py"]',
                                'type': 'dockerfile',
                                'language': 'dockerfile'
                            },
                            {
                                'path': 'requirements.txt',
                                'content': 'flask>=3.0.0',
                                'type': 'config',
                                'language': 'text'
                            }
                        ],
                        'run_instructions': {
                            'install': 'pip install -r requirements.txt',
                            'run': 'python app.py',
                            'test': 'pytest',
                            'docker_build': 'docker build -t app .',
                            'docker_run': 'docker run -p 5000:5000 app'
                        },
                        'dependencies': ['flask>=3.0.0'],
                        'notes': 'Simple Flask app'
                    },
                    'confidence': 'high',
                    'meta': {'elapsed': 2.0}
                },
                'model': model,
                'usage': {'total_tokens': 200}
            }
        elif 'QA' in content:
            return {
                'parsed': {
                    'status': 'success',
                    'task_id': 'qa-validation',
                    'agent': 'qa',
                    'payload': {
                        'validation_results': [
                            {
                                'check': 'Code completeness',
                                'passed': True,
                                'details': 'All code is complete',
                                'severity': 'info'
                            }
                        ],
                        'approval_status': 'approved',
                        'issues_found': [],
                        'fix_suggestions': [],
                        'test_coverage_estimate': '80%'
                    },
                    'confidence': 'high',
                    'meta': {'elapsed': 1.0}
                },
                'model': model,
                'usage': {'total_tokens': 100}
            }
        elif 'Documentation' in content or 'Docs' in content:
            return {
                'parsed': {
                    'status': 'success',
                    'task_id': 'docs-generation',
                    'agent': 'docs',
                    'payload': {
                        'readme': '# Test Project\\n\\nA simple Flask app with health endpoint.',
                        'summary': 'Flask microservice with /health endpoint.',
                        'release_notes': '## v1.0.0\\n\\nInitial release.',
                        'api_documentation': 'GET /health - Returns health status',
                        'changelog': [
                            {
                                'version': '1.0.0',
                                'date': '2024-01-01',
                                'changes': ['Initial release']
                            }
                        ],
                        'additional_docs': []
                    },
                    'confidence': 'high',
                    'meta': {'elapsed': 1.0}
                },
                'model': model,
                'usage': {'total_tokens': 100}
            }
        elif 'finalization' in content.lower():
            return {
                'parsed': {
                    'status': 'success',
                    'task_id': 'ceo-finalize',
                    'agent': 'ceo',
                    'payload': {
                        'project_spec': {
                            'title': 'Test Project',
                            'description': 'A test project',
                            'scope': 'Testing only',
                            'priority': 'medium'
                        },
                        'objectives': ['Create test endpoint'],
                        'constraints': ['Must be simple'],
                        'success_criteria': ['Endpoint works'],
                        'final_summary': 'Project completed successfully.',
                        'approval_status': 'approved'
                    },
                    'confidence': 'high',
                    'meta': {'elapsed': 1.0}
                },
                'model': model,
                'usage': {'total_tokens': 100}
            }
        else:
            return {
                'parsed': {
                    'status': 'success',
                    'task_id': 'unknown',
                    'agent': 'unknown',
                    'payload': {},
                    'confidence': 'low',
                    'meta': {'elapsed': 0.5}
                },
                'model': model,
                'usage': {'total_tokens': 50}
            }
    
    mock.complete_json = mock_complete_json
    return mock


@pytest.fixture
def mock_notifier():
    """Create mock notifier."""
    mock = Mock()
    mock.send_workflow_started.return_value = True
    mock.send_workflow_completed.return_value = True
    mock.send_workflow_failed.return_value = True
    return mock


@pytest.fixture
def app(mock_supabase_auth, mock_llm_client, mock_notifier):
    """Create Flask test application."""
    from app import create_app
    
    # Patch where the classes are IMPORTED/USED in the application factory
    with patch('app.auth.SupabaseAuth', return_value=mock_supabase_auth), \
         patch('orchestrator.llm_client.OpenRouterClient', return_value=mock_llm_client), \
         patch('orchestrator.tools.notifier.SMTPNotifier', return_value=mock_notifier):
            
            app = create_app()
            app.config['TESTING'] = True
            
            yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def auth_headers():
    """Create authorization headers."""
    return {'Authorization': 'Bearer test-token-123'}


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data


class TestAuthentication:
    """Tests for authentication."""
    
    def test_missing_auth_header(self, client):
        """Request without auth header should return 401."""
        response = client.post('/api/run', json={'request': 'test'})
        assert response.status_code == 401
        
        data = response.get_json()
        assert data['code'] == 'AUTH_MISSING'
    
    def test_invalid_auth_format(self, client):
        """Request with invalid auth format should return 401."""
        response = client.post(
            '/api/run',
            json={'request': 'test'},
            headers={'Authorization': 'InvalidFormat'}
        )
        assert response.status_code == 401
        
        data = response.get_json()
        assert data['code'] == 'AUTH_INVALID_FORMAT'
    
    def test_valid_auth(self, client, auth_headers):
        """Request with valid auth should proceed."""
        response = client.post(
            '/api/run',
            json={'request': 'Create a simple Flask app'},
            headers=auth_headers
        )
        # Should not return 401
        assert response.status_code != 401


class TestWorkflowAPI:
    """Tests for workflow API endpoints."""
    
    def test_run_workflow_missing_request(self, client, auth_headers):
        """Run workflow without request should return 400."""
        response = client.post(
            '/api/run',
            json={},
            headers=auth_headers
        )
        assert response.status_code == 400
        
        data = response.get_json()
        assert data['code'] == 'VALIDATION_ERROR'
    
    def test_run_workflow_empty_request(self, client, auth_headers):
        """Run workflow with empty request should return 400."""
        response = client.post(
            '/api/run',
            json={'request': '   '},
            headers=auth_headers
        )
        assert response.status_code == 400
    
    def test_run_workflow_success(self, client, auth_headers):
        """Run workflow should succeed with valid input."""
        response = client.post(
            '/api/run',
            json={
                'request': 'Create a minimal Flask microservice with /health endpoint',
                'options': {
                    'priority': 'high'
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'workflow_id' in data
        assert data['status'] in ['completed', 'failed']


class TestOrchestrator:
    """Tests for orchestrator functionality."""
    
    def test_orchestrator_full_workflow(
        self,
        mock_llm_client,
        mock_supabase_auth,
        mock_notifier
    ):
        """Test complete workflow execution."""
        from orchestrator import Orchestrator
        
        orchestrator = Orchestrator(
            llm_client=mock_llm_client,
            supabase_auth=mock_supabase_auth,
            notifier=mock_notifier
        )
        
        result = orchestrator.execute(
            workflow_id='test-workflow-001',
            user_id='test-user-id',
            user_request='Create a Flask microservice with /health endpoint, tests, Dockerfile, README.'
        )
        
        assert result['success'] == True
        assert result['workflow_id'] == 'test-workflow-001'
    
    def test_orchestrator_logs_events(
        self,
        mock_llm_client,
        mock_supabase_auth,
        mock_notifier
    ):
        """Test that orchestrator logs events."""
        from orchestrator import Orchestrator
        
        orchestrator = Orchestrator(
            llm_client=mock_llm_client,
            supabase_auth=mock_supabase_auth,
            notifier=mock_notifier
        )
        
        orchestrator.execute(
            workflow_id='test-workflow-002',
            user_id='test-user-id',
            user_request='Create a simple app'
        )
        
        # Verify events were logged
        assert mock_supabase_auth.log_event.called


class TestAgents:
    """Tests for individual agents."""
    
    def test_ceo_agent_specification(self, mock_llm_client):
        """Test CEO agent creates specification."""
        from orchestrator.agents import CEOAgent
        
        agent = CEOAgent(mock_llm_client)
        
        result = agent.execute(
            task_id='test-task',
            phase='specification',
            inputs={'user_request': 'Create a Flask app'}
        )
        
        assert result['status'] == 'success'
        assert result['agent'] == 'ceo'
    
    def test_pm_agent_planning(self, mock_llm_client):
        """Test PM agent creates task plan."""
        from orchestrator.agents import PMAgent
        
        agent = PMAgent(mock_llm_client)
        
        result = agent.execute(
            task_id='test-task',
            phase='planning',
            inputs={
                'project_spec': {'title': 'Test', 'description': 'Test project', 'scope': 'Testing'},
                'objectives': ['Create endpoint'],
                'constraints': []
            }
        )
        
        assert result['status'] == 'success'
        assert result['agent'] == 'pm'
    
    def test_coder_agent_implementation(self, mock_llm_client):
        """Test Coder agent creates code."""
        from orchestrator.agents import CoderAgent
        
        agent = CoderAgent(mock_llm_client)
        
        result = agent.execute(
            task_id='test-task',
            phase='execution',
            inputs={
                'task_description': 'Create health endpoint',
                'acceptance_criteria': ['Returns 200'],
                'project_spec': {'title': 'Test'}
            }
        )
        
        assert result['status'] == 'success'
        assert result['agent'] == 'coder'
    
    def test_qa_agent_validation(self, mock_llm_client):
        """Test QA agent validates code."""
        from orchestrator.agents import QAAgent
        
        agent = QAAgent(mock_llm_client)
        
        result = agent.execute(
            task_id='test-task',
            phase='validation',
            inputs={
                'task_description': 'Validate code',
                'coder_output': {'files': []},
                'project_requirements': {},
                'success_criteria': []
            }
        )
        
        assert result['status'] == 'success'
        assert result['agent'] == 'qa'


class TestMemoryStore:
    """Tests for memory storage."""
    
    def test_save_and_retrieve_workflow_state(self, tmp_path):
        """Test saving and retrieving workflow state."""
        from orchestrator.tools.memory import MemoryStore
        
        db_path = tmp_path / "test_memory.db"
        store = MemoryStore(str(db_path))
        
        state = {'status': 'running', 'iteration': 1}
        store.save_workflow_state('test-workflow', state)
        
        retrieved = store.get_workflow_state('test-workflow')
        assert retrieved == state
    
    def test_conversation_history(self, tmp_path):
        """Test conversation history tracking."""
        from orchestrator.tools.memory import MemoryStore
        
        db_path = tmp_path / "test_memory.db"
        store = MemoryStore(str(db_path))
        
        store.add_conversation_message(
            'test-workflow',
            'ceo',
            'user',
            'Create a Flask app'
        )
        
        history = store.get_conversation_history('test-workflow', 'ceo')
        assert len(history) == 1
        assert history[0]['role'] == 'user'


class TestNotifier:
    """Tests for SMTP notifier."""
    
    def test_notifier_disabled_without_config(self):
        """Test notifier is disabled without config."""
        from orchestrator.tools.notifier import SMTPNotifier
        
        with patch.dict(os.environ, {}, clear=True):
            notifier = SMTPNotifier()
            assert notifier.enabled == False
    
    def test_send_workflow_started(self):
        """Test workflow started notification."""
        from orchestrator.tools.notifier import SMTPNotifier
        
        # Mock smtplib to avoid actual network calls
        with patch('smtplib.SMTP') as mock_smtp:
            env = {
                'SMTP_HOST': 'smtp.example.com',
                'SMTP_USERNAME': 'user',
                'SMTP_PASSWORD': 'pass'
            }
            with patch.dict(os.environ, env):
                notifier = SMTPNotifier()
                result = notifier.send_workflow_started(
                    'test@example.com',
                    'test-workflow',
                    'Test request'
                )
                assert result == True