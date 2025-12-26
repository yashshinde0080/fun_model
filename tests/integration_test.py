"""
Integration Tests for Multi-Agent Corporate System
"""

import os
import json
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from langchain_core.messages import AIMessage

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
def mock_llm_response_content():
    """Helper to generate mock LLM content based on prompt."""
    def _generate(messages):
        content = str(messages)

        # 1. Clarification Gate
        if "Agile Input Analyzer" in content:
             return json.dumps({
                 "status": "clear",
                 "intent": "Build a Flask app",
                 "questions": []
             })

        # 2. CEO Specification
        if "CEO Agent" in content:
            return json.dumps({
                "requirements": ["Functional requirements here"],
                "acceptance_criteria": ["Criteria 1"],
                "assumptions": ["Assumption 1"]
            })

        # 3. Research
        if "Research Agent" in content:
            return json.dumps({
                "summary": "Use Flask and Supabase",
                "findings": [
                    {"topic": "Tech Stack", "content": "Python + Flask is good"}
                ],
                "recommendations": ["Use Docker"]
            })

        # 4. PM Planning
        if "Project Manager" in content:
             return json.dumps({
                "backlog": [
                    {"id": "1", "task": "Setup Project Structure"},
                    {"id": "2", "task": "Implement Health Check"}
                ],
                "definition_of_done": "Code passes tests"
            })

        # 5. Coder
        if "Senior Coder" in content:
             return json.dumps({
                 "files": {
                     "src/app.py": "from flask import Flask\napp = Flask(__name__)\n@app.route('/health')\ndef h(): return 'ok'",
                     "tests/test_app.py": "def test_h(): assert True",
                     "requirements.txt": "flask"
                 }
             })

        # 6. QA
        if "QA Lead" in content:
             return json.dumps({
                 "status": "PASS",
                 "feedback": []
             })

        # 7. Docs
        if "Generate comprehensive README" in content:
             return "# Project README\n\nThis is a test project."

        return "{}"
    return _generate


@pytest.fixture
def mock_chat_openai(mock_llm_response_content):
    """Mock ChatOpenAI to avoid real network calls."""
    with patch('orchestrator.graph.ChatOpenAI') as MockClass:
        mock_instance = MockClass.return_value

        def side_effect(messages, **kwargs):
            content = mock_llm_response_content(messages)
            return AIMessage(content=content)

        mock_instance.invoke.side_effect = side_effect
        yield MockClass


@pytest.fixture
def mock_notifier():
    """Create mock notifier."""
    mock = Mock()
    mock.send_workflow_started.return_value = True
    mock.send_workflow_completed.return_value = True
    mock.send_workflow_failed.return_value = True
    return mock


@pytest.fixture
def app(mock_supabase_auth, mock_chat_openai, mock_notifier):
    """Create Flask test application."""
    # We also mock OpenRouterClient just in case, though graph uses ChatOpenAI
    with patch('app.auth.SupabaseAuth', return_value=mock_supabase_auth), \
         patch('orchestrator.llm_client.OpenRouterClient'), \
         patch('orchestrator.tools.notifier.SMTPNotifier', return_value=mock_notifier):

            from app import create_app
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


class TestAuthentication:
    """Tests for authentication."""

    def test_missing_auth_header(self, client):
        """Request without auth header should return 401."""
        response = client.post('/api/run', json={'request': 'test'})
        assert response.status_code == 401

    def test_valid_auth(self, client, auth_headers):
        """Request with valid auth should proceed."""
        response = client.post(
            '/api/run',
            json={'request': 'Create a simple Flask app'},
            headers=auth_headers
        )
        assert response.status_code != 401


class TestOrchestrator:
    """Tests for orchestrator functionality."""

    def test_orchestrator_full_workflow(
        self,
        mock_chat_openai,
        mock_supabase_auth,
        mock_notifier
    ):
        """Test complete workflow execution."""
        from orchestrator import Orchestrator

        # We don't need real keys because we patched ChatOpenAI
        orchestrator = Orchestrator(
            supabase_auth=mock_supabase_auth,
            notifier=mock_notifier
        )

        result = orchestrator.execute(
            workflow_id='test-workflow-001',
            user_id='test-user-id',
            user_request='Create a Flask microservice'
        )

        if not result['success']:
             print(f"Workflow Failed: {result.get('error')}")

        assert result['success'] == True
        assert result['workflow_id'] == 'test-workflow-001'

        # Verify research findings are in the context/output
        # The result data structure depends on _compile_result
        # Research is part of the flow but strictly speaking _compile_result only maps specific fields
        # But we can assert that the graph ran through the nodes by checking side_effect calls or logs
        # For now, just success is good.
