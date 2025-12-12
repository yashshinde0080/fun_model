#!/usr/bin/env python3
"""
Multi-Agent Corporate System - Local Development Runner

A complete local development environment that works WITHOUT any external services.
No Supabase, OpenRouter, or SMTP required - everything is mocked locally.

Usage:
    python local.py                    # Run with mock services
    python local.py --port 8080        # Run on custom port
    python local.py --demo             # Run demo workflow automatically
    python local.py --reset            # Reset local database

Features:
    - Mock LLM responses (no API key needed)
    - Local SQLite database (no Supabase needed)
    - Mock email notifications (logged to console)
    - Auto-reload on code changes
    - Beautiful web UI
"""

import os
import sys
import json
import uuid
import time
import logging
import argparse
import threading
import webbrowser
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from functools import wraps
from contextlib import contextmanager

# =============================================================================
# Setup
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
except ImportError:
    pass

# Create directories
(PROJECT_ROOT / 'storage' / 'logs').mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / 'storage' / 'data').mkdir(parents=True, exist_ok=True)

# =============================================================================
# Logging
# =============================================================================

def setup_logging(level=logging.INFO):
    """Configure logging."""
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                PROJECT_ROOT / 'storage' / 'logs' / 'local.log',
                encoding='utf-8'
            )
        ]
    )
    
    # Reduce noise
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logging.getLogger('local')

logger = setup_logging()

# =============================================================================
# Local Database
# =============================================================================

class LocalDatabase:
    """SQLite database for local development."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(PROJECT_ROOT / 'storage' / 'data' / 'local.db')
        self._init_db()
        logger.info(f"Database initialized: {self.db_path}")
    
    @contextmanager
    def get_conn(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self.get_conn() as conn:
            conn.executescript("""
                -- Users table
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT DEFAULT 'authenticated',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Workflows table
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    input_request TEXT,
                    final_output TEXT,
                    iteration_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    error_message TEXT
                );
                
                -- Events table
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    task_id TEXT,
                    event_type TEXT NOT NULL,
                    status TEXT,
                    payload TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Artifacts table
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    agent_name TEXT,
                    artifact_type TEXT,
                    file_name TEXT,
                    content TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create default test users
                INSERT OR IGNORE INTO users (id, email, password, role)
                VALUES 
                    ('user-001', 'test@localhost', 'password123', 'authenticated'),
                    ('user-002', 'admin@localhost', 'admin123', 'admin');
            """)
    
    def reset(self):
        """Reset database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self._init_db()
        logger.info("Database reset complete")
    
    def insert(self, table: str, data: Dict[str, Any]) -> str:
        """Insert a record."""
        if 'id' not in data:
            data['id'] = str(uuid.uuid4())
        
        # Convert dicts/lists to JSON strings
        processed = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                processed[k] = json.dumps(v)
            else:
                processed[k] = v
        
        columns = ', '.join(processed.keys())
        placeholders = ', '.join(['?' for _ in processed])
        
        with self.get_conn() as conn:
            conn.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                list(processed.values())
            )
        
        return data['id']
    
    def update(self, table: str, id: str, data: Dict[str, Any]) -> bool:
        """Update a record."""
        processed = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                processed[k] = json.dumps(v)
            else:
                processed[k] = v
        
        processed['updated_at'] = datetime.now(timezone.utc).isoformat()
        
        set_clause = ', '.join([f"{k} = ?" for k in processed.keys()])
        
        with self.get_conn() as conn:
            conn.execute(
                f"UPDATE {table} SET {set_clause} WHERE id = ?",
                list(processed.values()) + [id]
            )
        return True
    
    def get(self, table: str, id: str) -> Optional[Dict[str, Any]]:
        """Get a record by ID."""
        with self.get_conn() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE id = ?", (id,)
            ).fetchone()
            return dict(row) if row else None
    
    def query(self, table: str, conditions: Dict[str, Any] = None, 
              order_by: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Query records."""
        query = f"SELECT * FROM {table}"
        params = []
        
        if conditions:
            clauses = [f"{k} = ?" for k in conditions.keys()]
            query += " WHERE " + " AND ".join(clauses)
            params = list(conditions.values())
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email."""
        with self.get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
            return dict(row) if row else None


# Global database instance
db: Optional[LocalDatabase] = None

def get_db() -> LocalDatabase:
    global db
    if db is None:
        db = LocalDatabase()
    return db

# =============================================================================
# Mock Authentication
# =============================================================================

class MockAuth:
    """Mock authentication system."""
    
    def __init__(self):
        self.db = get_db()
        self.tokens: Dict[str, Dict] = {}
        logger.info("Mock Auth initialized")
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token and return user info."""
        # Accept any token format for local dev
        if not token:
            return None
        
        # Check cached tokens
        if token in self.tokens:
            return self.tokens[token]
        
        # For simple tokens like "test-token", use default user
        if token in ('test-token', 'Bearer test-token'):
            user = {
                'sub': 'user-001',
                'email': 'test@localhost',
                'role': 'authenticated'
            }
            self.tokens[token] = user
            return user
        
        # Try to extract user ID from token
        if token.startswith('local-'):
            user_id = token.replace('local-', '')
            user_data = self.db.get('users', user_id)
            if user_data:
                user = {
                    'sub': user_data['id'],
                    'email': user_data['email'],
                    'role': user_data['role']
                }
                self.tokens[token] = user
                return user
        
        # Default to test user
        return {
            'sub': 'user-001',
            'email': 'test@localhost',
            'role': 'authenticated'
        }
    
    def login(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Login and return token."""
        user = self.db.get_user_by_email(email)
        if user and user['password'] == password:
            token = f"local-{user['id']}"
            self.tokens[token] = {
                'sub': user['id'],
                'email': user['email'],
                'role': user['role']
            }
            return {
                'access_token': token,
                'token_type': 'bearer',
                'user': {
                    'id': user['id'],
                    'email': user['email']
                }
            }
        return None
    
    def log_workflow(self, data: Dict[str, Any]) -> str:
        """Log workflow to database."""
        return self.db.insert('workflows', data)
    
    def update_workflow(self, workflow_id: str, data: Dict[str, Any]) -> bool:
        """Update workflow."""
        return self.db.update('workflows', workflow_id, data)
    
    def log_event(self, data: Dict[str, Any]) -> str:
        """Log event to database."""
        return self.db.insert('events', data)
    
    def store_artifact(self, data: Dict[str, Any]) -> str:
        """Store artifact."""
        return self.db.insert('artifacts', data)

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user details by ID."""
        user = self.db.get('users', user_id)
        if user:
            return {
                'id': user['id'],
                'email': user['email'],
                'created_at': user.get('created_at', datetime.now().isoformat()),
                'metadata': {}
            }
        return None

# =============================================================================
# Mock LLM Client
# =============================================================================

class MockLLMClient:
    """Mock LLM that generates realistic responses without any API."""
    
    def __init__(self):
        logger.info("Mock LLM Client initialized (no API key needed)")
    
    def complete(self, messages: List[Dict], model: str = "mock", **kwargs) -> Dict:
        """Generate completion."""
        content = messages[-1]['content'] if messages else ''
        time.sleep(0.3)  # Simulate API latency
        
        return {
            'content': f"Mock response for: {content[:100]}...",
            'model': 'mock-local',
            'usage': {'total_tokens': 50},
            'elapsed': 0.3
        }
    
    def complete_json(self, messages: List[Dict], model: str = "mock", **kwargs) -> Dict:
        """Generate JSON completion."""
        content = messages[-1]['content'] if messages else ''
        time.sleep(0.3)
        
        response = self._generate_response(content)
        
        return {
            'content': json.dumps(response),
            'parsed': response,
            'model': 'mock-local',
            'usage': {'total_tokens': 100},
            'elapsed': 0.3
        }
    
    def _generate_response(self, prompt: str) -> Dict:
        """Generate appropriate response based on prompt."""
        prompt_lower = prompt.lower()
        
        # Detect agent type from prompt
        if 'ceo' in prompt_lower and 'finalization' in prompt_lower:
            return self._ceo_final()
        elif 'ceo' in prompt_lower:
            return self._ceo_spec()
        elif 'pm' in prompt_lower or 'project manager' in prompt_lower:
            return self._pm_plan()
        elif 'research' in prompt_lower:
            return self._research()
        elif 'coder' in prompt_lower:
            return self._coder()
        elif 'qa' in prompt_lower:
            return self._qa()
        elif 'docs' in prompt_lower or 'documentation' in prompt_lower:
            return self._docs()
        
        return self._default()
    
    def _ceo_spec(self) -> Dict:
        return {
            'status': 'success',
            'task_id': 'ceo-spec',
            'agent': 'ceo',
            'payload': {
                'project_spec': {
                    'title': 'Flask Health Check Microservice',
                    'description': 'A minimal Flask microservice with health check endpoint',
                    'scope': 'Flask app, /health endpoint, tests, Dockerfile, README',
                    'priority': 'medium'
                },
                'objectives': [
                    'Create Flask app with /health endpoint',
                    'Write comprehensive tests',
                    'Create Dockerfile',
                    'Write documentation'
                ],
                'constraints': [
                    'Use Flask 3.0+',
                    'Python 3.11 compatible',
                    'Minimal dependencies'
                ],
                'success_criteria': [
                    '/health returns 200 OK',
                    'All tests pass',
                    'Docker builds successfully',
                    'README is complete'
                ]
            },
            'confidence': 'high',
            'meta': {'elapsed': 1.5}
        }
    
    def _ceo_final(self) -> Dict:
        return {
            'status': 'success',
            'task_id': 'ceo-final',
            'agent': 'ceo',
            'payload': {
                'project_spec': {
                    'title': 'Flask Health Check Microservice',
                    'description': 'Completed implementation',
                    'scope': 'All deliverables completed',
                    'priority': 'medium'
                },
                'objectives': ['All objectives met'],
                'constraints': ['All constraints satisfied'],
                'success_criteria': ['All criteria verified ✓'],
                'final_summary': 'Project completed successfully! All deliverables have been created including Flask application with health endpoint, comprehensive test suite, production-ready Dockerfile, and complete documentation.',
                'approval_status': 'approved'
            },
            'confidence': 'high',
            'meta': {'elapsed': 1.0}
        }
    
    def _pm_plan(self) -> Dict:
        return {
            'status': 'success',
            'task_id': 'pm-plan',
            'agent': 'pm',
            'payload': {
                'tasks': [
                    {
                        'id': 'task-001',
                        'name': 'Research Best Practices',
                        'assigned_to': 'research',
                        'description': 'Research Flask health check best practices',
                        'acceptance_criteria': ['Document best practices', 'List recommendations'],
                        'estimated_complexity': 'low'
                    },
                    {
                        'id': 'task-002',
                        'name': 'Implement Application',
                        'assigned_to': 'coder',
                        'description': 'Create Flask app with /health endpoint and tests',
                        'acceptance_criteria': ['Working endpoint', 'Tests pass', 'Docker builds'],
                        'estimated_complexity': 'medium'
                    },
                    {
                        'id': 'task-003',
                        'name': 'Quality Validation',
                        'assigned_to': 'qa',
                        'description': 'Validate code quality',
                        'acceptance_criteria': ['All checks pass'],
                        'estimated_complexity': 'low'
                    },
                    {
                        'id': 'task-004',
                        'name': 'Documentation',
                        'assigned_to': 'docs',
                        'description': 'Create README and docs',
                        'acceptance_criteria': ['Complete README'],
                        'estimated_complexity': 'low'
                    }
                ],
                'dependencies': {
                    'task-002': ['task-001'],
                    'task-003': ['task-002'],
                    'task-004': ['task-003']
                },
                'execution_order': ['task-001', 'task-002', 'task-003', 'task-004'],
                'timeline_estimate': '1-2 hours',
                'risk_assessment': []
            },
            'confidence': 'high',
            'meta': {'elapsed': 1.0}
        }
    
    def _research(self) -> Dict:
        return {
            'status': 'success',
            'task_id': 'task-001',
            'agent': 'research',
            'payload': {
                'summary': 'Health endpoints should return JSON with status, version, timestamp. Use 200 for healthy, 503 for unhealthy.',
                'findings': [
                    {
                        'topic': 'Response Format',
                        'content': 'Use JSON with status, version, timestamp fields',
                        'relevance': 'Standard health check format'
                    },
                    {
                        'topic': 'Production Server',
                        'content': 'Use Gunicorn with 2-4 workers',
                        'relevance': 'Production deployment'
                    }
                ],
                'citations': [
                    {
                        'source': 'Flask Documentation',
                        'reference': 'https://flask.palletsprojects.com/',
                        'accessed_date': datetime.now().strftime('%Y-%m-%d')
                    }
                ],
                'recommendations': [
                    'Use Flask application factory pattern',
                    'Include version in health response',
                    'Use python:3.11-slim for Docker'
                ]
            },
            'confidence': 'high',
            'meta': {'elapsed': 0.8}
        }
    
    def _coder(self) -> Dict:
        return {
            'status': 'success',
            'task_id': 'task-002',
            'agent': 'coder',
            'payload': {
                'files': [
                    {
                        'path': 'app.py',
                        'content': '''"""Flask Health Check Microservice."""
from datetime import datetime, timezone
from flask import Flask, jsonify

app = Flask(__name__)
VERSION = "1.0.0"

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': VERSION,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200

@app.route('/')
def root():
    """Root endpoint."""
    return jsonify({
        'service': 'health-microservice',
        'version': VERSION,
        'endpoints': {
            '/': 'Service info',
            '/health': 'Health check'
        }
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
''',
                        'type': 'source',
                        'language': 'python'
                    },
                    {
                        'path': 'test_app.py',
                        'content': '''"""Tests for health microservice."""
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

def test_health_status_code(client):
    """Test /health returns 200."""
    response = client.get('/health')
    assert response.status_code == 200

def test_health_json_structure(client):
    """Test /health returns correct JSON."""
    response = client.get('/health')
    data = response.get_json()
    assert 'status' in data
    assert data['status'] == 'healthy'
    assert 'version' in data
    assert 'timestamp' in data

def test_root_status_code(client):
    """Test / returns 200."""
    response = client.get('/')
    assert response.status_code == 200

def test_root_has_service_name(client):
    """Test / has service name."""
    response = client.get('/')
    data = response.get_json()
    assert 'service' in data
''',
                        'type': 'test',
                        'language': 'python'
                    },
                    {
                        'path': 'requirements.txt',
                        'content': 'flask>=3.0.0\npytest>=7.4.0\ngunicorn>=21.0.0',
                        'type': 'config',
                        'language': 'text'
                    },
                    {
                        'path': 'Dockerfile',
                        'content': '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
''',
                        'type': 'dockerfile',
                        'language': 'dockerfile'
                    }
                ],
                'run_instructions': {
                    'install': 'pip install -r requirements.txt',
                    'run': 'python app.py',
                    'test': 'pytest -v',
                    'docker_build': 'docker build -t health-service .',
                    'docker_run': 'docker run -p 5000:5000 health-service'
                },
                'dependencies': ['flask>=3.0.0', 'pytest>=7.4.0', 'gunicorn>=21.0.0'],
                'notes': 'Complete Flask health check microservice with tests and Docker support.'
            },
            'confidence': 'high',
            'meta': {'elapsed': 2.0}
        }
    
    def _qa(self) -> Dict:
        return {
            'status': 'success',
            'task_id': 'task-003',
            'agent': 'qa',
            'payload': {
                'validation_results': [
                    {'check': 'Code Completeness', 'passed': True, 'details': 'All files present', 'severity': 'info'},
                    {'check': 'Syntax Correctness', 'passed': True, 'details': 'Valid Python', 'severity': 'info'},
                    {'check': 'Test Coverage', 'passed': True, 'details': '4 tests found', 'severity': 'info'},
                    {'check': 'Dockerfile Valid', 'passed': True, 'details': 'Valid syntax', 'severity': 'info'},
                    {'check': 'Dependencies Listed', 'passed': True, 'details': 'requirements.txt complete', 'severity': 'info'}
                ],
                'approval_status': 'approved',
                'issues_found': [],
                'fix_suggestions': [],
                'test_coverage_estimate': '85%'
            },
            'confidence': 'high',
            'meta': {'elapsed': 1.0}
        }
    
    def _docs(self) -> Dict:
        return {
            'status': 'success',
            'task_id': 'task-004',
            'agent': 'docs',
            'payload': {
                'readme': '''# Flask Health Check Microservice

A minimal, production-ready Flask microservice with health check endpoint.

## Features

- `/health` endpoint returning JSON status
- Docker support with health checks
- Comprehensive test suite
- Production-ready with Gunicorn

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Run tests
pytest -v
```

## Files
- app.py: Flask application
- test_app.py: Pytest tests
- requirements.txt: Dependencies
- Dockerfile: Containerization

## Docker

Build:
```bash
docker build -t health-service .
```

Run:
```bash
docker run -p 5000:5000 health-service
```
''',
            },
            'confidence': 'high',
            'meta': {'elapsed': 0.5}
        }


# =============================================================================
# Mock Notifier
# =============================================================================

class MockNotifier:
    """Mock notifier that logs to console."""
    
    def __init__(self):
        logger.info("Mock Notifier initialized")
    
    def send_workflow_started(self, email: str, workflow_id: str, request_summary: str):
        logger.info(f"📧 Notification to {email}: Workflow {workflow_id} started. Request: {request_summary}")
    
    def send_workflow_completed(self, email: str, workflow_id: str, duration: str, iterations: int):
        logger.info(f"📧 Notification to {email}: Workflow {workflow_id} completed in {duration} ({iterations} iterations)")
    
    def send_workflow_failed(self, email: str, workflow_id: str, error_message: str):
        logger.info(f"📧 Notification to {email}: Workflow {workflow_id} failed. Error: {error_message}")


# =============================================================================
# Service Factories
# =============================================================================

def get_auth_service(config: dict):
    """Get authentication service (Real or Mock)."""
    if config.get('SUPABASE_URL') and config.get('SUPABASE_SERVICE_ROLE_KEY'):
        try:
            from app.auth import SupabaseAuth
            logger.info("Using REAL Supabase Auth")
            return SupabaseAuth(
                url=config['SUPABASE_URL'],
                anon_key=config['SUPABASE_ANON_KEY'],
                service_role_key=config['SUPABASE_SERVICE_ROLE_KEY']
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Supabase Auth: {e}. Falling back to Mock.")
    
    logger.info("Using MOCK Auth")
    return MockAuth()

def get_llm_service(config: dict):
    """Get LLM service (Real or Mock)."""
    if config.get('OPENROUTER_API_KEY'):
        try:
            from orchestrator.llm_client import OpenRouterClient
            logger.info("Using REAL OpenRouter LLM")
            return OpenRouterClient(api_key=config['OPENROUTER_API_KEY'])
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter: {e}. Falling back to Mock.")
    
    logger.info("Using MOCK LLM")
    return MockLLMClient()

def get_notifier_service(config: dict):
    """Get notifier service (Real or Mock)."""
    # Simply use MockNotifier for now as SMTP config is often missing in dev
    # You could add real SMTP check here if desired
    return MockNotifier()


# =============================================================================
# Main Application
# =============================================================================

def create_local_app():
    """Create local Flask application."""
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / 'app' / 'templates'),
        static_folder=str(PROJECT_ROOT / 'app' / 'static')
    )
    
    # Configuration
    app.config['SECRET_KEY'] = 'local-dev-secret'
    app.config['SUPABASE_URL'] = os.getenv('SUPABASE_URL')
    app.config['SUPABASE_ANON_KEY'] = os.getenv('SUPABASE_ANON_KEY')
    app.config['SUPABASE_SERVICE_ROLE_KEY'] = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    app.config['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')
    
    # Initialize services
    app.supabase_auth = get_auth_service(app.config)
    app.llm_client = get_llm_service(app.config)
    app.notifier = get_notifier_service(app.config)
    
    # Initialize Orchestrator
    from orchestrator.orchestrator import Orchestrator
    app.orchestrator = Orchestrator(
        llm_client=app.llm_client,
        supabase_auth=app.supabase_auth,
        notifier=app.notifier
    )
    
    # Register Blueprints
    from app.routes import main_bp, api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Add Local Login Endpoint (For Mock Mode UI)
    @app.route('/api/auth/login', methods=['POST'])
    def local_login():
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        # If using real Supabase, this endpoint is less useful unless we proxy it,
        # but for Mock mode it's essential.
        # If app.supabase_auth is an instance of MockAuth, we can call login.
        if isinstance(app.supabase_auth, MockAuth):
            result = app.supabase_auth.login(email, password)
            if result:
                return jsonify(result)
            return jsonify({'error': 'Invalid credentials'}), 401
        
        return jsonify({'error': 'Local login only available in mock mode'}), 400

    CORS(app)
    return app

def main():
    parser = argparse.ArgumentParser(description='Run local development server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on')
    parser.add_argument('--reset', action='store_true', help='Reset local database')
    args = parser.parse_args()
    
    if args.reset:
        get_db().reset()
        logger.info("Database reset")
        return

    app = create_local_app()
    logger.info(f"Starting Flask server on http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=True)

if __name__ == '__main__':
    main()
