"""
Supabase Authentication Middleware and Helpers
"""

import os
import logging
import uuid
from functools import wraps
from typing import Optional, Dict, Any, Callable, List
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

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow from Supabase."""
        try:
            if not self.admin_client:
                return None
            response = self.admin_client.table('workflows').select('*').eq('id', workflow_id).single().execute()
            return response.data if response.data else None
        except Exception as e:
            logger.error(f"Error fetching workflow: {e}")
            return None

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

    def get_workflow_events(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get events for a workflow."""
        try:
            if not self.admin_client:
                return []
            response = self.admin_client.table('events').select('*').eq('workflow_id', workflow_id).order('created_at').execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"Error fetching workflow events: {e}")
            return []


class MockSupabaseAuth:
    """Mock Supabase authentication handler for development with file persistence."""

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    DB_FILE = 'storage/mock_db.json'

    def __init__(self):
        logger.info("Mock Supabase auth client initialized (DEV MODE)")
        self.users = {}
        self.tokens = {}
        self.workflows = {}
        self.events = []
        self.artifacts = []

        self._load()
        
        # Ensure at least one test user
        if not self.users:
            self.users['test-user-id'] = {
                'id': 'test-user-id',
                'email': 'test@localhost',
                'password': 'password123',
                'role': 'authenticated',
                'user_metadata': {'name': 'Dev User'}
            }
            self._save()

    def _load(self):
        """Load data from JSON file."""
        if os.path.exists(self.DB_FILE):
            try:
                import json
                with open(self.DB_FILE, 'r') as f:
                    data = json.load(f)
                    self.users = data.get('users', {})
                    self.workflows = data.get('workflows', {})
                    self.events = data.get('events', [])
                    self.artifacts = data.get('artifacts', [])
                logger.info(f"Loaded mock DB from {self.DB_FILE}")
            except Exception as e:
                logger.error(f"Failed to load mock DB: {e}")

    def _save(self):
        """Save data to JSON file."""
        try:
            import json
            data = {
                'users': self.users,
                'workflows': self.workflows,
                'events': self.events,
                'artifacts': self.artifacts
            }
            # Ensure folder exists
            os.makedirs(os.path.dirname(self.DB_FILE), exist_ok=True)
            with open(self.DB_FILE, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save mock DB: {e}")

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Mock token verification."""
        logger.debug(f"Verifying token: {token[:10]}...")
        
        # Check if known token (in-memory)
        if token in self.tokens:
            return self.tokens[token]

        # Support legacy test token
        if token == 'test-token':
            return {
                'sub': 'test-user-id',
                'email': 'dev@example.com',
                'role': 'authenticated'
            }
        
        # Check for local persistent tokens
        if token.startswith('local-'):
            user_id = token.replace('local-', '')
            # Try to get user from memory, then reload if needed
            user = self.get_user_by_id(user_id)
            if not user:
                logger.debug(f"User {user_id} not found in memory, reloading DB...")
                self._load()
                user = self.get_user_by_id(user_id)
            
            if user:
                return {
                    'sub': user['id'],
                    'email': user['email'],
                    'role': user.get('role', 'authenticated')
                }
            else:
                 logger.warning(f"User {user_id} not found after reload.")

        logger.warning("Token verification failed")
        return None

    def login(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Mock Login."""
        logger.debug(f"Attempting login for {email}")
        for user in self.users.values():
            if user['email'] == email:
                if user.get('password') == password:
                    token = f"local-{user['id']}"
                    self.tokens[token] = {
                        'sub': user['id'],
                        'email': user['email'],
                        'role': user.get('role', 'authenticated')
                    }
                    logger.info(f"Login successful for {email}")
                    return {
                        'access_token': token,
                        'token_type': 'bearer',
                        'user': user
                    }
                else:
                    logger.warning(f"Invalid password for {email}")
                    return None
        logger.warning(f"User {email} not found during login")
        return None

    def reset_password_for_email(self, email: str) -> bool:
        """Mock Reset Password."""
        logger.debug(f"Attempting reset for {email}")
        for user in self.users.values():
            if user['email'] == email:
                # Reset to default
                user['password'] = 'password123'
                self._save()
                logger.info(f"Password reset to 'password123' for {email}")
                return True
        logger.warning(f"User {email} not found for reset")
        return False

    def signup(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Mock Signup."""
        logger.debug(f"Attempting signup for {email}")
        # Check existing
        for user in self.users.values():
            if user['email'] == email:
                logger.warning(f"Signup failed: User {email} already exists")
                return {'error': 'User already exists'}

        try:
            user_id = str(uuid.uuid4())
            new_user = {
                'id': user_id,
                'email': email,
                'password': password,
                'role': 'authenticated',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'user_metadata': {}
            }
            self.users[user_id] = new_user
            self._save()
            logger.info(f"Signup successful for {email} (id: {user_id})")
            
            return self.login(email, password)
        except Exception as e:
            logger.error(f"Signup error: {e}")
            return {'error': 'Internal server error during signup'}

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get mock user."""
        return self.users.get(user_id)

    def log_workflow(self, workflow_data: Dict[str, Any]) -> Optional[str]:
        """Log mock workflow."""
        w_id = workflow_data.get('id')
        self.workflows[w_id] = workflow_data
        logger.debug(f"[MOCK] Logged workflow {w_id}")
        self._save()
        return w_id

    def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> bool:
        """Update mock workflow."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].update(updates)
            logger.debug(f"[MOCK] Updated workflow {workflow_id}")
            self._save()
            return True
        return False

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get mock workflow."""
        return self.workflows.get(workflow_id)

    def log_event(self, event_data: Dict[str, Any]) -> Optional[str]:
        """Log mock event."""
        self.events.append(event_data)
        logger.debug(f"[MOCK] Logged event: {event_data.get('event_type')}")
        self._save()
        return str(len(self.events))

    def get_workflow_events(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get mock workflow events."""
        return [e for e in self.events if e.get('workflow_id') == workflow_id]

    def store_artifact(self, artifact_data: Dict[str, Any]) -> Optional[str]:
        """Store mock artifact."""
        self.artifacts.append(artifact_data)
        logger.debug(f"[MOCK] Stored artifact")
        self._save()
        return str(len(self.artifacts))


def require_auth(f: Callable) -> Callable:
    """Decorator to require authentication for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')

        # Check for Dev Mode bypass
        if current_app.config.get('DEV_MODE') and not auth_header:
             # In dev mode, if no header, default to test user if configured,
             # OR still require header but accept the test token.
             # Let's enforce header even in dev mode for consistency, but allow 'Bearer test-token'
             pass

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

        supabase_auth = current_app.supabase_auth
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
        # Allow if dev mode
        if current_app.config.get('DEV_MODE'):
            return f(*args, **kwargs)

        if not hasattr(g, 'user_role') or g.user_role != 'admin':
            return jsonify({
                'error': 'Admin access required',
                'code': 'AUTH_FORBIDDEN'
            }), 403
        return f(*args, **kwargs)
    return decorated_function
