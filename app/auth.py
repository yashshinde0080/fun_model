"""
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


class MockSupabaseAuth:
    """Mock Supabase authentication handler for development."""
    
    def __init__(self):
        logger.info("Mock Supabase auth client initialized (DEV MODE)")
        self.users = {
            'test-user': {
                'id': 'test-user-id',
                'email': 'dev@example.com',
                'role': 'authenticated',
                'user_metadata': {'name': 'Dev User'}
            }
        }
        self.workflows = {}
        self.events = []
        self.artifacts = []

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Mock token verification."""
        if token == 'test-token':
            return {
                'sub': 'test-user-id',
                'email': 'dev@example.com',
                'role': 'authenticated'
            }
        return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get mock user."""
        for user in self.users.values():
            if user['id'] == user_id:
                return user
        return None

    def log_workflow(self, workflow_data: Dict[str, Any]) -> Optional[str]:
        """Log mock workflow."""
        w_id = workflow_data.get('id')
        self.workflows[w_id] = workflow_data
        logger.debug(f"[MOCK] Logged workflow {w_id}")
        return w_id

    def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> bool:
        """Update mock workflow."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].update(updates)
            logger.debug(f"[MOCK] Updated workflow {workflow_id}")
            return True
        return False

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get mock workflow."""
        return self.workflows.get(workflow_id)

    def log_event(self, event_data: Dict[str, Any]) -> Optional[str]:
        """Log mock event."""
        self.events.append(event_data)
        logger.debug(f"[MOCK] Logged event: {event_data.get('event_type')}")
        return str(len(self.events))

    def store_artifact(self, artifact_data: Dict[str, Any]) -> Optional[str]:
        """Store mock artifact."""
        self.artifacts.append(artifact_data)
        logger.debug(f"[MOCK] Stored artifact")
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
