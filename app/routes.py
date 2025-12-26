"""
Flask Routes - Main UI and API Endpoints
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

import threading
from flask import Blueprint, render_template, request, jsonify, g, current_app

from flask import Blueprint, render_template, request, jsonify, g, current_app
from app.auth import require_auth, require_admin, MockSupabaseAuth

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__)


@main_bp.route('/')
def index():
    """Main UI page."""
    # If in DEV_MODE, force frontend Mock Mode by withholding real attributes
    if current_app.config.get('DEV_MODE'):
        return render_template('index.html', supabase_url='', supabase_anon_key='')
    
    return render_template('index.html',
                         supabase_url=current_app.config.get('SUPABASE_URL'),
                         supabase_anon_key=current_app.config.get('SUPABASE_ANON_KEY'))


@main_bp.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '1.0.0'
    })


def run_async_workflow(app, workflow_id, user_id, user_request, options, notify_email):
    """Run workflow in a background thread."""
    with app.app_context():
        try:
            logger.info(f"Starting async workflow {workflow_id}")
            orchestrator = app.orchestrator
            supabase_auth = app.supabase_auth
            notifier = app.notifier

            result = orchestrator.execute(
                workflow_id=workflow_id,
                user_id=user_id,
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

        except Exception as e:
            logger.exception(f"Async workflow error: {e}")
            # Try to update status if possible
            try:
                app.supabase_auth.update_workflow(workflow_id, {
                    'status': 'failed',
                    'final_output': {'error': str(e)}
                })
            except:
                pass


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
        # Create initial record
        supabase_auth.log_workflow({
            'id': workflow_id,
            'user_id': g.user_id,
            'status': 'pending',
            'created_at': datetime.now(timezone.utc).isoformat(),
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

        # Start async execution
        # We need to capture the real app object to pass to the thread (current_app is a proxy)
        app = current_app._get_current_object()

        thread = threading.Thread(
            target=run_async_workflow,
            args=(app, workflow_id, g.user_id, user_request, options, notify_email)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'workflow_id': workflow_id,
            'status': 'pending',
            'message': 'Workflow started successfully'
        }), 202

    except Exception as e:
        logger.exception(f"Error initiating workflow: {e}")
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR',
            'message': str(e)
        }), 500


@api_bp.route('/workflows', methods=['GET'])
@require_auth
def list_workflows():
    """List workflows for the authenticated user."""
    # This is a stub, currently MockSupabase doesn't support listing nicely but we can return empty
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
    supabase_auth = current_app.supabase_auth
    workflow = supabase_auth.get_workflow(workflow_id)

    if not workflow:
        return jsonify({
            'error': 'Workflow not found',
            'code': 'NOT_FOUND'
        }), 404

    # Also fetch events
    events = supabase_auth.get_workflow_events(workflow_id)

    # If the workflow is finished, the result might already contain events, but for running ones we need them
    response = {
        'workflow': workflow,
        'events': events
    }

    return jsonify(response)


@api_bp.route('/logs', methods=['GET'])
def get_logs():
    """Get recent application logs."""
    try:
        from pathlib import Path
        import os
        
        # Find latest log file
        log_dir = Path('storage/logs')
        if not log_dir.exists():
            return "No logs directory found."
        
        log_files = sorted(log_dir.glob('app_*.log'), key=os.path.getmtime, reverse=True)
        if not log_files:
            return "No log files found."
            
        # Read latest file
        latest_log = log_files[0]
        with open(latest_log, 'r', encoding='utf-8') as f:
            content = f.read()
            # Limit to last 50KB
            if len(content) > 50000:
                content = "...[truncated]...\n" + content[-50000:]
            return content
    except Exception as e:
        return f"Error reading logs: {str(e)}"


@api_bp.route('/reset', methods=['POST'])
@require_auth
def reset_db():
    """Reset the database (Dev Mode only)."""
    if not current_app.config.get('DEV_MODE'):
        return jsonify({'error': 'Reset only allowed in DEV_MODE'}), 403
        
    auth = current_app.supabase_auth
    # Check if it has internal storage (MockSupabaseAuth)
    if hasattr(auth, 'workflows') and isinstance(auth.workflows, dict):
        auth.workflows = {}
        auth.events = []
        auth.artifacts = []
        # Keep users
        if hasattr(auth, '_save'):
            auth._save()
        return jsonify({'status': 'reset_complete'})
    else:
        return jsonify({'error': 'Reset not supported for this DB backend'}), 400


@api_bp.route('/auth/login', methods=['POST'])
def local_login():
    """Local login endpoint for Mock Mode."""
    # Only allow in DEV_MODE
    if not current_app.config.get('DEV_MODE') or not isinstance(current_app.supabase_auth, MockSupabaseAuth):
         return jsonify({'error': 'Endpoint unavailable in production'}), 404

    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    result = current_app.supabase_auth.login(email, password)
    if result:
        return jsonify(result)
    return jsonify({'error': 'Invalid credentials'}), 401


@api_bp.route('/auth/signup', methods=['POST'])
def local_signup():
    """Local signup endpoint for Mock Mode."""
    # Only allow in DEV_MODE
    if not current_app.config.get('DEV_MODE') or not isinstance(current_app.supabase_auth, MockSupabaseAuth):
         return jsonify({'error': 'Endpoint unavailable in production'}), 404

    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    result = current_app.supabase_auth.signup(email, password)
    if result and not result.get('error'):
        return jsonify(result)
    return jsonify({'error': result.get('error', 'Signup failed')}), 400


@api_bp.route('/auth/reset-password', methods=['POST'])
def local_reset_password():
    """Local reset password endpoint for Mock Mode."""
    # Only allow in DEV_MODE
    if not current_app.config.get('DEV_MODE') or not isinstance(current_app.supabase_auth, MockSupabaseAuth):
         return jsonify({'error': 'Endpoint unavailable in production'}), 404

    data = request.get_json()
    email = data.get('email')
    
    if current_app.supabase_auth.reset_password_for_email(email):
        return jsonify({'message': 'Password reset successful', 'data': {}})
    return jsonify({'error': 'User not found'}), 404
