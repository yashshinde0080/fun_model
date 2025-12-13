"""
Flask Routes - Main UI and API Endpoints
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

import threading
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
    
    # If there is a final output with events, use that preferentially as it might be more complete
    # but for simplicity, we just return what we have.
    
    return jsonify(response)
