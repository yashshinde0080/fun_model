"""
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
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max request size
        DEV_MODE=os.getenv('DEV_MODE', 'false').lower() == 'true'
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
    from app.auth import SupabaseAuth, MockSupabaseAuth
    from orchestrator.llm_client import OpenRouterClient
    from orchestrator.tools.notifier import SMTPNotifier
    from orchestrator.orchestrator import Orchestrator

    # Initialize Supabase auth
    if app.config.get('DEV_MODE'):
        logger.warning("Initializing MockSupabaseAuth for DEV_MODE")
        app.supabase_auth = MockSupabaseAuth()
    else:
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
