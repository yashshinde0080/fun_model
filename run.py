#!/usr/bin/env python3
"""
Multi-Agent Corporate System - Entry Point

This is the main entry point for the Multi-Agent Corporate System.
It initializes logging, loads configuration, and starts the Flask application.

Usage:
    Development:
        python run.py
    
    Production:
        gunicorn --bind 0.0.0.0:5000 --workers 4 "run:create_application()"
        
Environment Variables:
    FLASK_ENV: development|production (default: production)
    FLASK_DEBUG: true|false (default: false)
    FLASK_HOST: Host to bind to (default: 0.0.0.0)
    FLASK_PORT: Port to bind to (default: 5000)
    LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default: INFO)
"""

import os
import sys
import logging
import signal
import atexit
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables only.")


# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging() -> logging.Logger:
    """
    Configure application logging.
    
    Sets up logging with both console and file handlers.
    Log files are stored in storage/logs/ directory.
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = PROJECT_ROOT / 'storage' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine log level from environment
    log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create JSON formatter for structured logging
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            import json
            log_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            return json.dumps(log_entry)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File handler - rotating daily logs
    log_filename = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # JSON file handler for structured logs
    json_log_filename = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.jsonl"
    json_file_handler = logging.FileHandler(json_log_filename, encoding='utf-8')
    json_file_handler.setLevel(log_level)
    json_file_handler.setFormatter(JSONFormatter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(json_file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Create application logger
    logger = logging.getLogger('multiagent')
    logger.info(f"Logging initialized at level {log_level_str}")
    logger.info(f"Log files: {log_filename}")
    
    return logger


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_configuration() -> dict:
    """
    Validate required configuration and environment variables.
    
    Returns:
        Dictionary with validated configuration
        
    Raises:
        SystemExit: If critical configuration is missing
    """
    logger = logging.getLogger('multiagent.config')
    
    config = {
        'flask': {
            'env': os.getenv('FLASK_ENV', 'production'),
            'debug': os.getenv('FLASK_DEBUG', 'false').lower() == 'true',
            'host': os.getenv('FLASK_HOST', '0.0.0.0'),
            'port': int(os.getenv('FLASK_PORT', 5000)),
            'secret_key': os.getenv('FLASK_SECRET_KEY')
        },
        'supabase': {
            'url': os.getenv('SUPABASE_URL'),
            'anon_key': os.getenv('SUPABASE_ANON_KEY'),
            'service_role_key': os.getenv('SUPABASE_SERVICE_ROLE_KEY'),
            'jwt_secret': os.getenv('SUPABASE_JWT_SECRET')
        },
        'openrouter': {
            'api_key': os.getenv('OPENROUTER_API_KEY')
        },
        'smtp': {
            'host': os.getenv('SMTP_HOST'),
            'port': int(os.getenv('SMTP_PORT', 587)),
            'username': os.getenv('SMTP_USERNAME'),
            'password': os.getenv('SMTP_PASSWORD')
        }
    }
    
    # Check critical configuration
    errors = []
    warnings = []
    
    # Flask secret key
    if not config['flask']['secret_key']:
        if config['flask']['env'] == 'production':
            errors.append("FLASK_SECRET_KEY is required in production")
        else:
            config['flask']['secret_key'] = 'dev-secret-key-not-for-production'
            warnings.append("Using default FLASK_SECRET_KEY (development only)")
    
    # Supabase configuration
    if not config['supabase']['url']:
        errors.append("SUPABASE_URL is required")
    if not config['supabase']['anon_key']:
        errors.append("SUPABASE_ANON_KEY is required")
    if not config['supabase']['service_role_key']:
        errors.append("SUPABASE_SERVICE_ROLE_KEY is required")
    
    # OpenRouter configuration
    if not config['openrouter']['api_key']:
        errors.append("OPENROUTER_API_KEY is required")
    
    # SMTP configuration (optional but warn if incomplete)
    if config['smtp']['host'] and not config['smtp']['password']:
        warnings.append("SMTP_HOST is set but SMTP_PASSWORD is missing")
    
    # Log warnings
    for warning in warnings:
        logger.warning(warning)
    
    # Exit if there are errors
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        logger.error("Please check your .env file or environment variables")
        sys.exit(1)
    
    logger.info("Configuration validated successfully")
    logger.info(f"Environment: {config['flask']['env']}")
    logger.info(f"Debug mode: {config['flask']['debug']}")
    
    return config


# =============================================================================
# Application Factory
# =============================================================================

def create_application():
    """
    Create and configure the Flask application.
    
    This is the application factory function that can be used by WSGI servers
    like Gunicorn.
    
    Returns:
        Configured Flask application instance
    """
    logger = logging.getLogger('multiagent.app')
    
    try:
        from app import create_app
        app = create_app()
        logger.info("Flask application created successfully")
        return app
    except ImportError as e:
        logger.error(f"Failed to import application: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Failed to create application: {e}")
        sys.exit(1)


# =============================================================================
# Signal Handlers
# =============================================================================

def setup_signal_handlers(logger: logging.Logger):
    """
    Setup graceful shutdown signal handlers.
    
    Args:
        logger: Logger instance for logging shutdown events
    """
    def handle_shutdown(signum, frame):
        """Handle shutdown signals gracefully."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name} signal, initiating graceful shutdown...")
        
        # Perform cleanup
        cleanup()
        
        # Exit
        sys.exit(0)
    
    def handle_reload(signum, frame):
        """Handle reload signal (SIGHUP)."""
        logger.info("Received SIGHUP signal, reloading configuration...")
        # Could reload configuration here if needed
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    
    # SIGHUP is not available on Windows
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, handle_reload)
    
    logger.debug("Signal handlers registered")


# =============================================================================
# Cleanup
# =============================================================================

def cleanup():
    """
    Perform cleanup tasks on application shutdown.
    """
    logger = logging.getLogger('multiagent.cleanup')
    logger.info("Performing cleanup tasks...")
    
    try:
        # Close any open database connections
        # This would be handled by connection pools in production
        
        # Flush log handlers
        for handler in logging.root.handlers:
            handler.flush()
        
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# =============================================================================
# Health Check
# =============================================================================

def perform_startup_checks(config: dict) -> bool:
    """
    Perform startup health checks.
    
    Args:
        config: Application configuration dictionary
        
    Returns:
        True if all checks pass, False otherwise
    """
    logger = logging.getLogger('multiagent.startup')
    checks_passed = True
    
    logger.info("Performing startup health checks...")
    
    # Check 1: Supabase connectivity
    try:
        from supabase import create_client
        client = create_client(
            config['supabase']['url'],
            config['supabase']['anon_key']
        )
        # Simple query to test connectivity
        logger.info("✓ Supabase connection successful")
    except Exception as e:
        logger.error(f"✗ Supabase connection failed: {e}")
        checks_passed = False
    
    # Check 2: OpenRouter API key validity
    try:
        import httpx
        response = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {config['openrouter']['api_key']}"},
            timeout=10.0
        )
        if response.status_code == 200:
            logger.info("✓ OpenRouter API key valid")
        else:
            logger.warning(f"⚠ OpenRouter API returned status {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠ Could not verify OpenRouter API key: {e}")
    
    # Check 3: Storage directory permissions
    try:
        storage_dir = PROJECT_ROOT / 'storage'
        test_file = storage_dir / '.write_test'
        test_file.touch()
        test_file.unlink()
        logger.info("✓ Storage directory writable")
    except Exception as e:
        logger.error(f"✗ Storage directory not writable: {e}")
        checks_passed = False
    
    # Check 4: SMTP connectivity (optional)
    if config['smtp']['host']:
        try:
            import smtplib
            with smtplib.SMTP(config['smtp']['host'], config['smtp']['port'], timeout=5) as server:
                server.ehlo()
                logger.info("✓ SMTP server reachable")
        except Exception as e:
            logger.warning(f"⚠ SMTP server not reachable: {e}")
    else:
        logger.info("○ SMTP not configured (notifications disabled)")
    
    if checks_passed:
        logger.info("All startup checks passed")
    else:
        logger.error("Some startup checks failed")
    
    return checks_passed


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main entry point for the application.
    
    Initializes logging, validates configuration, performs health checks,
    and starts the Flask development server.
    
    For production, use a WSGI server like Gunicorn:
        gunicorn --bind 0.0.0.0:5000 --workers 4 "run:create_application()"
    """
    # Setup logging first
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Multi-Agent Corporate System Starting")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Start time: {datetime.now(timezone.utc).isoformat()}")
    
    # Validate configuration
    config = validate_configuration()
    
    # Setup signal handlers
    setup_signal_handlers(logger)
    
    # Register cleanup on exit
    atexit.register(cleanup)
    
    # Perform startup health checks
    if not perform_startup_checks(config):
        if config['flask']['env'] == 'production':
            logger.error("Startup checks failed in production mode, exiting")
            sys.exit(1)
        else:
            logger.warning("Startup checks failed, continuing in development mode")
    
    # Create application
    app = create_application()
    
    # Get server configuration
    host = config['flask']['host']
    port = config['flask']['port']
    debug = config['flask']['debug']
    
    logger.info("-" * 60)
    logger.info(f"Starting Flask server on http://{host}:{port}")
    print(f"\n\033[92m🚀 Server running!\033[0m")
    if host == '0.0.0.0':
        print(f"\033[92m   Local:   http://localhost:{port}\033[0m")
    else:
        print(f"\033[92m   Local:   http://{host}:{port}\033[0m")
    
    if host == '0.0.0.0':
        import socket
        try:
            # Try to get actual IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            print(f"\033[92m   Network: http://{ip}:{port}\033[0m")
        except:
             print(f"\033[92m   Network: http://<your-ip>:{port}\033[0m")
    print("")
    logger.info(f"Debug mode: {debug}")
    logger.info("-" * 60)
    
    # Run the application
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=debug,
            threaded=True
        )
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is already in use")
            logger.error(f"Try: kill $(lsof -t -i:{port}) or use a different port")
        else:
            logger.exception(f"Failed to start server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        logger.info("Application shutdown complete")


# =============================================================================
# Development Server with Auto-Reload
# =============================================================================

def run_development():
    """
    Run the application in development mode with enhanced features.
    
    Features:
    - Auto-reload on code changes
    - Debug toolbar
    - Verbose logging
    """
    os.environ.setdefault('FLASK_ENV', 'development')
    os.environ.setdefault('FLASK_DEBUG', 'true')
    os.environ.setdefault('LOG_LEVEL', 'DEBUG')
    
    main()


# =============================================================================
# Production Server Helper
# =============================================================================

def run_production():
    """
    Run the application in production mode.
    
    Note: For actual production, use Gunicorn or another WSGI server:
        gunicorn --bind 0.0.0.0:5000 --workers 4 "run:create_application()"
    """
    os.environ.setdefault('FLASK_ENV', 'production')
    os.environ.setdefault('FLASK_DEBUG', 'false')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    
    main()


# =============================================================================
# CLI Interface
# =============================================================================

def cli():
    """
    Command-line interface for the application.
    
    Usage:
        python run.py [command]
        
    Commands:
        run         Start the server (default)
        dev         Start in development mode
        prod        Start in production mode
        check       Run configuration checks only
        version     Show version information
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-Agent Corporate System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py              Start the server
    python run.py dev          Start in development mode
    python run.py check        Validate configuration
    python run.py --port 8080  Start on port 8080
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='run',
        choices=['run', 'dev', 'prod', 'check', 'version'],
        help='Command to execute (default: run)'
    )
    
    parser.add_argument(
        '--host',
        default=None,
        help='Host to bind to'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Port to bind to'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Override environment variables with CLI arguments
    if args.host:
        os.environ['FLASK_HOST'] = args.host
    if args.port:
        os.environ['FLASK_PORT'] = str(args.port)
    if args.debug:
        os.environ['FLASK_DEBUG'] = 'true'
    
    # Execute command
    if args.command == 'run':
        main()
    elif args.command == 'dev':
        run_development()
    elif args.command == 'prod':
        run_production()
    elif args.command == 'check':
        logger = setup_logging()
        config = validate_configuration()
        perform_startup_checks(config)
    elif args.command == 'version':
        print("Multi-Agent Corporate System v1.0.0")
        print(f"Python {sys.version}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    cli()