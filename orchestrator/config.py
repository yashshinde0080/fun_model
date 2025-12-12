"""
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
