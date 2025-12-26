"""
Base Agent Class
"""

import time
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any

from orchestrator.llm_client import OpenRouterClient, LLMError
from orchestrator.config import get_agent_config

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    AGENT_NAME: str = "base"

    def __init__(self, llm_client: OpenRouterClient):
        self.llm_client = llm_client
        self.config = get_agent_config(self.AGENT_NAME)
        self.prompts_dir = Path(__file__).parent.parent / 'prompts'
        logger.debug(f"Initialized {self.AGENT_NAME} agent")

    def _load_prompt_template(self, filename: str) -> str:
        """Load prompt template from file."""
        try:
            path = self.prompts_dir / filename
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load prompt template {filename}: {e}")
            raise

    def _get_template_params(self, task_id: str, phase: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare parameters for prompt template injection.
        Override this in subclasses to add agent-specific parameters.
        """
        params = {
            'task_id': task_id,
            'phase': phase,
            'user_request': inputs.get('user_request', ''),
            'additional_context': '',
            'phase_instructions': ''
        }

        # Flatten inputs into params for easy access
        for k, v in inputs.items():
            if isinstance(v, (dict, list)):
                params[k] = json.dumps(v, indent=2)
            else:
                params[k] = str(v)

        return params

    def execute(
        self,
        task_id: str,
        phase: str,
        inputs: Dict[str, Any],
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """Execute the agent's task."""
        start_time = time.time()

        try:
            # Load template
            template_name = f"{self.AGENT_NAME}_prompt.txt"
            template = self._load_prompt_template(template_name)

            # Prepare parameters
            params = self._get_template_params(task_id, phase, inputs)

            # Format prompt with safe substitution (optional, but using standard format here)
            # Using format() allows {key} substitution.
            # We need to be careful about JSON braces in text, but the prompts seem to use {key}.
            # Manual substitution to avoid conflicts with JSON braces in prompt
            prompt = template
            for key, value in params.items():
                # Replace {key} with value
                prompt = prompt.replace(f"{{{key}}}", str(value))

            # No longer using format(), so we don't need to catch KeyError for braces
            # But we should verify if all expected parameters were used if strictness is required.
            # For now, this is safer for JSON-heavy prompts.

            model = self.config.get('model', 'anthropic/claude-3-sonnet')
            temperature = self.config.get('temperature', 0.5)
            max_tokens = self.config.get('max_tokens', 4096)

            # We send the entire formatted prompt as the User message
            # and a generic System message to set the behavior.
            messages = [
                {"role": "system", "content": f"You are the {self.AGENT_NAME.upper()} Agent. Return JSON only."},
                {"role": "user", "content": prompt}
            ]

            # Get API key from config if present
            api_key = self.config.get('api_key')

            response = self.llm_client.complete_json(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key
            )

            result = response.get('parsed', {})
            result['task_id'] = task_id
            result['agent'] = self.AGENT_NAME

            if 'meta' not in result:
                result['meta'] = {}
            result['meta']['elapsed'] = time.time() - start_time
            result['meta']['model_used'] = response.get('model', model)

            if 'status' not in result:
                result['status'] = 'success'
            if 'confidence' not in result:
                result['confidence'] = 'medium'

            # Ensure payload exists and contains the actual data
            if 'payload' not in result:
                # If LLM didn't nest it in 'payload', treat the remaining keys as payload
                exclude_keys = {'task_id', 'agent', 'meta', 'status', 'confidence'}
                result['payload'] = {k: v for k, v in result.items() if k not in exclude_keys}

            return result

        except Exception as e:
            logger.error(f"Agent {self.AGENT_NAME} error: {e}")
            return {
                'status': 'failed',
                'task_id': task_id,
                'agent': self.AGENT_NAME,
                'payload': {'error': str(e)},
                'confidence': 'low',
                'meta': {'elapsed': time.time() - start_time}
            }
