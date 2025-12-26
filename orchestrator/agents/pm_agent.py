"""
PM Agent - Task Planning
"""

from typing import Dict, Any
import json
from orchestrator.agents.base_agent import BaseAgent


class PMAgent(BaseAgent):
    """Project Manager Agent for task planning."""

    AGENT_NAME = "pm"

    def _get_template_params(self, task_id: str, phase: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for PM prompt."""
        params = super()._get_template_params(task_id, phase, inputs)

        # Ensure project_spec is formatted
        if 'project_spec' in inputs:
            params['project_spec'] = json.dumps(inputs['project_spec'], indent=2)

        # Ensure objectives are formatted
        if 'objectives' in inputs:
            obj = inputs['objectives']
            params['objectives'] = json.dumps(obj, indent=2) if isinstance(obj, list) else str(obj)

        # Ensure constraints are formatted
        if 'constraints' in inputs:
            const = inputs['constraints']
            params['constraints'] = json.dumps(const, indent=2) if isinstance(const, list) else str(const)

        return params
