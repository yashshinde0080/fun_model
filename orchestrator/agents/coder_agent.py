"""
Coder Agent
"""

from typing import Dict, Any
import json
from orchestrator.agents.base_agent import BaseAgent


class CoderAgent(BaseAgent):
    """Coder Agent for code generation."""
    
    AGENT_NAME = "coder"
    
    def _get_template_params(self, task_id: str, phase: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for Coder prompt."""
        params = super()._get_template_params(task_id, phase, inputs)
        
        # Ensure fields are present
        if 'task_description' not in params:
            params['task_description'] = inputs.get('description', 'No description provided')
            
        if 'acceptance_criteria' in inputs:
            ac = inputs['acceptance_criteria']
            params['acceptance_criteria'] = json.dumps(ac, indent=2) if isinstance(ac, list) else str(ac)
        else:
            params['acceptance_criteria'] = "[]"
            
        if 'project_spec' in inputs:
            params['project_spec'] = json.dumps(inputs['project_spec'], indent=2)
        elif 'project_context' in inputs:
             params['project_spec'] = json.dumps(inputs['project_context'], indent=2)
        else:
             params['project_spec'] = "{}"
             
        if 'research_context' in inputs:
            params['research_context'] = json.dumps(inputs['research_context'], indent=2)
        else:
            params['research_context'] = "No specific research provided."
            
        return params
