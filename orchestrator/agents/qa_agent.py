"""
QA Agent
"""

from typing import Dict, Any
import json
from orchestrator.agents.base_agent import BaseAgent


class QAAgent(BaseAgent):
    """QA Agent for validation."""
    
    AGENT_NAME = "qa"
    
    def _get_template_params(self, task_id: str, phase: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for QA prompt."""
        params = super()._get_template_params(task_id, phase, inputs)
        
        if 'task_description' not in params:
            params['task_description'] = inputs.get('description', 'Validate codebase')
            
        if 'coder_output' in inputs:
            params['coder_output'] = json.dumps(inputs['coder_output'], indent=2)
        else:
            params['coder_output'] = "{}"
            
        if 'project_requirements' in inputs:
             params['project_requirements'] = json.dumps(inputs['project_requirements'], indent=2)
        elif 'project_spec' in inputs:
             params['project_requirements'] = json.dumps(inputs['project_spec'], indent=2)
        else:
             params['project_requirements'] = "{}"
             
        if 'success_criteria' in inputs:
            sc = inputs['success_criteria']
            params['success_criteria'] = json.dumps(sc, indent=2) if isinstance(sc, list) else str(sc)
        else:
            params['success_criteria'] = "[]"
            
        return params
