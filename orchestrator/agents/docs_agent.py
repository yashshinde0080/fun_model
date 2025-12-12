"""
Docs Agent
"""

from typing import Dict, Any
import json
from orchestrator.agents.base_agent import BaseAgent


class DocsAgent(BaseAgent):
    """Documentation Agent."""
    
    AGENT_NAME = "docs"
    
    def _get_template_params(self, task_id: str, phase: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for Docs prompt."""
        params = super()._get_template_params(task_id, phase, inputs)
        
        if 'project_spec' in inputs:
            params['project_spec'] = json.dumps(inputs['project_spec'], indent=2)
        else:
            params['project_spec'] = "{}"
            
        if 'coder_output' in inputs:
            params['coder_output'] = json.dumps(inputs['coder_output'], indent=2)
        else:
            params['coder_output'] = "{}"
            
        if 'qa_report' in inputs:
            params['qa_report'] = json.dumps(inputs['qa_report'], indent=2)
        else:
            params['qa_report'] = "No QA report available."
            
        return params
