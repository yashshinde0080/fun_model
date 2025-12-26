"""
CEO Agent - Project Specification and Finalization
"""

from typing import Dict, Any
import json
from orchestrator.agents.base_agent import BaseAgent


class CEOAgent(BaseAgent):
    """CEO Agent for project specification and approval."""

    AGENT_NAME = "ceo"

    def _get_template_params(self, task_id: str, phase: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for CEO prompt."""
        params = super()._get_template_params(task_id, phase, inputs)

        # Add phase instructions
        if phase == 'specification':
            params['phase_instructions'] = (
                "Analyze the user request and create a detailed project specification. "
                "Focus on clarity, feasibility, and alignment with the user's goals."
            )
        elif phase == 'finalization':
            params['phase_instructions'] = (
                "Review the completed work (all_outputs) against the original specification. "
                "Determine if the project is complete and successful. "
                "Generate a final summary and approval status."
            )
        else:
            params['phase_instructions'] = "Perform the task as requested."

        # Add additional context if available
        if 'all_outputs' in inputs:
            params['additional_context'] = f"All Outputs:\n{json.dumps(inputs['all_outputs'], indent=2)}"
        elif 'qa_report' in inputs:
            params['additional_context'] = f"QA Report:\n{json.dumps(inputs['qa_report'], indent=2)}"

        return params
