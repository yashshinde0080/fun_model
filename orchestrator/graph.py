
import os
import logging
from typing import Dict, Any, List, TypedDict, Optional, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser

from orchestrator.config import get_agent_config

logger = logging.getLogger(__name__)

# --- State Definition (Schema) ---
class AgentState(TypedDict):
    """The global state of the LangGraph workflow."""
    
    # Core User Input
    user_request: str
    user_id: str
    workspace_id: str

    # Workflow Artifacts
    clarifications: List[str]
    intent: str
    specification: Dict[str, Any]
    project_plan: Dict[str, Any]
    backlog: List[Dict[str, Any]]
    
    # Execution State
    codebase: Dict[str, str]  # file_path -> content
    qa_feedback: List[str]
    qa_status: str  # "PASS" | "FAIL" | "PENDING"
    iteration_count: int
    max_iterations: int
    
    # Final Artifacts
    documentation: Dict[str, str]
    final_decision: str  # "APPROVED" | "REJECTED"
    
    # Research
    research_info: Dict[str, Any]

    # Meta / History
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Errors
    error: Optional[str]


# --- Nodes ---

class WorkflowNodes:
    def __init__(self, llm_client=None, event_callback=None):
        self.llm_client = llm_client 
        self.event_callback = event_callback
        # Note: We will use ChatOpenAI for LangChan compat, 
        # but configured to point to OpenRouter as per existing setup.

    def _log(self, agent: str, event_type: str, payload: Dict[str, Any]):
        """Helper to log events if callback is provided."""
        if self.event_callback:
            try:
                self.event_callback(agent, event_type, payload)
            except Exception as e:
                logger.error(f"Failed to log event: {e}")

    def _get_llm(self, agent_name: str, temperature: float = 0.7):
        """Get a LangChain ChatOpenAI instance configured for OpenRouter."""
        
        # 1. Get Config
        config = get_agent_config(agent_name)
        model_name = config.get('model', 'meta-llama/llama-3.1-405b-instruct:free')
        
        # 2. Interceptor Logic (Mirroring llm_client.py)
        if "nemotron" in model_name.lower() or "nvidia" in model_name.lower():
            logger.warning(f"Intercepting broken model '{model_name}'. Redirecting to 'meta-llama/llama-3.1-405b-instruct:free'")
            model_name = 'meta-llama/llama-3.1-405b-instruct:free'

        # 3. API Key
        # We assume OPENROUTER_API_KEY is in env or config
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
             # Fallback to loading from yaml if needed, but usually env is best
             pass

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature
        )

    def clarification_gate(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Analyze Input & Clarify"""
        logger.info("--- Phase 1: Clarification Gate ---")
        
        llm = self._get_llm("ceo", temperature=0.7)
        request = state['user_request']
        
        prompt = f"""
        You are the Agile Input Analyzer.
        Analyze this request: "{request}"
        
        Is this request enough to start a project?
        Unless it is complete gibberish, say YES.
        If it is vague, Infer a reasonable intent.
        
        Return JSON ONLY: {{ "status": "clear", "intent": "summary of intent (inferred if needed)", "questions": [] }}
        """
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            parser = JsonOutputParser()
            result = parser.parse(response.content)
            
            if result.get('status') == 'ambiguous':
                self._log("ceo", "task_completed", {"output": {"questions": result.get('questions', [])}, "intent": "AMBIGUOUS"})
                return {
                    "clarifications": result.get('questions', []),
                    "intent": "AMBIGUOUS"
                }
            else:
                 self._log("ceo", "task_completed", {"output": {"intent": result.get('intent')}, "status": "clear"})
                 return {
                    "intent": result.get('intent', 'Clear intent verified'),
                    "clarifications": []
                }
        except Exception as e:
            logger.error(f"Clarification failed: {e}")
            self._log("ceo", "error", {"error": str(e)})
            # Fallback
            return {"intent": "Assumed clear (error recovery)", "clarifications": []}

    def specification(self, state: AgentState) -> Dict[str, Any]:
        """Node 2: CEO Specification"""
        logger.info("--- Phase 2: CEO Specification ---")
        
        llm = self._get_llm("ceo", temperature=0.7)
        
        prompt = f"""
        You are the CEO Agent.
        User Request: {state['user_request']}
        Intent: {state['intent']}
        
        Create a detailed software specification.
        Include: Functional Requirements, Non-Functional Requirements, Acceptance Criteria, Assumptions.
        Do NOT write code.
        
        Return JSON ONLY: {{ "requirements": [], "acceptance_criteria": [], "assumptions": [] }}
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        parser = JsonOutputParser()
        spec = parser.parse(response.content)
        
        self._log("ceo", "task_completed", {"output": {"project_spec": spec}})
        return {"specification": spec}

    def research(self, state: AgentState) -> Dict[str, Any]:
        """Node 2.5: Research"""
        logger.info("--- Phase 2.5: Research ---")
        
        llm = self._get_llm("research", temperature=0.5)
        spec = state['specification']
        
        prompt = f"""
        You are the Research Agent.
        Specification: {spec}
        
        Conduct research on the best libraries, tools, and patterns to implement this.
        
        Return JSON ONLY: {{ "summary": "markdown summary...", "findings": [{{ "topic": "topic", "content": "details" }}], "recommendations": ["rec1", "rec2"] }}
        """
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            parser = JsonOutputParser()
            findings = parser.parse(response.content)
            
            self._log("research", "task_completed", {"output": findings})
            return {"research_info": findings}
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {"research_info": {"error": str(e)}}

    def sprint_planning(self, state: AgentState) -> Dict[str, Any]:
        """Node 3: PM Planning"""
        logger.info("--- Phase 3: Sprint Planning ---")
        
        llm = self._get_llm("pm", temperature=0.5)
        spec = state['specification']
        
        prompt = f"""
        You are the Project Manager.
        Specification: {spec}
        
        Break this down into a Sprint Backlog.
        Define the Definition of Done.
        
        Return JSON ONLY: {{ "backlog": [{{ "id": "1", "task": "..." }}], "definition_of_done": "..." }}
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        parser = JsonOutputParser()
        plan = parser.parse(response.content)
        
        self._log("pm", "task_completed", {"output": {"tasks": plan.get("backlog", []), "plan": plan}})
        return {
            "project_plan": plan, 
            "backlog": plan.get("backlog", []),
            "iteration_count": 0,
            "max_iterations": 5
        }

    def implementation(self, state: AgentState) -> Dict[str, Any]:
        """Node 4A: Coder Implementation"""
        logger.info(f"--- Phase 4A: Implementation (Iteration {state['iteration_count'] + 1}) ---")
        
        llm = self._get_llm("coder", temperature=0.3)
        backlog = state['backlog']
        codebase = state.get('codebase', {})
        feedback = state.get('qa_feedback', [])
        
        prompt = f"""
        You are the Senior Coder.
        Backlog: {backlog}
        Existing Code: {list(codebase.keys())}
        QA Feedback (if any): {feedback}
        
        Implement the solution. Write PRODUCTION-READY code.
        Return a JSON mapping of filenames to file content.
        
        Return JSON ONLY: {{ "files": {{ "src/main.py": "code...", "README.md": "..." }} }}
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        parser = JsonOutputParser()
        result = parser.parse(response.content)
        
        new_files = result.get('files', {})
        # Merge with existing
        codebase.update(new_files)
        
        # Convert to list for UI compatibility
        files_list = [{'path': k, 'content': v} for k, v in new_files.items()]
        
        self._log("coder", "task_completed", {"output": {"files": files_list}, "iteration": state['iteration_count'] + 1})
        return {"codebase": codebase}

    def validation(self, state: AgentState) -> Dict[str, Any]:
        """Node 4B: QA Validation"""
        logger.info("--- Phase 4B: QA Validation ---")
        
        llm = self._get_llm("qa", temperature=0.3)
        codebase = state['codebase']
        spec = state['specification']
        
        prompt = f"""
        You are the QA Lead.
        Specification: {spec}
        Codebase Files: {list(codebase.keys())}
        
        Review the code content (simulated).
        Does it meet the criteria?
        
        Return JSON ONLY: {{ "status": "PASS" | "FAIL", "feedback": ["issue 1", "issue 2"] }}
        """
        
        # In a real tool-backed step, we would run the code here.
        # For this simulation/text-based check:
        response = llm.invoke([HumanMessage(content=prompt)])
        parser = JsonOutputParser()
        result = parser.parse(response.content)
        
        self._log("qa", "task_completed", {"output": {"validation_results": result}, "status": result.get('status', 'FAIL')})
        return {
            "qa_status": result.get('status', 'FAIL'),
            "qa_feedback": result.get('feedback', []),
            "iteration_count": state['iteration_count'] + 1
        }

    def documentation(self, state: AgentState) -> Dict[str, Any]:
        """Node 5: Documentation"""
        logger.info("--- Phase 5: Documentation ---")
        
        llm = self._get_llm("docs", temperature=0.4)
        
        prompt = "Generate comprehensive README.md and documentation based on the final codebase."
        response = llm.invoke([HumanMessage(content=prompt)])
        
        docs = {"README.md": response.content}
        self._log("docs", "task_completed", {"output": {"readme": response.content}})
        return {"documentation": docs}

    def final_review(self, state: AgentState) -> Dict[str, Any]:
        """Node 6: Final Review"""
        logger.info("--- Phase 6: Final Review ---")
        
        # CEO final signoff
        return {"final_decision": "APPROVED"} # Simplified for success path


# --- Graph Construction ---

def build_workflow_graph(event_callback=None):
    nodes = WorkflowNodes(event_callback=event_callback)
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("clarification_gate", nodes.clarification_gate)
    workflow.add_node("specification", nodes.specification)
    workflow.add_node("research", nodes.research)
    workflow.add_node("sprint_planning", nodes.sprint_planning)
    workflow.add_node("implementation", nodes.implementation)
    workflow.add_node("validation", nodes.validation)
    workflow.add_node("documentation", nodes.documentation)
    workflow.add_node("final_review", nodes.final_review)

    # Define Edges / Conditional Logic
    
    def check_ambiguity(state):
        if state.get('clarifications') and len(state['clarifications']) > 0:
            return "stop" # Or loop back to user input in a real interactive app
        return "specification"

    def check_qa(state):
        if state['qa_status'] == 'PASS':
            return "documentation"
        if state['iteration_count'] >= state['max_iterations']:
             return "fail" # Stop iteration
        return "implementation" # Loop back
        
    # Build Graph
    workflow.set_entry_point("clarification_gate")
    
    workflow.add_conditional_edges(
        "clarification_gate",
        check_ambiguity,
        {
            "stop": END, # In this sync runner, we stop. Interaction handled by caller.
            "specification": "specification"
        }
    )
    
    workflow.add_edge("specification", "research")
    workflow.add_edge("research", "sprint_planning")
    workflow.add_edge("sprint_planning", "implementation")
    workflow.add_edge("implementation", "validation")
    
    workflow.add_conditional_edges(
        "validation",
        check_qa,
        {
            "implementation": "implementation", # The Loop
            "documentation": "documentation",
            "fail": END
        }
    )
    
    workflow.add_edge("documentation", "final_review")
    workflow.add_edge("final_review", END)
    
    return workflow.compile()