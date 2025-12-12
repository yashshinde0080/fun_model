import os

# All folders to create
FOLDERS = [
    "multiagent-corp/app/templates",
    "multiagent-corp/app/static",
    "multiagent-corp/config",
    "multiagent-corp/orchestrator/agents",
    "multiagent-corp/orchestrator/prompts",
    "multiagent-corp/orchestrator/tools",
    "multiagent-corp/storage/logs",
    "multiagent-corp/infra",
    "multiagent-corp/tests",
]

# Files with default content
FILES = {
    "multiagent-corp/app/__init__.py": "",
    "multiagent-corp/app/routes.py": "from flask import Blueprint\nroutes = Blueprint('routes', __name__)\n",
    "multiagent-corp/app/auth.py": "# Supabase auth middleware & helpers\n",
    "multiagent-corp/app/templates/index.html": "<html><body><h1>Multiagent Corp</h1></body></html>",
    "multiagent-corp/app/static/style.css": "body { font-family: sans-serif; }",
    
    "multiagent-corp/config/agents.yaml": "# agent config\n",
    "multiagent-corp/config/openrouter.yaml": "# openrouter config\n",
    "multiagent-corp/config/supabase.yaml": "# supabase config\n",
    "multiagent-corp/config/smtp.yaml": "# smtp config\n",
    
    "multiagent-corp/orchestrator/__init__.py": "",
    "multiagent-corp/orchestrator/orchestrator.py": "class Orchestrator:\n    pass\n",
    "multiagent-corp/orchestrator/llm_client.py": "class LLMClient:\n    pass\n",

    # Agents
    "multiagent-corp/orchestrator/agents/__init__.py": "",
    "multiagent-corp/orchestrator/agents/base_agent.py": "class BaseAgent:\n    pass\n",
    "multiagent-corp/orchestrator/agents/ceo_agent.py": "class CEOAgent:\n    pass\n",
    "multiagent-corp/orchestrator/agents/pm_agent.py": "class PMAgent:\n    pass\n",
    "multiagent-corp/orchestrator/agents/research_agent.py": "class ResearchAgent:\n    pass\n",
    "multiagent-corp/orchestrator/agents/coder_agent.py": "class CoderAgent:\n    pass\n",
    "multiagent-corp/orchestrator/agents/qa_agent.py": "class QAAgent:\n    pass\n",
    "multiagent-corp/orchestrator/agents/docs_agent.py": "class DocsAgent:\n    pass\n",

    # Prompts
    "multiagent-corp/orchestrator/prompts/ceo_prompt.txt": "",
    "multiagent-corp/orchestrator/prompts/pm_prompt.txt": "",
    "multiagent-corp/orchestrator/prompts/research_prompt.txt": "",
    "multiagent-corp/orchestrator/prompts/coder_prompt.txt": "",
    "multiagent-corp/orchestrator/prompts/qa_prompt.txt": "",
    "multiagent-corp/orchestrator/prompts/docs_prompt.txt": "",

    # Tools
    "multiagent-corp/orchestrator/tools/sandbox_runner.py": "class SandboxRunner:\n    pass\n",
    "multiagent-corp/orchestrator/tools/memory.py": "class Memory:\n    pass\n",
    "multiagent-corp/orchestrator/tools/notifier.py": "class Notifier:\n    pass\n",

    # Storage
    "multiagent-corp/storage/memory.db": "",
    
    # Infra
    "multiagent-corp/infra/Dockerfile": "FROM python:3.11\n",
    "multiagent-corp/infra/docker-compose.yml": "version: '3'\nservices:\n  app:\n    build: .\n",

    # Tests
    "multiagent-corp/tests/integration_test.py": "def test_integration():\n    assert True\n",

    # Root files
    "multiagent-corp/run.py": "print('Running multiagent-corp')",
    "multiagent-corp/requirements.txt": "",
    "multiagent-corp/README.md": "# Multiagent Corp\n",
}

def ensure_dirs():
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)

def ensure_files():
    for filepath, content in FILES.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

def main():
    ensure_dirs()
    ensure_files()
    print("Project structure created.")

if __name__ == "__main__":
    main()
