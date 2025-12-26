# Hierarchical AI Organization Architecture

This document describes the high-level architecture, folder structure, and technical design of the Multi-Agent Corporate System.

## 1. High-Level Architecture Summary

The system is a **Hierarchical Multi-Agent AI Organization** designed to autonomously build software projects. It mimics a corporate structure where a CEO sets the vision, a PM plans the work, Specialists (Coder, Research) execute, and QA/Docs ensure quality.

### core components:
1.  **Orchestrator (The Controller)**:
    *   Manages the state machine of the workflow.
    *   Enforces rules: decomposed tasks, retries (max 2), timeouts (60s), iteration limits (6).
    *   Handles routing between agents.
2.  **Agent Logic (The Employees)**:
    *   **CEO**: Project spec & Final sign-off.
    *   **PM**: Task decomposition & Dependency graph.
    *   **Specialists**: Research & Coder.
    *   **Support**: QA (Validation) & Docs (Documentation).
    *   All agents operate on strict **JSON Contracts**.
3.  **Backend Infrastructure**:
    *   **Flask**: API Gateway & UI Server.
    *   **Supabase**: Authentication, Workflows Log, Events Log, Artifacts.
    *   **OpenRouter**: LLM Provider (Model agnostic).
    *   **SMTP**: Notification system.

## 2. Technical Design

### Data Flow
`User Request` → `CEO (Spec)` → `PM (Plan)` → `Loop [Coder/Research]` → `QA (Verify)` → `Docs (Write)` → `CEO (Finalize)` → `User Deliverable`

### Schema Contracts
All agents return:
```json
{
  "status": "success|failed",
  "task_id": "string",
  "agent": "agent_name",
  "payload": { ...agent_specific_data... },
  "confidence": "low|medium|high",
  "meta": { "elapsed": 1.23 }
}
```

### Prompt Engineering
- **JSON Enforcement**: All prompts explicitly forbid prose and require strictly formatted JSON.
- **Param Injection**: `{task_description}`, `{project_spec}`, etc. are injected dynamically.

## 3. Folder Structure

```
D:\fun_model\
├── app\
│   ├── static\         # CSS/JS assets
│   ├── templates\      # HTML templates (index.html)
│   ├── auth.py         # Supabase Middleware
│   └── routes.py       # Flask Endpoints
├── config\             # YAML Configuration
│   ├── agents.yaml     # Agent models & rules (retries, timeouts)
│   └── ...
├── orchestrator\
│   ├── agents\         # Agent implementations
│   │   ├── base_agent.py
│   │   ├── ceo_agent.py
│   │   └── ...
│   ├── prompts\        # Prompt templates (.txt)
│   ├── tools\          # Utilities (notifier, sandbox)
│   ├── llm_client.py   # OpenRouter Client
│   └── orchestrator.py # Workflow Engine
├── storage\            # Logs & Temp files
├── run.py              # Entry point
└── ...
```

## 4. Workflow Configuration

Controlled via `config/agents.yaml`:
```yaml
orchestration:
  max_retries_per_task: 2
  max_iterations_per_workflow: 6
  task_timeout_seconds: 60
  require_qa_approval: true
  require_ceo_finalization: true
```

## 5. Deployment
- **Backend**: Python 3.13 + Flask
- **Env Vars**: `.env` handling keys for Supabase/OpenRouter/SMTP.
- **Run**: `python run.py` (Dev) / Gunicorn (Prod).

## 6. Example Workflow Output (Abstract)

**CEO**:
```json
{
    "project_spec": {
        "title": "Flask Microservice",
        "description": "Create a health check service",
        "scope": "Minimal flask app with tests"
    }
}
```

**PM**:
```json
{
    "tasks": [
        {"id": "t1", "name": "Setup", "assigned_to": "coder"},
        {"id": "t2", "name": "Tests", "assigned_to": "coder"}
    ],
    "execution_order": ["t1", "t2"]
}
```

**Coder (t1)**:
```json
{
    "files": [
        {"path": "app.py", "content": "..."}
    ]
}
```

**QA**:
```json
{
    "validation_results": [{"test": "syntax", "status": "pass"}]
}
```
