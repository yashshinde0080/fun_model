"""
OpenRouter LLM Client
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class RateLimitError(LLMError):
    """Rate limit exceeded."""
    pass


class OpenRouterClient:
    """Client for OpenRouter API."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str = None, timeout: float = 60.0):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.timeout = timeout

        # Global lock for serializing requests
        import threading
        self._lock = threading.Lock()

        if not self.api_key:
            raise ValueError("OpenRouter API key is required")

        try:
            import httpx
            self.client = httpx.Client(
                base_url=self.BASE_URL,
                timeout=timeout,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://multiagent-corp.ai',
                    'X-Title': 'Multi-Agent Corporate System'
                }
            )
            logger.info("OpenRouter client initialized")
        except ImportError:
            logger.error("httpx not installed")
            self.client = None

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str = "meta-llama/llama-3.1-405b-instruct:free",
        temperature: float = 0.7,
        max_tokens: int = 1500,
        api_key: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a completion."""
        if not self.client:
            raise LLMError("HTTP client not initialized")

        # FIX: Intercept known broken/404 models from user config
        if "nemotron" in model.lower() or "nvidia" in model.lower():
            logger.warning(f"Intercepting broken model '{model}'. Redirecting to 'meta-llama/llama-3.1-405b-instruct:free'")
            model = "meta-llama/llama-3.1-405b-instruct:free"

        # STRICT SERIALIZATION: LOCK
        with self._lock:
            start_time = time.time()

            payload = {
                'model': model,
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens,
                **kwargs
            }

            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'

            retries = 5
            backoff = 5.0  # Start with 5 seconds wait

            # Throttling strict: wait 5 seconds BEFORE every request while holding the lock
            logger.info("Acquired lock. Waiting 5s before sending request...")
            time.sleep(5.0)

            for attempt in range(retries):
                response = None
                try:
                    logger.info(f"Sending request to {model} (Attempt {attempt+1}/{retries})")
                    response = self.client.post('/chat/completions', json=payload, headers=headers)

                    if response.status_code == 429:
                        if attempt < retries - 1:
                            sleep_time = backoff * (2 ** attempt)  # 5, 10, 20...
                            logger.warning(f"Rate limited (429). Retrying in {sleep_time}s...")
                            time.sleep(sleep_time)
                            continue
                        else:
                            logger.error("Rate limit exceeded after retries")
                            raise RateLimitError("Rate limit exceeded")

                    if response.is_error:
                        if response.status_code == 401:
                             logger.error("401 Unauthorized: Invalid API Key or Headers")

                        logger.error(f"OpenRouter Error Status: {response.status_code}")
                        logger.error(f"OpenRouter Error Body: {response.text}")
                    response.raise_for_status()
                    data = response.json()

                    # Log success
                    logger.info(f"Request successful. Tokens: {data.get('usage', {}).get('total_tokens', 'unknown')}")

                    return {
                        'content': data['choices'][0]['message']['content'],
                        'model': data.get('model', model),
                        'usage': data.get('usage', {}),
                        'elapsed': time.time() - start_time
                    }
                except Exception as e:
                    # If it's a 401/Auth error, we handled it above or need to handle it here if it raised before
                    if isinstance(e, (RateLimitError, LLMError)):
                         raise

                    # Catch-all for other connection errors
                    is_rate_limit = (response and response.status_code == 429)
                    if attempt == retries - 1 and not is_rate_limit:
                        logger.error(f"LLM completion error: {e}")
                        raise LLMError(f"Completion failed: {e}") from e

    def _mock_response(self, messages, model):
        """Generate a mock response for testing/demo when API fails."""
        import json
        import re

        # Determine identifying context from System Message
        agent_type = "unknown"
        last_msg = messages[-1]['content'].lower()

        # Try to extract the user topic if possible
        topic = "Project"
        try:
            # Look for "User Request: <text>" pattern
            match = re.search(r"user request:\s*(.+?)(?:\n|$)", messages[-1]['content'], re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                if len(topic) > 50: topic = topic[:47] + "..."

            # If extracting directly fails, try to look at logical params
            if topic == "Project" and "description" in last_msg:
                 # Poor man's extraction if structured extraction fails
                 pass
        except:
            pass

        # Refine topic cleaneup - remove quotes
        topic = topic.replace('"', '').replace("'", "")

        # Priority: Check system message for agent identity
        for msg in messages:
            if msg.get('role') == 'system':
                content = msg.get('content', '').lower()
                if 'ceo' in content: agent_type = 'ceo'
                elif 'pm agent' in content or 'project manager' in content: agent_type = 'pm'
                elif 'research' in content: agent_type = 'research'
                elif 'coder' in content: agent_type = 'coder'
                elif 'qa' in content: agent_type = 'qa'
                elif 'docs' in content: agent_type = 'docs'
                break

        # Fallback: Check user message if system message check failed (legacy behavior)
        if agent_type == 'unknown':
            if "ceo" in last_msg: agent_type = 'ceo'
            elif "research" in last_msg: agent_type = 'research'
            elif "plan" in last_msg or "pm" in last_msg: agent_type = 'pm'
            elif "code" in last_msg or "implement" in last_msg: agent_type = 'coder'
            elif "qa" in last_msg or "verify" in last_msg: agent_type = 'qa'
            elif "documentation" in last_msg or "readme" in last_msg: agent_type = 'docs'
            elif "summary" in last_msg: agent_type = 'ceo_final'

        content = "Mock response content"

        # Return appropriate mock content
        if agent_type == 'ceo':
             content = f"""```json
{{
    "project_spec": {{
        "title": "{topic}",
        "description": "## Project Overview\\n\\nWe are initiating the **{topic}** project. This is a critical initiative to demonstrate our engineering capabilities.\\n\\n### Core Vision\\nTo build a robust, scalable solution that addresses the user's needs with precision and elegance.\\n\\n### Technical Strategy\\nWe will leverage a modern tech stack ensuring high performance and maintainability.",
        "scope": "**In Scope**:\\n- Backend API development\\n- Frontend User Interface\\n- Comprehensive Testing\\n\\n**Out of Scope**:\\n- Mobile native apps\\n- Legacy system migration",
        "priority": "High",
        "features": [
            "User Authentication & Authorization",
            "Real-time Data Processing",
            "Responsive Dashboard",
            "Automated Reporting"
        ],
        "tech_stack": ["Python 3.11+", "Flask", "Supabase", "React/Vanilla JS"]
    }},
    "objectives": [
        "Deliver a functional MVP within the timeline",
        "Ensure 90% test coverage",
        "Achieve sub-200ms API response times"
    ],
    "constraints": [
        "Must run in Docker",
        "Strict strict type checking",
        "Follow PEP 8 guidelines"
    ],
    "success_criteria": [
        "All critical user flows verified",
        "Zero critical security vulnerabilities",
        "Stakeholder sign-off"
    ]
}}
```"""

        elif agent_type == 'research':
             content = f"""```json
{{
    "summary": "### Research Summary for {topic}\\n\\nBased on the analysis of current market trends and technical requirements, the following architectural decisions have been made.\\n\\n#### Architecture Pattern\\nWe recommend a **Microservices Architecture** (or Modular Monolith) to ensure scalability and independent deployment cycles.",
    "findings": [
        {{
            "topic": "Frontend Framework",
            "content": "React 18 with Vite offers the best balance of performance and developer experience. State management via Zustand is recommended over Redux for simplicity.",
            "relevance": "High"
        }},
        {{
            "topic": "Database Strategy",
            "content": "PostgreSQL (via Supabase) provides robust relational data integrity with built-in realtime capabilities.",
            "relevance": "Critical"
        }},
        {{
            "topic": "Security Compliance",
            "content": "OWASP Top 10 mitigation strategies must be implemented at the gateway level. JWT rotation is mandatory.",
            "relevance": "High"
        }}
    ],
    "citations": [
        {{
            "source": "React Documentation",
            "url": "https://react.dev",
            "reference": "v18.2.0"
        }},
        {{
            "source": "OWASP Security Guide",
            "url": "https://owasp.org/www-project-top-ten/",
            "reference": "2024 Edition"
        }}
    ],
    "recommendations": [
        "Adopt TypeScript for type safety across the stack",
        "Implement CI/CD pipelines using GitHub Actions",
        "Use Docker Compose for local development consistency"
    ]
}}
```"""

        elif agent_type == 'pm':
             content = f"""```json
{{
  "tasks": [
    {{
      "id": "task-001",
      "name": "Technical Research",
      "assigned_to": "research",
      "description": "Conduct a deep dive into the architectural requirements for {topic}. Evaluate libraries and define the data schema.",
      "acceptance_criteria": [
          "Architecture diagram created",
          "Library selection finalized",
          "Database schema defined"
      ],
      "estimated_complexity": "medium"
    }},
    {{
      "id": "task-002",
      "name": "Core Implementation",
      "assigned_to": "coder",
      "description": "Develop the foundational codebase for {topic}. Implement the main logic and API endpoints.",
      "acceptance_criteria": [
          "API endpoints reachable",
          "Core logic passed unit tests",
          "Error handling implemented"
      ],
      "estimated_complexity": "high"
    }},
    {{
      "id": "task-003",
      "name": "Quality Assurance",
      "assigned_to": "qa",
      "description": "Run comprehensive tests against the implemented solution. Verify edge cases and security compliance.",
      "acceptance_criteria": [
          "All tests passed",
          "Security scan clear",
          "Performance metrics met"
      ],
      "estimated_complexity": "medium"
    }},
    {{
      "id": "task-004",
      "name": "Documentation",
      "assigned_to": "docs",
      "description": "Generate technical documentation, API references, and user guides.",
      "acceptance_criteria": [
          "README.md complete",
          "API docs generated",
          "Setup guide verified"
      ],
      "estimated_complexity": "low"
    }}
  ],
  "dependencies": {{
    "task-002": ["task-001"],
    "task-003": ["task-002"],
    "task-004": ["task-003"]
  }},
  "execution_order": ["task-001", "task-002", "task-003", "task-004"],
  "timeline_estimate": "3-5 days",
  "risk_assessment": [
      "Scope creep potential in core features",
      "Third-party API rate limits"
  ]
}}
```"""

        elif agent_type == 'coder':
             content = f"""```json
{{
    "thought_process": "To implement **{topic}** with high quality, I have designed a solution that prioritizes maintainability and scalability.\\n\\n### Architecture Decisions\\n1. **Framework**: Using Flask for its lightweight and flexible nature.\\n2. **Containerization**: Implementing a multi-stage Dockerfile to optimize image size.\\n3. **Testing**: Including comprehensive unit tests using `pytest`.\\n\\nI will now generate the requisite files, ensuring all code follows PEP 8 standards and includes type hints.",
    "implementation_steps": [
        "Create app.py with Flask server",
        "Define requirements.txt",
        "Create Dockerfile",
        "Write basic tests"
    ],
    "files": [
        {{
            "path": "app.py",
            "content": "from flask import Flask, jsonify, request\\nfrom flask_cors import CORS\\nimport os\\n\\napp = Flask(__name__)\\nCORS(app)\\n\\n@app.route('/health')\\ndef health():\\n    return jsonify({{'status': 'healthy', 'version': '1.0.0'}})\\n\\n@app.route('/api/v1/resource', methods=['GET'])\\ndef get_resource():\\n    # Implementation of core logic\\n    return jsonify({{'data': 'Resource content', 'timestamp': '2024-01-01'}})\\n\\nif __name__ == '__main__':\\n    app.run(port=int(os.getenv('PORT', 5000)))"
        }},
        {{
            "path": "requirements.txt",
            "content": "flask>=3.0.0\\nflask-cors>=4.0.0\\ngunicorn>=21.2.0"
        }},
        {{
            "path": "Dockerfile",
            "content": "FROM python:3.11-slim\\nWORKDIR /app\\nCOPY requirements.txt .\\nRUN pip install -r requirements.txt\\nCOPY . .\\nCMD [\\"gunicorn\\", \\"--bind\\", \\"0.0.0.0:5000\\", \\"app:app\\"]"
        }},
        {{
            "path": "README.md",
            "content": "# {topic}\\n\\n## Setup\\n1. `pip install -r requirements.txt`\\n2. `python app.py`"
        }}
    ],
    "run_instructions": {{"run": "python app.py", "test": "pytest"}},
    "dependencies": ["flask", "flask-cors", "gunicorn"]
}}
```"""

        elif agent_type == 'qa':
             content = """```json
{
    "thought_process": "I have conducted a thorough code review of the submission. \\n\\n**Key Observations:**\\n- Code structure is modular and clean.\\n- Dockerfile follows best practices (using slim images).\\n- Tests cover the critical paths.\\n- No security vulnerabilities detected in the static analysis.",
    "review_summary": "### QA Validation Passed\\n\\nThe solution meets all acceptance criteria. The code is well-documented and robust. I recommend proceeding to the deployment phase.",
    "validation_results": [
        {"test": "Syntax Check", "status": "pass", "message": "Code is valid Python syntax."},
        {"test": "Dependency Check", "status": "pass", "message": "All dependencies are pinned in requirements.txt."},
        {"test": "Security Scan", "status": "pass", "message": "No hardcoded secrets found."},
        {"test": "Dockerfile Lint", "status": "pass", "message": "Dockerfile follows best practices."}
    ],
    "approval_status": "approved",
    "test_coverage_estimate": "85%"
}
```"""

        elif agent_type == 'docs':
             content = f"""```json
{{
    "readme": "# {topic}\\n\\n## Overview\\nThis project implements {topic} using a scalable architecture.\\n\\n## Features\\n- **REST API**: Fully documented endpoints.\\n- **Dockerized**: Ready for container orchestration.\\n- **Secure**: Implements standard security practices.\\n\\n## Getting Started\\nClone the repo and run `docker-compose up`.",
    "summary": "Complete documentation suite generated including API reference and deployment guide.",
    "release_notes": "v1.0.0-alpha: Initial release with core features."
}}
```"""

        elif agent_type == 'ceo_final':
             content = f"The project '{topic}' has been successfully completed. The team has delivered a robust solution including source code, comprehensive tests, and documentation. All objectives outlined in the initial specification have been met."

        return {
            'content': content,
            'model': 'mock-model',
            'usage': {'total_tokens': 0},
            'elapsed': 0.1
        }

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        model: str = "anthropic/claude-3-sonnet",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a JSON completion."""
        response = self.complete(messages=messages, model=model, **kwargs)
        content = response['content']

        import json
        import re

        # Robust JSON extraction
        try:
            # 0. Try direct parsing first (pure JSON response)
            try:
                response['parsed'] = json.loads(content)
                logger.info(f"Successfully parsed pure JSON from {model}.")
                return response
            except json.JSONDecodeError:
                pass

            # 1. Try finding a markdown JSON block (tolerant of casing and missing language identifier)
            # Match ``` followed optionally by json/JSON, then capture the brace content
            json_block = re.search(r'```(?:json|JSON)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_block:
                content_to_parse = json_block.group(1)
                try:
                    response['parsed'] = json.loads(content_to_parse)
                    logger.info(f"Successfully parsed JSON from markdown block.")
                    return response
                except json.JSONDecodeError:
                    logger.warning("Found markdown block but failed to parse JSON content.")

            # 2. Try looking for the outer curly braces with validation
            parsed = None
            
            # Find all occurrences of '{'
            starts = [i for i, char in enumerate(content) if char == '{']
            
            for start in starts:
                balance = 0
                for i in range(start, len(content)):
                    if content[i] == '{':
                        balance += 1
                    elif content[i] == '}':
                        balance -= 1
                    
                    if balance == 0:
                        candidate = content[start:i+1]
                        try:
                            parsed_candidate = json.loads(candidate)
                            # Heuristic: if it has "status" or "payload", it's likely ours.
                            if isinstance(parsed_candidate, dict) and ('status' in parsed_candidate or 'payload' in parsed_candidate or 'task_id' in parsed_candidate):
                                parsed = parsed_candidate
                                break
                        except json.JSONDecodeError:
                            continue
                
                if parsed:
                    break

            if parsed:
                response['parsed'] = parsed
            else:
                # 3. Attempt to clean and parse
                try:
                    # Remove markdown code blocks if present but regex failed earlier
                    clean_content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
                    clean_content = re.sub(r'^```\s*', '', clean_content, flags=re.MULTILINE)
                    clean_content = re.sub(r'```$', '', clean_content, flags=re.MULTILINE)
                    
                    # Try to find the outer braces again on cleaned content
                    start_idx = clean_content.find('{')
                    end_idx = clean_content.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1:
                        json_str = clean_content[start_idx:end_idx+1]
                        
                        # aggressive cleanup: escape unescaped newlines in values
                        # This is risky but helps with "code" blocks in JSON
                        # We don't have a perfect parser, but we can try basic repairs
                        
                        response['parsed'] = json.loads(json_str)
                        logger.info("Successfully parsed JSON after manual cleanup.")
                    else:
                        raise ValueError("No JSON object found")
                except Exception:
                     # Final fallback: Wide search
                    start_idx = content.find('{')
                    end_idx = content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                         response['parsed'] = json.loads(content[start_idx:end_idx+1])
                    else:
                        raise

            logger.info(f"Successfully parsed JSON from {model}. Keys: {list(response['parsed'].keys())}")
            return response

        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw content: {content}")

            # Fallback for creating an error response structure
            response['parsed'] = {
                "status": "failed",
                "error": f"JSON parsing failed: {str(e)}",
                "task_id": "unknown", 
                "agent": "unknown",
                "payload": {
                    "error": "JSON parsing failed",
                    "raw_content": content
                },
                "raw_content": content
            }
            return response

    def close(self):
        if self.client:
            self.client.close()
