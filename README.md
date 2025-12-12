# Multiagent Corp
# Multi-Agent Corporate System

A production-grade multi-agent AI system modeled as a corporate organization for automated software development workflows.

## 🏢 Architecture

The system operates as a virtual software development organization with specialized AI agents:

| Agent | Role | Responsibilities |
|-------|------|-----------------|
| **CEO** | Executive | Project specification, final approval |
| **PM** | Project Manager | Task planning, dependency mapping |
| **Research** | Analyst | Best practices research, citations |
| **Coder** | Developer | Code implementation, tests, Docker |
| **QA** | Quality Assurance | Validation, code review |
| **Docs** | Technical Writer | README, documentation, release notes |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Supabase account
- OpenRouter API key
- SMTP server (optional, for notifications)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/multiagent-corp.git
cd multiagent-corp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials