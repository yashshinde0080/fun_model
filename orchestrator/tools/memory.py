"""
Memory Layer
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class MemoryStore:
    """Local memory store using SQLite."""

    def __init__(self, db_path: str = "storage/data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"Memory store initialized: {db_path}")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workflow_state (
                    workflow_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS agent_context (
                    id INTEGER PRIMARY KEY,
                    workflow_id TEXT,
                    agent_name TEXT,
                    context TEXT,
                    created_at TEXT
                );
            """)

    def save_workflow_state(self, workflow_id: str, state: Dict[str, Any]) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO workflow_state VALUES (?, ?, ?, ?)",
                (workflow_id, json.dumps(state), now, now)
            )

    def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT state FROM workflow_state WHERE workflow_id = ?",
                (workflow_id,)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def add_conversation_message(self, workflow_id: str, agent: str, role: str, content: str) -> None:
        """Add a message to conversation history."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        context = json.dumps({
            'role': role,
            'content': content,
            'timestamp': now
        })

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO agent_context (workflow_id, agent_name, context, created_at) VALUES (?, ?, ?, ?)",
                (workflow_id, agent, context, now)
            )

    def get_conversation_history(self, workflow_id: str, agent: str) -> List[Dict[str, Any]]:
        """Get conversation history for an agent."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT context FROM agent_context WHERE workflow_id = ? AND agent_name = ? ORDER BY created_at",
                (workflow_id, agent)
            ).fetchall()
            return [json.loads(row[0]) for row in rows]
