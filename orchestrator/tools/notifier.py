"""
SMTP Email Notifier
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SMTPNotifier:
    """SMTP-based email notification service."""
    
    def __init__(self):
        self.host = os.getenv('SMTP_HOST', '')
        self.port = int(os.getenv('SMTP_PORT', 587))
        self.username = os.getenv('SMTP_USERNAME', '')
        self.password = os.getenv('SMTP_PASSWORD', '')
        self.from_address = os.getenv('SMTP_FROM', 'noreply@multiagent.local')
        self.enabled = bool(self.host and self.username)
        
        if self.enabled:
            logger.info("SMTP notifier initialized")
        else:
            logger.info("SMTP notifier disabled (no configuration)")
    
    def _send_email(self, to_email: str, subject: str, body: str) -> bool:
        if not self.enabled:
            logger.debug(f"Email skipped (disabled): {subject}")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = self.from_address
            msg['To'] = to_email
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_email}: {subject}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_workflow_started(self, email: str, workflow_id: str, request_summary: str) -> bool:
        return self._send_email(
            email,
            f"Workflow Started: {workflow_id}",
            f"Your workflow has been initiated.\n\nWorkflow ID: {workflow_id}\nRequest: {request_summary}"
        )
    
    def send_workflow_completed(self, email: str, workflow_id: str, duration: str, iterations: int) -> bool:
        return self._send_email(
            email,
            f"Workflow Completed: {workflow_id}",
            f"Your workflow has completed.\n\nWorkflow ID: {workflow_id}\nDuration: {duration}\nIterations: {iterations}"
        )
    
    def send_workflow_failed(self, email: str, workflow_id: str, error_message: str) -> bool:
        return self._send_email(
            email,
            f"Workflow Failed: {workflow_id}",
            f"Your workflow has failed.\n\nWorkflow ID: {workflow_id}\nError: {error_message}"
        )
