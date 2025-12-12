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
        model: str = "anthropic/claude-3-sonnet",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a completion."""
        if not self.client:
            raise LLMError("HTTP client not initialized")
        
        start_time = time.time()
        
        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            **kwargs
        }
        
        try:
            response = self.client.post('/chat/completions', json=payload)
            if response.is_error:
                logger.error(f"OpenRouter Error Status: {response.status_code}")
                logger.error(f"OpenRouter Error Body: {response.text}")
            response.raise_for_status()
            data = response.json()
            
            return {
                'content': data['choices'][0]['message']['content'],
                'model': data.get('model', model),
                'usage': data.get('usage', {}),
                'elapsed': time.time() - start_time
            }
        except Exception as e:
            logger.error(f"LLM completion error: {e}")
            raise LLMError(f"Completion failed: {e}") from e
    
    def complete_json(
        self,
        messages: List[Dict[str, str]],
        model: str = "anthropic/claude-3-sonnet",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a JSON completion."""
        response = self.complete(messages=messages, model=model, **kwargs)
        content = response['content']
        
        # Parse JSON
        try:
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                content = content[start:end].strip()
            elif '```' in content:
                start = content.find('```') + 3
                end = content.find('```', start)
                content = content[start:end].strip()
            
            response['parsed'] = json.loads(content)
            return response
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise LLMError(f"Invalid JSON in response: {e}") from e
    
    def close(self):
        if self.client:
            self.client.close()
