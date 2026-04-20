"""
Multi-Provider LLM Client
Automatically detects and uses available LLM provider (OpenAI → WatsonX → Ollama)
"""

import os
from typing import Optional, Dict, Any
from src.utils import log


class MultiProviderLLM:
    """Multi-provider LLM client with automatic fallback"""
    
    def __init__(self):
        """Initialize multi-provider LLM client"""
        self.provider = None
        self.client = None
        self.model = None
        
        log.info("Initializing Multi-Provider LLM Client")
        log.info("Provider priority: openai → watsonx → ollama")
        
        # Try providers in order
        if self._try_openai():
            return
        if self._try_watsonx():
            return
        if self._try_ollama():
            return
        
        raise RuntimeError("No LLM provider available. Please configure at least one provider.")
    
    def _try_openai(self) -> bool:
        """Try to initialize OpenAI"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            log.info("OpenAI API key not configured")
            return False
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = os.getenv('LLM_MODEL', 'gpt-4-turbo-preview')
            self.provider = 'openai'
            log.info(f"✓ Using OpenAI: {self.model}")
            return True
        except Exception as e:
            log.warning(f"Failed to initialize OpenAI: {e}")
            return False
    
    def _try_watsonx(self) -> bool:
        """Try to initialize WatsonX AI"""
        api_key = os.getenv('WATSONX_API_KEY')
        project_id = os.getenv('WATSONX_PROJECT_ID')
        
        if not api_key or not project_id:
            log.info("WatsonX credentials not configured")
            return False
        
        try:
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai import Credentials
            
            self.client = ModelInference(
                model_id=os.getenv('WATSONX_TEXT_MODEL', 'meta-llama/llama-3-70b-instruct'),
                credentials=Credentials(
                    api_key=api_key,
                    url=os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
                ),
                project_id=project_id
            )
            self.model = os.getenv('WATSONX_TEXT_MODEL', 'meta-llama/llama-3-70b-instruct')
            self.provider = 'watsonx'
            log.info(f"✓ Using WatsonX AI: {self.model}")
            return True
        except Exception as e:
            log.warning(f"Failed to initialize WatsonX: {e}")
            return False
    
    def _try_ollama(self) -> bool:
        """Try to initialize Ollama (local)"""
        try:
            import requests
            # Check if Ollama is running
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code != 200:
                log.info("Ollama not running")
                return False
            
            self.client = 'ollama'  # We'll use requests directly
            self.model = os.getenv('OLLAMA_TEXT_MODEL', 'llama3')
            self.provider = 'ollama'
            log.info(f"✓ Using Ollama (local): {self.model}")
            return True
        except Exception as e:
            log.info(f"Ollama not available: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1500
    ) -> str:
        """
        Generate text using available provider
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if self.provider == 'openai':
            return self._generate_openai(prompt, temperature, max_tokens)
        elif self.provider == 'watsonx':
            return self._generate_watsonx(prompt, temperature, max_tokens)
        elif self.provider == 'ollama':
            return self._generate_ollama(prompt, temperature, max_tokens)
        else:
            raise RuntimeError("No LLM provider initialized")
    
    def _generate_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"OpenAI generation error: {e}")
            raise
    
    def _generate_watsonx(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate using WatsonX AI"""
        try:
            response = self.client.generate_text(
                prompt=prompt,
                params={
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": 50
                }
            )
            return response.strip()
        except Exception as e:
            log.error(f"WatsonX generation error: {e}")
            raise
    
    def _generate_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate using Ollama"""
        try:
            import requests
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()['response'].strip()
        except Exception as e:
            log.error(f"Ollama generation error: {e}")
            raise
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about active provider"""
        return {
            'provider': self.provider,
            'model': self.model
        }

# Made with Bob
