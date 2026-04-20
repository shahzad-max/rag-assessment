"""
LLM client for answer generation
"""

from typing import Optional, Dict, Any
import time
from openai import OpenAI, OpenAIError

from src.utils import log, count_tokens
from config import settings


class LLMClient:
    """Client for LLM-based answer generation"""
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM client
        
        Args:
            model: Model name (uses settings if None)
            temperature: Sampling temperature (uses settings if None)
            max_tokens: Maximum tokens in response (uses settings if None)
            api_key: OpenAI API key (uses settings if None)
        """
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        
        api_key = api_key or settings.openai_api_key
        self.client = OpenAI(api_key=api_key)
        
        log.info(f"Initialized LLMClient: {self.model}")
        log.info(f"Temperature: {self.temperature}, Max tokens: {self.max_tokens}")
        
        # Track usage
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        prompt_tokens = count_tokens(prompt)
        log.debug(f"Generating response (prompt tokens: {prompt_tokens})...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response
            answer = response.choices[0].message.content
            
            # Track usage
            usage = response.usage
            if usage:
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens
                
                # Estimate cost (approximate for GPT-4-turbo)
                prompt_cost = usage.prompt_tokens * 0.01 / 1000  # $0.01 per 1K tokens
                completion_cost = usage.completion_tokens * 0.03 / 1000  # $0.03 per 1K tokens
                query_cost = prompt_cost + completion_cost
                self.total_cost += query_cost
                
                log.debug(f"Response generated: {usage.completion_tokens} tokens")
                log.debug(f"Query cost: ${query_cost:.4f}")
            
            return answer if answer else ""
            
        except OpenAIError as e:
            log.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            log.error(f"Error generating response: {e}")
            raise
    
    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate with automatic retry on failure
        
        Args:
            prompt: Input prompt
            max_retries: Maximum number of retries
            retry_delay: Delay between retries (seconds)
            **kwargs: Additional arguments for generate()
            
        Returns:
            Generated text
        """
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    log.warning(f"Generation failed (attempt {attempt + 1}/{max_retries}): {e}")
                    log.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    log.error(f"Generation failed after {max_retries} attempts")
                    raise
        
        # Should never reach here, but for type safety
        return ""
    
    def generate_streaming(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Generate response with streaming
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Yields:
            Text chunks as they are generated
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            log.error(f"Error in streaming generation: {e}")
            raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics
        
        Returns:
            Dictionary with usage stats
        """
        return {
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_prompt_tokens + self.total_completion_tokens,
            'total_cost': self.total_cost,
            'model': self.model
        }
    
    def reset_usage(self):
        """Reset usage statistics"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        log.debug("Usage statistics reset")
    
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        log.info("Usage statistics reset")

