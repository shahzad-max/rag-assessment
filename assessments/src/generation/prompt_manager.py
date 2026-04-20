"""
Prompt template management for different query types
"""

from typing import List, Dict, Optional
import yaml
from pathlib import Path

from src.utils import log, format_context_with_numbers


class PromptManager:
    """Manages prompt templates for different query types"""
    
    def __init__(self, config_path: str = 'config/prompts.yaml'):
        """
        Initialize prompt manager
        
        Args:
            config_path: Path to prompts configuration file
        """
        self.config_path = Path(config_path)
        self.prompts = self._load_prompts()
        
        log.info(f"Loaded {len(self.prompts)} prompt templates")
    
    def _load_prompts(self) -> Dict[str, Dict]:
        """Load prompts from YAML file"""
        if not self.config_path.exists():
            log.warning(f"Prompts file not found: {self.config_path}")
            return self._get_default_prompts()
        
        with open(self.config_path, 'r') as f:
            prompts = yaml.safe_load(f)
        
        return prompts
    
    def _get_default_prompts(self) -> Dict[str, Dict]:
        """Get default prompts if file not found"""
        return {
            'default': {
                'template': """You are an expert on the EU AI Act (Regulation 2024/1689).

Context:
{context}

Question: {question}

Answer based on the provided context and cite relevant articles:"""
            }
        }
    
    def get_prompt(
        self,
        query: str,
        context: List[str],
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate prompt for query using single dynamic template
        
        Args:
            query: User query
            context: List of retrieved context chunks
            conversation_history: Previous conversation turns
            
        Returns:
            Formatted prompt string
        """
        # Always use default template (single dynamic prompt)
        template_data = self.prompts.get('default', {})
        template = template_data.get('template', '') if template_data else ''
        
        # Format context
        context_str = format_context_with_numbers(context)
        
        # Format conversation history
        history_str = self._format_history(conversation_history)
        
        # Fill template
        prompt = template.format(
            context=context_str,
            question=query,
            history=history_str if history_str else ""
        )
        
        log.debug(f"Generated prompt ({len(prompt)} chars)")
        
        return prompt
    
    def _format_history(self, history: Optional[List[Dict]]) -> str:
        """
        Format conversation history
        
        Args:
            history: List of conversation turns
            
        Returns:
            Formatted history string
        """
        if not history:
            return ""
        
        formatted = ["Previous conversation:"]
        for turn in history:
            formatted.append(f"User: {turn.get('query', '')}")
            formatted.append(f"Assistant: {turn.get('response', '')}")
        
        return '\n'.join(formatted)
    
    def get_available_types(self) -> List[str]:
        """Get list of available prompt types"""
        return list(self.prompts.keys())
    
    def add_prompt_template(
        self,
        query_type: str,
        template: str,
        description: str = ""
    ):
        """
        Add a new prompt template
        
        Args:
            query_type: Type identifier
            template: Prompt template string
            description: Optional description
        """
        self.prompts[query_type] = {
            'template': template,
            'description': description
        }
        log.info(f"Added prompt template: {query_type}")
    
    def save_prompts(self, filepath: Optional[str] = None):
        """
        Save prompts to YAML file
        
        Args:
            filepath: Path to save file (uses config_path if None)
        """
        save_path = Path(filepath) if filepath else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.prompts, f, default_flow_style=False)
        
        log.info(f"Saved prompts to: {save_path}")

