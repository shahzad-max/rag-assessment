"""
Embedding model configurations and metadata
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class EmbeddingModel:
    """Configuration for an embedding model"""
    name: str
    provider: Literal['openai', 'sentence-transformers', 'huggingface']
    dimensions: int
    max_tokens: int
    cost_per_1k_tokens: float
    description: str


# Model configurations
EMBEDDING_MODELS = {
    'text-embedding-3-large': EmbeddingModel(
        name='text-embedding-3-large',
        provider='openai',
        dimensions=3072,
        max_tokens=8191,
        cost_per_1k_tokens=0.00013,
        description='OpenAI latest large embedding model - best quality'
    ),
    'text-embedding-3-small': EmbeddingModel(
        name='text-embedding-3-small',
        provider='openai',
        dimensions=1536,
        max_tokens=8191,
        cost_per_1k_tokens=0.00002,
        description='OpenAI latest small embedding model - cost effective'
    ),
    'text-embedding-ada-002': EmbeddingModel(
        name='text-embedding-ada-002',
        provider='openai',
        dimensions=1536,
        max_tokens=8191,
        cost_per_1k_tokens=0.0001,
        description='OpenAI legacy embedding model'
    ),
    'all-MiniLM-L6-v2': EmbeddingModel(
        name='all-MiniLM-L6-v2',
        provider='sentence-transformers',
        dimensions=384,
        max_tokens=256,
        cost_per_1k_tokens=0.0,
        description='Fast, lightweight model for local deployment'
    ),
    'all-mpnet-base-v2': EmbeddingModel(
        name='all-mpnet-base-v2',
        provider='sentence-transformers',
        dimensions=768,
        max_tokens=384,
        cost_per_1k_tokens=0.0,
        description='High quality sentence-transformers model'
    ),
    'e5-large-v2': EmbeddingModel(
        name='intfloat/e5-large-v2',
        provider='huggingface',
        dimensions=1024,
        max_tokens=512,
        cost_per_1k_tokens=0.0,
        description='State-of-the-art open-source embedding model'
    )
}


def get_model_config(model_name: str) -> EmbeddingModel:
    """
    Get configuration for an embedding model
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        EmbeddingModel configuration
        
    Raises:
        ValueError: If model name is not recognized
    """
    if model_name not in EMBEDDING_MODELS:
        available = ', '.join(EMBEDDING_MODELS.keys())
        raise ValueError(
            f"Unknown embedding model: {model_name}. "
            f"Available models: {available}"
        )
    
    return EMBEDDING_MODELS[model_name]


def list_available_models() -> list[str]:
    """Get list of available embedding models"""
    return list(EMBEDDING_MODELS.keys())


def get_models_by_provider(provider: str) -> list[EmbeddingModel]:
    """
    Get all models from a specific provider
    
    Args:
        provider: Provider name ('openai', 'sentence-transformers', 'huggingface')
        
    Returns:
        List of EmbeddingModel configurations
    """
    return [
        model for model in EMBEDDING_MODELS.values()
        if model.provider == provider
    ]


def estimate_cost(num_tokens: int, model_name: str) -> float:
    """
    Estimate cost for embedding generation
    
    Args:
        num_tokens: Number of tokens to embed
        model_name: Name of the embedding model
        
    Returns:
        Estimated cost in USD
    """
    model = get_model_config(model_name)
    return (num_tokens / 1000) * model.cost_per_1k_tokens

