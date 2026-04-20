"""
Utility helper functions
"""

import tiktoken
from typing import List, Dict, Any
import json
from pathlib import Path


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken
    
    Args:
        text: Input text
        encoding_name: Tokenizer encoding name
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def truncate_text(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """
    Truncate text to maximum number of tokens
    
    Args:
        text: Input text
        max_tokens: Maximum number of tokens
        encoding_name: Tokenizer encoding name
        
    Returns:
        Truncated text
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def save_json(data: Any, filepath: Path) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> Any:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_context_with_numbers(chunks: List[str]) -> str:
    """
    Format context chunks with numbering
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Formatted context string
    """
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        formatted.append(f"[{i}] {chunk}")
    return '\n\n'.join(formatted)


def extract_article_number(text: str) -> str:
    """
    Extract article number from text
    
    Args:
        text: Text containing article reference
        
    Returns:
        Article number or empty string
    """
    import re
    
    # Pattern to match "Article X" or "Article X(Y)"
    pattern = r'Article\s+(\d+(?:\(\d+\))?(?:\([a-z]\))?)'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(1)
    
    return ""


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize scores to [0, 1] using min-max normalization
    
    Args:
        scores: Dictionary of scores
        
    Returns:
        Normalized scores
    """
    if not scores:
        return {}
    
    min_score = min(scores.values())
    max_score = max(scores.values())
    
    if max_score == min_score:
        return {k: 1.0 for k in scores.keys()}
    
    return {
        k: (v - min_score) / (max_score - min_score)
        for k, v in scores.items()
    }


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Batch items into chunks
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# classify_query_type function removed - using single dynamic prompt instead

