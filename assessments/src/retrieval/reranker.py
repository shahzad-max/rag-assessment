"""
Cross-encoder reranking for improved relevance
"""

from typing import List
from sentence_transformers import CrossEncoder
import torch

from src.retrieval.dense_retriever import RetrievalResult
from src.utils import log
from config import settings


class CrossEncoderReranker:
    """Rerank results using cross-encoder model"""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None
    ):
        """
        Initialize reranker
        
        Args:
            model_name: HuggingFace model name (uses settings if None)
            device: 'cuda' or 'cpu', auto-detect if None
        """
        self.model_name = model_name or settings.reranker_model
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        log.info(f"Initializing CrossEncoderReranker: {self.model_name}")
        log.info(f"Device: {self.device}")
        
        # Load model
        self.model = CrossEncoder(self.model_name, device=self.device)
        
        log.info("Reranker model loaded successfully")
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = None
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results
        
        Args:
            query: Search query
            results: List of RetrievalResult objects
            top_k: Number of results to return (uses settings if None)
            
        Returns:
            Reranked list of results
        """
        if not results:
            return []
        
        top_k = top_k or settings.rerank_top_k
        
        log.debug(f"Reranking {len(results)} results...")
        
        # Prepare pairs for cross-encoder
        pairs = [[query, result.text] for result in results]
        
        # Get scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Sort by score
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores and ranks
        reranked = []
        for rank, (result, score) in enumerate(scored_results[:top_k], 1):
            # Create new result with updated score and rank
            reranked_result = RetrievalResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=float(score),
                metadata=result.metadata,
                rank=rank,
                chunk=result.chunk
            )
            reranked.append(reranked_result)
        
        log.debug(f"Reranking complete: {len(reranked)} results")
        
        return reranked
    
    def batch_rerank(
        self,
        queries: List[str],
        results_list: List[List[RetrievalResult]],
        top_k: int = None
    ) -> List[List[RetrievalResult]]:
        """
        Rerank multiple result sets
        
        Args:
            queries: List of search queries
            results_list: List of result lists
            top_k: Number of results per query
            
        Returns:
            List of reranked result lists
        """
        reranked_list = []
        
        for query, results in zip(queries, results_list):
            reranked = self.rerank(query, results, top_k)
            reranked_list.append(reranked)
        
        return reranked_list

