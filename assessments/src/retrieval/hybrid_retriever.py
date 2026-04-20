"""
Hybrid retrieval combining dense and sparse methods
"""

from typing import List, Dict, Literal
from collections import defaultdict

from src.retrieval.dense_retriever import DenseRetriever, RetrievalResult
from src.retrieval.sparse_retriever import SparseRetriever
from src.utils import log, normalize_scores
from config import settings


class HybridRetriever:
    """Combines dense and sparse retrieval with fusion"""
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        fusion_method: Literal['rrf', 'weighted'] = None,
        rrf_k: int = None,
        alpha: float = None
    ):
        """
        Initialize hybrid retriever
        
        Args:
            dense_retriever: Dense retrieval component
            sparse_retriever: Sparse retrieval component (BM25)
            fusion_method: 'rrf' or 'weighted'
            rrf_k: Constant for RRF (default from settings)
            alpha: Weight for dense scores in weighted fusion
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_method = fusion_method or settings.fusion_method
        self.rrf_k = rrf_k or settings.rrf_k
        self.alpha = alpha or settings.alpha
        
        log.info(f"Initialized HybridRetriever with fusion method: {self.fusion_method}")
        if self.fusion_method == 'rrf':
            log.info(f"RRF k parameter: {self.rrf_k}")
        else:
            log.info(f"Weighted fusion alpha: {self.alpha}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        dense_k: int = None,
        sparse_k: int = None,
        dense_weight: float = None,
        sparse_weight: float = None
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid approach
        
        Args:
            query: Search query
            top_k: Number of final results (uses settings if None)
            dense_k: Number of results from dense retrieval
            sparse_k: Number of results from sparse retrieval
            dense_weight: Weight for dense scores (for weighted fusion)
            sparse_weight: Weight for sparse scores (for weighted fusion)
            
        Returns:
            List of RetrievalResult objects
        """
        top_k = top_k or settings.final_top_k
        dense_k = dense_k or settings.dense_top_k
        sparse_k = sparse_k or settings.sparse_top_k
        
        # Use provided weights or defaults
        if dense_weight is None or sparse_weight is None:
            dense_weight = settings.dense_weight
            sparse_weight = settings.sparse_weight
        
        log.debug(f"Hybrid retrieval for query: {query[:50]}...")
        log.debug(f"Dense top-k: {dense_k}, Sparse top-k: {sparse_k}, Final top-k: {top_k}")
        
        # Get results from both retrievers
        dense_results = self.dense_retriever.retrieve(query, top_k=dense_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=sparse_k)
        
        log.debug(f"Dense retrieved: {len(dense_results)}, Sparse retrieved: {len(sparse_results)}")
        
        # Fuse results
        if self.fusion_method == 'rrf':
            fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        else:
            fused_results = self._weighted_fusion(
                dense_results,
                sparse_results,
                dense_weight,
                sparse_weight
            )
        
        # Return top-k
        final_results = fused_results[:top_k]
        
        log.debug(f"Hybrid retrieval complete: {len(final_results)} results")
        
        return final_results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion
        
        Formula: score(d) = Σ(1 / (k + rank_i(d)))
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            
        Returns:
            Fused and sorted results
        """
        scores = defaultdict(float)
        metadata = {}
        
        # Add dense scores
        for result in dense_results:
            chunk_id = result.chunk_id
            scores[chunk_id] += 1.0 / (self.rrf_k + result.rank)
            metadata[chunk_id] = result
        
        # Add sparse scores
        for result in sparse_results:
            chunk_id = result.chunk_id
            scores[chunk_id] += 1.0 / (self.rrf_k + result.rank)
            if chunk_id not in metadata:
                metadata[chunk_id] = result
        
        # Sort by score
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create results
        results = []
        for rank, (chunk_id, score) in enumerate(sorted_ids, 1):
            result = metadata[chunk_id]
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=result.text,
                score=float(score),
                metadata=result.metadata,
                rank=rank,
                chunk=result.chunk
            ))
        
        return results
    
    def _weighted_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        dense_weight: float,
        sparse_weight: float
    ) -> List[RetrievalResult]:
        """
        Weighted linear combination
        
        Formula: score = α × dense_score + (1-α) × sparse_score
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            dense_weight: Weight for dense scores
            sparse_weight: Weight for sparse scores
            
        Returns:
            Fused and sorted results
        """
        # Collect scores
        dense_scores = {r.chunk_id: r.score for r in dense_results}
        sparse_scores = {r.chunk_id: r.score for r in sparse_results}
        
        # Normalize scores to [0, 1]
        dense_scores = normalize_scores(dense_scores)
        sparse_scores = normalize_scores(sparse_scores)
        
        # Combine scores
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined_scores = {}
        metadata = {}
        
        for chunk_id in all_ids:
            dense_score = dense_scores.get(chunk_id, 0.0)
            sparse_score = sparse_scores.get(chunk_id, 0.0)
            combined_scores[chunk_id] = (
                dense_weight * dense_score +
                sparse_weight * sparse_score
            )
            
            # Get metadata from either source
            for result in dense_results + sparse_results:
                if result.chunk_id == chunk_id:
                    metadata[chunk_id] = result
                    break
        
        # Sort and create results
        sorted_ids = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        results = []
        for rank, (chunk_id, score) in enumerate(sorted_ids, 1):
            result = metadata[chunk_id]
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=result.text,
                score=float(score),
                metadata=result.metadata,
                rank=rank,
                chunk=result.chunk
            ))
        
        return results
    
    def retrieve_with_query_type(
        self,
        query: str,
        query_type: str,
        top_k: int = None
    ) -> List[RetrievalResult]:
        """
        Retrieve with query-type specific weights
        
        Args:
            query: Search query
            query_type: One of 'fact', 'abstract', 'reasoning', 'comparative'
            top_k: Number of results
            
        Returns:
            List of RetrievalResult objects
        """
        # Get query-type specific weights
        dense_weight, sparse_weight = settings.get_query_type_weights(query_type)
        
        # Get max chunks for query type
        if top_k is None:
            top_k = settings.get_max_chunks_for_query_type(query_type)
        
        log.info(f"Retrieving for {query_type} query with weights: "
                f"dense={dense_weight}, sparse={sparse_weight}")
        
        return self.retrieve(
            query=query,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )

