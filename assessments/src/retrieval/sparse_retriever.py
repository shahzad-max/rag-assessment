"""
Sparse retrieval using BM25
"""

from typing import List
from rank_bm25 import BM25Okapi
import numpy as np

from src.ingestion.chunker import Chunk
from src.utils import log
from src.retrieval.dense_retriever import RetrievalResult
from config import settings


class SparseRetriever:
    """Sparse retrieval using BM25 keyword matching"""
    
    def __init__(
        self,
        chunks: List[Chunk],
        k1: float = None,
        b: float = None
    ):
        """
        Initialize sparse retriever
        
        Args:
            chunks: List of document chunks
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        self.chunks = chunks
        self.k1 = k1 or settings.bm25_k1
        self.b = b or settings.bm25_b
        
        log.info(f"Initializing SparseRetriever with {len(chunks)} chunks")
        log.info(f"BM25 parameters: k1={self.k1}, b={self.b}")
        
        # Build BM25 index
        self._build_index()
    
    def _build_index(self):
        """Build BM25 index from chunks"""
        # Tokenize all chunks
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in self.chunks]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.k1,
            b=self.b
        )
        
        log.info("BM25 index built successfully")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization with lowercasing
        # For production, consider using more sophisticated tokenization
        tokens = text.lower().split()
        return tokens
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Convert to results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = self.chunks[idx]
            score = float(scores[idx])
            
            # Normalize score to [0, 1] range
            # BM25 scores are unbounded, so we use a simple normalization
            # In practice, scores rarely exceed 50
            normalized_score = min(score / 50.0, 1.0)
            
            result = RetrievalResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=normalized_score,
                metadata=chunk.metadata,
                rank=rank,
                chunk=chunk
            )
            results.append(result)
        
        log.debug(f"Sparse retrieval: {len(results)} results for query: {query[:50]}...")
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve for multiple queries
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            
        Returns:
            List of result lists
        """
        all_results = []
        
        for query in queries:
            results = self.retrieve(query, top_k)
            all_results.append(results)
        
        return all_results
    
    def get_document_frequency(self, term: str) -> int:
        """
        Get document frequency for a term
        
        Args:
            term: Search term
            
        Returns:
            Number of documents containing the term
        """
        term_lower = term.lower()
        return sum(1 for doc in self.bm25.corpus_size if term_lower in doc)

