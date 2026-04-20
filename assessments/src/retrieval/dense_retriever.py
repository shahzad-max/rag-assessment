"""
Dense retrieval using FAISS vector search
"""

import numpy as np
from typing import List, Optional
import faiss

from src.embeddings.multi_provider_generator import MultiProviderEmbeddingGenerator
from src.ingestion.chunker import Chunk
from src.utils import log
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Represents a retrieval result"""
    chunk_id: str
    text: str
    score: float
    metadata: dict
    rank: int
    chunk: Optional[Chunk] = None


class DenseRetriever:
    """Dense retrieval using FAISS vector similarity search"""
    
    def __init__(
        self,
        index: faiss.Index,
        chunks: List[Chunk],
        embedding_generator: MultiProviderEmbeddingGenerator
    ):
        """
        Initialize dense retriever
        
        Args:
            index: FAISS index
            chunks: List of document chunks
            embedding_generator: Generator for query embeddings
        """
        self.index = index
        self.chunks = chunks
        self.embedding_generator = embedding_generator
        
        # Create chunk ID to index mapping
        self.chunk_id_to_idx = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
        
        log.info(f"Initialized DenseRetriever with {len(chunks)} chunks")
        log.info(f"Index size: {index.ntotal} vectors")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using dense vector search
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search index using FAISS
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        # Flatten results (FAISS returns 2D arrays)
        distances = distances[0]
        indices = indices[0]
        
        # Convert to results
        results = []
        for rank, (idx, distance) in enumerate(zip(indices, distances), 1):
            if idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx]
                
                # Convert distance to similarity score
                # FAISS returns L2 distance for normalized vectors
                # For normalized vectors: similarity = 1 - (distance^2 / 2)
                # Or use inner product directly if using IndexFlatIP
                similarity = 1.0 - (distance / 2.0)
                similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
                
                result = RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(similarity),
                    metadata=chunk.metadata,
                    rank=rank,
                    chunk=chunk
                )
                results.append(result)
        
        log.debug(f"Dense retrieval: {len(results)} results for query: {query[:50]}...")
        
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
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """
        Get chunk by ID
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk object or None
        """
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self.chunks[idx]
        return None

