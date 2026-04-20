"""
End-to-end RAG pipeline for EU AI Act question answering
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import time

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.llm_client import LLMClient
from src.generation.prompt_manager import PromptManager
from src.generation.citation_tracker import CitationTracker, Citation
from src.utils import log


@dataclass
class RAGResponse:
    """Response from RAG pipeline"""
    query: str
    answer: str
    query_type: str
    retrieved_chunks: List[str]
    reranked_chunks: List[str]
    citations: List[Citation]
    verified_citations: List[Citation]
    unverified_citations: List[Citation]
    retrieval_time: float
    generation_time: float
    total_time: float
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class RAGPipeline:
    """Complete RAG pipeline for EU AI Act QA"""
    
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        citation_tracker: Optional[CitationTracker] = None,
        use_reranking: bool = True,
        use_citation_verification: bool = True
    ):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: Hybrid retriever instance (required)
            reranker: Cross-encoder reranker instance
            llm_client: LLM client instance
            prompt_manager: Prompt manager instance
            citation_tracker: Citation tracker instance
            use_reranking: Whether to use reranking
            use_citation_verification: Whether to verify citations
        """
        self.retriever = retriever
        self.reranker = reranker if use_reranking else None
        self.llm_client = llm_client or LLMClient()
        self.prompt_manager = prompt_manager or PromptManager()
        self.citation_tracker = citation_tracker or CitationTracker()
        
        self.use_reranking = use_reranking
        self.use_citation_verification = use_citation_verification
        
        log.info(f"RAG Pipeline initialized (reranking={use_reranking}, citation_verification={use_citation_verification})")
    
    def query(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: int = 5,
        conversation_history: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> RAGResponse:
        """
        Process query through RAG pipeline
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            rerank_top_k: Number of chunks after reranking
            conversation_history: Previous conversation turns
            temperature: LLM temperature override
            max_tokens: Max tokens override
            
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant chunks
        retrieval_start = time.time()
        retrieved_results = self.retriever.retrieve(
            query=query,
            top_k=top_k
        )
        retrieved_chunks = [r.text for r in retrieved_results]
        retrieval_time = time.time() - retrieval_start
        
        log.info(f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.2f}s")
        
        # Step 3: Rerank if enabled
        if self.use_reranking and self.reranker:
            reranked_results = self.reranker.rerank(
                query=query,
                results=retrieved_results,
                top_k=rerank_top_k
            )
            reranked_chunks = [r.text for r in reranked_results]
            log.info(f"Reranked to top {len(reranked_chunks)} chunks")
        else:
            reranked_chunks = retrieved_chunks[:rerank_top_k]
        
        # Step 3: Generate prompt (using single dynamic template)
        prompt = self.prompt_manager.get_prompt(
            query=query,
            context=reranked_chunks,
            conversation_history=conversation_history
        )
        
        # Step 4: Generate answer
        generation_start = time.time()
        answer = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        generation_time = time.time() - generation_start
        
        log.info(f"Generated answer in {generation_time:.2f}s")
        
        # Step 5: Extract and verify citations
        citations = self.citation_tracker.extract_citations(answer)
        
        if self.use_citation_verification:
            verified, unverified = self.citation_tracker.verify_citations(
                citations=citations,
                context_chunks=reranked_chunks
            )
        else:
            verified = citations
            unverified = []
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Create response
        response = RAGResponse(
            query=query,
            answer=answer,
            query_type='dynamic',  # Single dynamic prompt handles all types
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=reranked_chunks,
            citations=citations,
            verified_citations=verified,
            unverified_citations=unverified,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            metadata={
                'top_k': top_k,
                'rerank_top_k': rerank_top_k,
                'use_reranking': self.use_reranking,
                'use_citation_verification': self.use_citation_verification,
                'num_citations': len(citations),
                'num_verified_citations': len(verified),
                'num_unverified_citations': len(unverified)
            }
        )
        
        log.info(f"Query processed in {total_time:.2f}s")
        
        return response
    
    def batch_query(
        self,
        queries: List[str],
        top_k: int = 10,
        rerank_top_k: int = 5,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[RAGResponse]:
        """
        Process multiple queries
        
        Args:
            queries: List of queries
            top_k: Number of chunks to retrieve
            rerank_top_k: Number of chunks after reranking
            temperature: LLM temperature override
            max_tokens: Max tokens override
            
        Returns:
            List of RAGResponse objects
        """
        responses = []
        
        log.info(f"Processing {len(queries)} queries in batch")
        
        for i, query in enumerate(queries, 1):
            log.info(f"Processing query {i}/{len(queries)}")
            
            response = self.query(
                query=query,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            responses.append(response)
        
        log.info(f"Batch processing complete: {len(responses)} responses")
        
        return responses
    
    def get_usage_stats(self) -> Dict:
        """
        Get usage statistics
        
        Returns:
            Dictionary with usage stats
        """
        llm_stats = self.llm_client.get_usage_stats()
        citation_stats = self.citation_tracker.get_citation_stats()
        
        return {
            'llm': llm_stats,
            'citations': citation_stats
        }
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.llm_client.reset_usage()
        self.citation_tracker.reset()
        log.info("Pipeline statistics reset")


def create_pipeline(
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
    use_reranking: bool = True,
    use_citation_verification: bool = True
) -> RAGPipeline:
    """
    Create a complete RAG pipeline
    
    Args:
        dense_retriever: Initialized dense retriever
        sparse_retriever: Initialized sparse retriever
        use_reranking: Whether to use reranking
        use_citation_verification: Whether to verify citations
        
    Returns:
        Configured RAGPipeline instance
    """
    # Initialize hybrid retriever
    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever
    )
    
    # Initialize other components
    reranker = CrossEncoderReranker() if use_reranking else None
    llm_client = LLMClient()
    prompt_manager = PromptManager()
    citation_tracker = CitationTracker()
    
    # Create pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm_client,
        prompt_manager=prompt_manager,
        citation_tracker=citation_tracker,
        use_reranking=use_reranking,
        use_citation_verification=use_citation_verification
    )
    
    log.info("RAG pipeline created successfully")
    
    return pipeline

