"""
FastAPI REST API for EU AI Act RAG System
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Set
import os
import time
from pathlib import Path
import pickle
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.embeddings.multi_provider_generator import MultiProviderEmbeddingGenerator
from src.generation.rag_pipeline import RAGPipeline
from src.generation.llm_client import LLMClient
from src.generation.prompt_manager import PromptManager
from src.generation.citation_tracker import CitationTracker
from src.evaluation.unified_metrics import calculate_comprehensive_metrics

# Initialize FastAPI app
app = FastAPI(
    title="EU AI Act RAG API",
    description="REST API for querying the EU AI Act using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for system components
rag_pipeline = None
embedding_generator = None
retriever = None
conversation_sessions = {}  # Store conversation history by session_id


# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    models: Dict[str, Any]
    vector_store: Dict[str, Any]
    system_info: Dict[str, Any]


class SingleQARequest(BaseModel):
    query: str = Field(..., description="User question about EU AI Act")
    top_k: int = Field(default_factory=lambda: int(os.getenv('FINAL_TOP_K', '5')), description="Number of results to retrieve")
    temperature: Optional[float] = Field(default_factory=lambda: float(os.getenv('LLM_TEMPERATURE', '0.0')), description="LLM temperature")
    max_tokens: Optional[int] = Field(default_factory=lambda: int(os.getenv('LLM_MAX_TOKENS', '1500')), description="Max tokens in response")
    # Optional ground truth for evaluation
    ground_truth_answer: Optional[str] = Field(None, description="Ground truth answer for evaluation")
    ground_truth_citations: Optional[Set[str]] = Field(None, description="Ground truth citations for evaluation")
    relevant_chunk_ids: Optional[Set[str]] = Field(None, description="Relevant chunk IDs for retrieval metrics")


class SingleQAResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    verified_citations: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = Field(None, description="Comprehensive evaluation metrics (if ground truth provided)")


class ConversationalQARequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    query: str = Field(..., description="User question")
    top_k: int = Field(default_factory=lambda: int(os.getenv('FINAL_TOP_K', '5')), description="Number of results to retrieve")
    temperature: Optional[float] = Field(default_factory=lambda: float(os.getenv('LLM_TEMPERATURE', '0.0')), description="LLM temperature")
    max_tokens: Optional[int] = Field(default_factory=lambda: int(os.getenv('LLM_MAX_TOKENS', '1500')), description="Max tokens in response")
    # Optional ground truth for evaluation
    ground_truth_answer: Optional[str] = Field(None, description="Ground truth answer for evaluation")
    ground_truth_citations: Optional[Set[str]] = Field(None, description="Ground truth citations for evaluation")
    relevant_chunk_ids: Optional[Set[str]] = Field(None, description="Relevant chunk IDs for retrieval metrics")


class ConversationalQAResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    conversation_history: List[Dict[str, str]]
    metadata: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = Field(None, description="Comprehensive evaluation metrics (if ground truth provided)")


class VectorStoreInfo(BaseModel):
    type: str
    total_vectors: int
    dimension: int
    index_type: str
    chunks_count: int
    embedding_model: str


def initialize_system():
    """Initialize RAG system components"""
    global rag_pipeline, embedding_generator, retriever
    
    try:
        # Load chunks
        index_dir = os.getenv('INDEX_DIR', 'data/indexes')
        chunks_path = Path(f"{index_dir}/chunks.pkl")
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        
        # Initialize embedding generator
        embedding_generator = MultiProviderEmbeddingGenerator()
        
        # Load FAISS index
        index_path = Path(f"{index_dir}/faiss_index.bin")
        index = faiss.read_index(str(index_path))
        
        # Initialize retrievers
        dense_retriever = DenseRetriever(
            index=index,
            chunks=chunks,
            embedding_generator=embedding_generator
        )
        
        sparse_retriever = SparseRetriever(
            chunks=chunks,
            k1=float(os.getenv('BM25_K1', '1.5')),
            b=float(os.getenv('BM25_B', '0.75'))
        )
        
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion_method=os.getenv('FUSION_METHOD', 'rrf'),  # type: ignore
            rrf_k=int(os.getenv('RRF_K', '60')),
            alpha=float(os.getenv('ALPHA', '0.5'))
        )
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=LLMClient(),
            prompt_manager=PromptManager(),
            citation_tracker=CitationTracker(),
            use_reranking=False,
            use_citation_verification=True
        )
        
        return True
    except Exception as e:
        print(f"Error initializing system: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    success = initialize_system()
    if not success:
        print("WARNING: System initialization failed!")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "EU AI Act RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "single_qa": "/api/qa",
            "conversational_qa": "/api/chat",
            "vector_store": "/api/vector-store"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns system status, active models, and vector store info
    """
    try:
        # Check environment variables for models
        models_status = {
            "openai": {
                "configured": bool(os.getenv("OPENAI_API_KEY")),
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                "status": "active" if os.getenv("OPENAI_API_KEY") else "inactive"
            },
            "watsonx": {
                "configured": bool(os.getenv("WATSONX_API_KEY")),
                "model": "ibm/granite-embedding-278m-multilingual",
                "status": "active" if os.getenv("WATSONX_API_KEY") else "inactive"
            },
            "ollama": {
                "configured": bool(os.getenv("OLLAMA_BASE_URL")),
                "model": os.getenv("OLLAMA_MODEL", "llama3"),
                "status": "active" if os.getenv("OLLAMA_BASE_URL") else "inactive"
            }
        }
        
        # Vector store info
        vector_store_info = {
            "type": "FAISS",
            "status": "active" if rag_pipeline else "inactive",
            "index_type": "IndexHNSWFlat",
            "total_vectors": retriever.dense_retriever.index.ntotal if retriever else 0,
            "dimension": 768,
            "chunks_count": len(retriever.dense_retriever.chunks) if retriever else 0
        }
        
        # System info
        system_info = {
            "rag_pipeline": "initialized" if rag_pipeline else "not initialized",
            "embedding_generator": "initialized" if embedding_generator else "not initialized",
            "retriever": "initialized" if retriever else "not initialized",
            "active_sessions": len(conversation_sessions)
        }
        
        return HealthResponse(
            status="healthy" if rag_pipeline else "unhealthy",
            models=models_status,
            vector_store=vector_store_info,
            system_info=system_info
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/api/qa", response_model=SingleQAResponse, tags=["Question Answering"])
async def single_qa(request: SingleQARequest):
    """
    Single question answering endpoint
    Processes a single query without conversation history
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    try:
        start_time = time.time()
        
        # Process query through RAG pipeline
        response = rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Format sources
        sources = [
            {
                "chunk_id": chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{i}",
                "text": chunk[:int(os.getenv('CONTEXT_PREVIEW_LENGTH', '200'))] + "..." if len(chunk) > int(os.getenv('CONTEXT_PREVIEW_LENGTH', '200')) else chunk,
                "score": None
            }
            for i, chunk in enumerate(response.retrieved_chunks[:5])
        ]
        
        # Format citations
        citations = [
            {
                "article": c.article,
                "recital": c.recital,
                "annex": c.annex,
                "text": c.text
            }
            for c in response.citations
        ]
        
        verified_citations = [
            {
                "article": c.article,
                "recital": c.recital,
                "annex": c.annex,
                "text": c.text
            }
            for c in response.verified_citations
        ]
        
        # Calculate comprehensive metrics if ground truth provided
        metrics = None
        if request.ground_truth_answer or request.ground_truth_citations or request.relevant_chunk_ids:
            # Extract predicted citations
            predicted_citations = set()
            import re
            citation_pattern = r'\[(Article \d+[a-z]?|Recital \d+|Annex [IVX]+)\]'
            for match in re.finditer(citation_pattern, response.answer):
                predicted_citations.add(match.group(1))
            
            # Get chunk IDs and context texts
            retrieved_chunk_ids = [
                chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{i}"
                for i, chunk in enumerate(response.retrieved_chunks)
            ]
            context_texts = [
                chunk.text if hasattr(chunk, 'text') else str(chunk)
                for chunk in response.retrieved_chunks
            ]
            
            # Calculate ALL comprehensive metrics
            metrics = calculate_comprehensive_metrics(
                query=request.query,
                answer=response.answer,
                retrieved_chunks=response.retrieved_chunks,
                retrieved_chunk_ids=retrieved_chunk_ids,
                context_texts=context_texts,
                predicted_citations=predicted_citations,
                latency_ms=latency_ms,
                ground_truth_answer=request.ground_truth_answer,
                ground_truth_citations=request.ground_truth_citations,
                relevant_chunk_ids=request.relevant_chunk_ids
            )
        
        return SingleQAResponse(
            query=response.query,
            answer=response.answer,
            citations=citations,
            verified_citations=verified_citations,
            sources=sources,
            metadata={
                "retrieval_time": response.retrieval_time,
                "generation_time": response.generation_time,
                "total_time": response.total_time,
                "num_citations": len(response.citations),
                "num_verified_citations": len(response.verified_citations)
            },
            metrics=metrics
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/api/chat", response_model=ConversationalQAResponse, tags=["Conversational QA"])
async def conversational_qa(request: ConversationalQARequest):
    """
    Conversational/multi-turn question answering endpoint
    Maintains conversation history per session
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    try:
        start_time = time.time()
        
        # Get or create conversation history for session
        if request.session_id not in conversation_sessions:
            conversation_sessions[request.session_id] = []
        
        conversation_history = conversation_sessions[request.session_id]
        
        # Process query with conversation history
        response = rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            conversation_history=conversation_history,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Update conversation history
        conversation_sessions[request.session_id].append({
            "query": request.query,
            "response": response.answer
        })
        
        # Keep only last 10 turns
        if len(conversation_sessions[request.session_id]) > 10:
            conversation_sessions[request.session_id] = conversation_sessions[request.session_id][-10:]
        
        # Format sources
        sources = [
            {
                "chunk_id": chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{i}",
                "text": chunk[:int(os.getenv('CONTEXT_PREVIEW_LENGTH', '200'))] + "..." if len(chunk) > int(os.getenv('CONTEXT_PREVIEW_LENGTH', '200')) else chunk
            }
            for i, chunk in enumerate(response.retrieved_chunks[:5])
        ]
        
        # Format citations
        citations = [
            {
                "article": c.article,
                "recital": c.recital,
                "annex": c.annex,
                "text": c.text
            }
            for c in response.citations
        ]
        
        # Calculate comprehensive metrics if ground truth provided
        metrics = None
        if request.ground_truth_answer or request.ground_truth_citations or request.relevant_chunk_ids:
            # Extract predicted citations
            predicted_citations = set()
            import re
            citation_pattern = r'\[(Article \d+[a-z]?|Recital \d+|Annex [IVX]+)\]'
            for match in re.finditer(citation_pattern, response.answer):
                predicted_citations.add(match.group(1))
            
            # Get chunk IDs and context texts
            retrieved_chunk_ids = [
                chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{i}"
                for i, chunk in enumerate(response.retrieved_chunks)
            ]
            context_texts = [
                chunk.text if hasattr(chunk, 'text') else str(chunk)
                for chunk in response.retrieved_chunks
            ]
            
            # Calculate ALL comprehensive metrics
            metrics = calculate_comprehensive_metrics(
                query=request.query,
                answer=response.answer,
                retrieved_chunks=response.retrieved_chunks,
                retrieved_chunk_ids=retrieved_chunk_ids,
                context_texts=context_texts,
                predicted_citations=predicted_citations,
                latency_ms=latency_ms,
                ground_truth_answer=request.ground_truth_answer,
                ground_truth_citations=request.ground_truth_citations,
                relevant_chunk_ids=request.relevant_chunk_ids
            )
        
        return ConversationalQAResponse(
            session_id=request.session_id,
            query=response.query,
            answer=response.answer,
            citations=citations,
            sources=sources,
            conversation_history=conversation_sessions[request.session_id],
            metadata={
                "retrieval_time": response.retrieval_time,
                "generation_time": response.generation_time,
                "total_time": response.total_time,
                "num_citations": len(response.citations),
                "conversation_turns": len(conversation_sessions[request.session_id])
            },
            metrics=metrics
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversational query processing failed: {str(e)}"
        )


@app.get("/api/vector-store", response_model=VectorStoreInfo, tags=["Vector Store"])
async def vector_store_info():
    """
    Get information about the vector store currently in use
    """
    if not retriever:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized"
        )
    
    try:
        return VectorStoreInfo(
            type="FAISS",
            total_vectors=retriever.dense_retriever.index.ntotal,
            dimension=768,
            index_type="IndexHNSWFlat",
            chunks_count=len(retriever.dense_retriever.chunks),
            embedding_model="ibm/granite-embedding-278m-multilingual"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vector store info: {str(e)}"
        )


@app.delete("/api/chat/{session_id}", tags=["Conversational QA"])
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
        return {"message": f"Conversation history cleared for session {session_id}"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', '8000'))
    )

