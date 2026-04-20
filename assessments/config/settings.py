"""
Configuration management for EU AI Act RAG system
"""

from pydantic_settings import BaseSettings
from typing import Literal, List, Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # ============================================================================
    # API KEYS & AUTHENTICATION
    # ============================================================================
    openai_api_key: str
    
    # ============================================================================
    # EMBEDDING CONFIGURATION
    # ============================================================================
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    
    # ============================================================================
    # LLM CONFIGURATION
    # ============================================================================
    llm_model: str = "gpt-4-turbo-preview"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1500
    
    # ============================================================================
    # DOCUMENT PROCESSING PATHS
    # ============================================================================
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    index_dir: str = "data/indexes"
    
    # ============================================================================
    # CHUNKING CONFIGURATION
    # ============================================================================
    chunking_strategy: Literal['fixed', 'semantic', 'sliding', 'hierarchical'] = 'semantic'
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # ============================================================================
    # RETRIEVAL CONFIGURATION
    # ============================================================================
    # Dense Retrieval
    dense_top_k: int = 20
    dense_weight: float = 0.5
    
    # Sparse Retrieval (BM25)
    sparse_top_k: int = 20
    sparse_weight: float = 0.5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # Final Results
    final_top_k: int = 10
    
    # Fusion
    fusion_method: Literal['rrf', 'weighted'] = 'rrf'
    rrf_k: int = 60
    alpha: float = 0.5  # For weighted fusion
    
    # ============================================================================
    # RERANKING CONFIGURATION
    # ============================================================================
    use_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 10
    
    # ============================================================================
    # CONTEXT WINDOW OPTIMIZATION
    # ============================================================================
    max_context_chunks: int = 10
    context_ordering: Literal['bookend', 'alternating', 'descending'] = 'bookend'
    
    # ============================================================================
    # QUERY PROCESSING
    # ============================================================================
    use_query_classification: bool = True
    use_query_expansion: bool = False
    query_expansion_method: Literal['llm', 'synonym'] = 'llm'
    
    # ============================================================================
    # PERFORMANCE & OPTIMIZATION
    # ============================================================================
    embedding_batch_size: int = 100
    use_cache: bool = True
    cache_ttl: int = 3600
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    
    # ============================================================================
    # EVALUATION CONFIGURATION
    # ============================================================================
    calculate_precision: bool = True
    calculate_recall: bool = True
    calculate_mrr: bool = True
    calculate_ndcg: bool = True
    precision_k: str = "1,3,5,10"
    recall_k: str = "5,10,20"
    ndcg_k: str = "5,10"
    
    # ============================================================================
    # LOGGING & MONITORING
    # ============================================================================
    log_level: str = "INFO"
    log_rotation: str = "daily"
    log_retention_days: int = 30
    log_performance_metrics: bool = True
    log_queries: bool = True
    
    # ============================================================================
    # EXPERIMENT CONFIGURATION
    # ============================================================================
    experiment_mode: bool = False
    experiment_output_dir: str = "experiments/results"
    save_intermediate_results: bool = True
    
    # ============================================================================
    # API CONFIGURATION
    # ============================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    enable_cors: bool = True
    cors_origins: str = "*"
    rate_limit_per_minute: int = 60
    
    # ============================================================================
    # ADVANCED SETTINGS
    # ============================================================================
    faiss_index_type: str = "IndexHNSWFlat"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    use_gpu: bool = False
    random_seed: int = 42
    
    # ============================================================================
    # QUERY TYPE SPECIFIC SETTINGS
    # ============================================================================
    # Fact-based queries
    fact_dense_weight: float = 0.3
    fact_sparse_weight: float = 0.7
    
    # Abstract queries
    abstract_dense_weight: float = 0.7
    abstract_sparse_weight: float = 0.3
    
    # Reasoning queries
    reasoning_dense_weight: float = 0.5
    reasoning_sparse_weight: float = 0.5
    
    # Comparative queries
    comparative_dense_weight: float = 0.5
    comparative_sparse_weight: float = 0.5
    comparative_max_chunks: int = 15
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
    
    # ============================================================================
    # COMPUTED PROPERTIES
    # ============================================================================
    
    @property
    def data_path(self) -> Path:
        """Get data directory path"""
        return Path(self.data_dir)
    
    @property
    def index_path(self) -> Path:
        """Get index directory path"""
        return Path(self.index_dir)
    
    @property
    def raw_data_path(self) -> Path:
        """Get raw data directory path"""
        return Path(self.raw_data_dir)
    
    @property
    def processed_data_path(self) -> Path:
        """Get processed data directory path"""
        return Path(self.processed_data_dir)
    
    @property
    def experiment_output_path(self) -> Path:
        """Get experiment output directory path"""
        return Path(self.experiment_output_dir)
    
    @property
    def precision_k_values(self) -> List[int]:
        """Parse precision K values"""
        return [int(k.strip()) for k in self.precision_k.split(',')]
    
    @property
    def recall_k_values(self) -> List[int]:
        """Parse recall K values"""
        return [int(k.strip()) for k in self.recall_k.split(',')]
    
    @property
    def ndcg_k_values(self) -> List[int]:
        """Parse NDCG K values"""
        return [int(k.strip()) for k in self.ndcg_k.split(',')]
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins"""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(',')]
    
    def get_query_type_weights(self, query_type: str) -> tuple[float, float]:
        """
        Get retrieval weights for specific query type
        
        Args:
            query_type: One of 'fact', 'abstract', 'reasoning', 'comparative'
            
        Returns:
            Tuple of (dense_weight, sparse_weight)
        """
        if query_type == 'fact':
            return (self.fact_dense_weight, self.fact_sparse_weight)
        elif query_type == 'abstract':
            return (self.abstract_dense_weight, self.abstract_sparse_weight)
        elif query_type == 'reasoning':
            return (self.reasoning_dense_weight, self.reasoning_sparse_weight)
        elif query_type == 'comparative':
            return (self.comparative_dense_weight, self.comparative_sparse_weight)
        else:
            return (self.dense_weight, self.sparse_weight)
    
    def get_max_chunks_for_query_type(self, query_type: str) -> int:
        """
        Get maximum chunks for specific query type
        
        Args:
            query_type: One of 'fact', 'abstract', 'reasoning', 'comparative'
            
        Returns:
            Maximum number of chunks
        """
        if query_type == 'comparative':
            return self.comparative_max_chunks
        return self.max_context_chunks
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.data_path,
            self.raw_data_path,
            self.processed_data_path,
            self.index_path,
            self.experiment_output_path,
            Path("logs")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_settings(self) -> bool:
        """
        Validate settings for consistency
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check weights sum to 1.0 (approximately)
        if abs((self.dense_weight + self.sparse_weight) - 1.0) > 0.01:
            raise ValueError("Dense and sparse weights must sum to 1.0")
        
        # Check chunk size is reasonable
        if self.chunk_size < 50 or self.chunk_size > 4096:
            raise ValueError("Chunk size must be between 50 and 4096 tokens")
        
        # Check overlap is less than chunk size
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        # Check top-k values are positive
        if self.dense_top_k <= 0 or self.sparse_top_k <= 0 or self.final_top_k <= 0:
            raise ValueError("Top-K values must be positive")
        
        # Check final_top_k is not greater than retrieval top-k
        max_retrieval_k = max(self.dense_top_k, self.sparse_top_k)
        if self.final_top_k > max_retrieval_k:
            raise ValueError(f"Final top-K ({self.final_top_k}) cannot exceed retrieval top-K ({max_retrieval_k})")
        
        return True


# Global settings instance
try:
    settings = Settings()
    settings.validate_settings()
    settings.create_directories()
except Exception as e:
    # If .env doesn't exist or is invalid, create a default instance
    # This allows imports to work even without .env file
    import os
    if not os.path.exists('.env'):
        print("Warning: .env file not found. Using default settings.")
        print("Please copy .env.example to .env and configure your API keys.")
    settings = None

