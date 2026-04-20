"""
Embedding generation and FAISS index management
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path
import pickle
from tqdm import tqdm
import time

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss

from config import settings
from src.utils import log, batch_items, count_tokens
from src.embeddings.models import get_model_config, estimate_cost
from src.ingestion.chunker import Chunk


class EmbeddingGenerator:
    """Generate embeddings using various models"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of embedding model (uses settings if None)
            batch_size: Batch size for generation (uses settings if None)
        """
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.model_config = get_model_config(self.model_name)
        
        log.info(f"Initializing EmbeddingGenerator with model: {self.model_name}")
        log.info(f"Model dimensions: {self.model_config.dimensions}")
        log.info(f"Batch size: {self.batch_size}")
        
        # Initialize model based on provider
        if self.model_config.provider == 'openai':
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.model = None
        elif self.model_config.provider in ['sentence-transformers', 'huggingface']:
            self.client = None
            self.model = SentenceTransformer(self.model_config.name)
            log.info(f"Loaded SentenceTransformer model: {self.model_config.name}")
        else:
            raise ValueError(f"Unsupported provider: {self.model_config.provider}")
    
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        log.info(f"Generating embeddings for {len(texts)} texts...")
        
        # Estimate cost for OpenAI models
        if self.model_config.provider == 'openai':
            total_tokens = sum(count_tokens(text) for text in texts)
            estimated_cost = estimate_cost(total_tokens, self.model_name)
            log.info(f"Estimated tokens: {total_tokens:,}")
            log.info(f"Estimated cost: ${estimated_cost:.4f}")
        
        # Generate embeddings in batches
        all_embeddings = []
        batches = batch_items(texts, self.batch_size)
        
        iterator = tqdm(batches, desc="Generating embeddings") if show_progress else batches
        
        for batch in iterator:
            batch_embeddings = self._generate_batch(batch)
            all_embeddings.append(batch_embeddings)
            
            # Rate limiting for API calls
            if self.model_config.provider == 'openai':
                time.sleep(0.1)  # Small delay to avoid rate limits
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
        log.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _generate_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a single batch
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if self.model_config.provider == 'openai':
            return self._generate_openai_batch(texts)
        else:
            return self._generate_local_batch(texts)
    
    def _generate_openai_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
            
        except Exception as e:
            log.error(f"Error generating OpenAI embeddings: {e}")
            raise
    
    def _generate_local_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local model"""
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings
            
        except Exception as e:
            log.error(f"Error generating local embeddings: {e}")
            raise
    
    def generate_for_chunks(
        self,
        chunks: List[Chunk],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of Chunk objects
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk.text for chunk in chunks]
        return self.generate_embeddings(texts, show_progress=show_progress)
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepath: Union[str, Path],
        metadata: Optional[dict] = None
    ):
        """
        Save embeddings to disk
        
        Args:
            embeddings: Numpy array of embeddings
            filepath: Path to save file
            metadata: Optional metadata to save with embeddings
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'embeddings': embeddings,
            'model_name': self.model_name,
            'dimensions': self.model_config.dimensions,
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        log.info(f"Saved embeddings to: {filepath}")
        log.info(f"Shape: {embeddings.shape}, Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
    
    def load_embeddings(self, filepath: Union[str, Path]) -> tuple[np.ndarray, dict]:
        """
        Load embeddings from disk
        
        Args:
            filepath: Path to embeddings file
            
        Returns:
            Tuple of (embeddings array, metadata dict)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        embeddings = data['embeddings']
        metadata = data.get('metadata', {})
        
        log.info(f"Loaded embeddings from: {filepath}")
        log.info(f"Shape: {embeddings.shape}, Model: {data.get('model_name')}")
        
        return embeddings, metadata
    
    def build_faiss_index(
        self,
        embeddings: np.ndarray,
        index_type: Optional[str] = None
    ) -> faiss.Index:
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings
            index_type: Type of FAISS index (uses settings if None)
            
        Returns:
            FAISS index
        """
        index_type = index_type or settings.faiss_index_type
        dimension = embeddings.shape[1]
        
        log.info(f"Building FAISS index: {index_type}")
        log.info(f"Embeddings shape: {embeddings.shape}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        if index_type == 'IndexFlatL2':
            # Exact search with L2 distance
            index = faiss.IndexFlatL2(dimension)
            
        elif index_type == 'IndexFlatIP':
            # Exact search with inner product (cosine similarity after normalization)
            index = faiss.IndexFlatIP(dimension)
            
        elif index_type == 'IndexHNSWFlat':
            # HNSW (Hierarchical Navigable Small World) - fast approximate search
            index = faiss.IndexHNSWFlat(dimension, settings.hnsw_m)
            index.hnsw.efConstruction = settings.hnsw_ef_construction
            index.hnsw.efSearch = settings.hnsw_ef_search
            
        elif index_type == 'IndexIVFFlat':
            # IVF (Inverted File) - memory efficient approximate search
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            log.info("Training IVF index...")
            index.train(embeddings)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add embeddings to index
        log.info("Adding embeddings to index...")
        index.add(embeddings)
        
        log.info(f"FAISS index built successfully. Total vectors: {index.ntotal}")
        
        return index
    
    def save_index(self, index: faiss.Index, filepath: Union[str, Path]):
        """
        Save FAISS index to disk
        
        Args:
            index: FAISS index
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, str(filepath))
        
        log.info(f"Saved FAISS index to: {filepath}")
        log.info(f"Index size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
    
    def load_index(self, filepath: Union[str, Path]) -> faiss.Index:
        """
        Load FAISS index from disk
        
        Args:
            filepath: Path to index file
            
        Returns:
            FAISS index
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        index = faiss.read_index(str(filepath))
        
        log.info(f"Loaded FAISS index from: {filepath}")
        log.info(f"Total vectors: {index.ntotal}")
        
        return index
    
    def search(
        self,
        index: faiss.Index,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search FAISS index
        
        Args:
            index: FAISS index
            query_embedding: Query embedding (1D or 2D array)
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = index.search(query_embedding, k)
        
        return distances[0], indices[0]

