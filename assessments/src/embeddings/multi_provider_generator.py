"""
Multi-Provider Embedding Generator
Supports OpenAI, WatsonX AI, and Ollama with automatic fallback
Priority: OpenAI → WatsonX → Ollama
"""

import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import pickle
from tqdm import tqdm
import time
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

from src.ingestion.chunker import Chunk

# Simple token counter
def count_tokens(text: str) -> int:
    """Estimate token count (rough approximation)"""
    return len(text.split()) * 1.3

def batch_items(items: List, batch_size: int):
    """Yield successive batches from items"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


class MultiProviderEmbeddingGenerator:
    """Generate embeddings with multi-provider fallback"""
    
    # Model configurations
    MODELS = {
        'openai': {
            'text-embedding-3-large': {'dimensions': 3072, 'cost_per_1k': 0.00013},
            'text-embedding-3-small': {'dimensions': 1536, 'cost_per_1k': 0.00002},
        },
        'watsonx': {
            'ibm/granite-embedding-278m-multilingual': {'dimensions': 768, 'cost_per_1k': 0.0001},
            'ibm/slate-125m-english-rtrvr': {'dimensions': 384, 'cost_per_1k': 0.0001},
            'ibm/slate-30m-english-rtrvr': {'dimensions': 384, 'cost_per_1k': 0.0001},
        },
        'ollama': {
            'BAAI/bge-m3': {'dimensions': 1024, 'cost_per_1k': 0.0},
            'nomic-embed-text': {'dimensions': 768, 'cost_per_1k': 0.0},
        }
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 100,
        provider_priority: List[str] = ['openai', 'watsonx', 'ollama']
    ):
        """
        Initialize multi-provider embedding generator
        
        Args:
            model_name: Specific model name (auto-selects if None)
            batch_size: Batch size for generation
            provider_priority: Order of providers to try
        """
        self.batch_size = batch_size
        self.provider_priority = provider_priority
        self.provider = None
        self.client = None
        self.model = None
        self.model_name = model_name
        self.dimensions = None
        
        log.info("Initializing Multi-Provider Embedding Generator")
        log.info(f"Provider priority: {' → '.join(provider_priority)}")
        
        # Try to initialize providers in order
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Try to initialize providers in priority order"""
        for provider in self.provider_priority:
            try:
                if provider == 'openai':
                    if self._init_openai():
                        self.provider = 'openai'
                        log.info(f"✓ Using OpenAI: {self.model_name}")
                        return
                elif provider == 'watsonx':
                    if self._init_watsonx():
                        self.provider = 'watsonx'
                        log.info(f"✓ Using WatsonX AI: {self.model_name}")
                        return
                elif provider == 'ollama':
                    if self._init_ollama():
                        self.provider = 'ollama'
                        log.info(f"✓ Using Ollama (local): {self.model_name}")
                        return
            except Exception as e:
                log.warning(f"Failed to initialize {provider}: {e}")
                continue
        
        raise RuntimeError("No embedding provider available. Please configure OpenAI, WatsonX, or Ollama.")
    
    def _init_openai(self) -> bool:
        """Initialize OpenAI client"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key.startswith('sk-your-') or api_key.startswith('sk-5678'):
            log.info("OpenAI API key not configured")
            return False
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            
            # Test the API key with a small request
            test_response = self.client.embeddings.create(
                model='text-embedding-3-small',
                input=['test']
            )
            
            # If successful, set model
            if self.model_name and self.model_name in self.MODELS['openai']:
                model = self.model_name
            else:
                model = 'text-embedding-3-large'  # Default
            
            self.model_name = model
            self.dimensions = self.MODELS['openai'][model]['dimensions']
            return True
            
        except Exception as e:
            log.warning(f"OpenAI initialization failed: {e}")
            return False
    
    def _init_watsonx(self) -> bool:
        """Initialize WatsonX AI client"""
        api_key = os.getenv('WATSONX_API_KEY')
        project_id = os.getenv('WATSONX_PROJECT_ID')
        url = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
        
        if not api_key or not project_id:
            log.info("WatsonX AI credentials not configured")
            return False
        
        try:
            from ibm_watsonx_ai.foundation_models import Embeddings
            from ibm_watsonx_ai import Credentials
            
            # Get model from environment or use default
            # Don't use self.model_name if it's an OpenAI model
            if self.model_name and self.model_name in self.MODELS['watsonx']:
                model_id = self.model_name
            else:
                model_id = os.getenv('WATSONX_EMBEDDING_MODEL', 'ibm/granite-embedding-278m-multilingual')
            
            # Initialize credentials
            credentials = Credentials(
                url=url,
                api_key=api_key
            )
            
            # Initialize embeddings client
            self.client = Embeddings(
                model_id=model_id,
                credentials=credentials,
                project_id=project_id
            )
            
            # Test with a small request
            test_result = self.client.embed_documents(['test'])
            
            self.model_name = model_id
            self.dimensions = self.MODELS['watsonx'].get(model_id, {'dimensions': 768})['dimensions']
            return True
            
        except Exception as e:
            log.warning(f"WatsonX initialization failed: {e}")
            return False
    
    def _init_ollama(self) -> bool:
        """Initialize Ollama (local) client"""
        try:
            import ollama
            
            # Check if Ollama is running
            models = ollama.list()
            
            model_name = self.model_name if self.model_name else 'BAAI/bge-m3'
            
            # Check if model is available
            available_models = [m['name'] for m in models.get('models', [])]
            if model_name not in available_models:
                log.info(f"Pulling Ollama model: {model_name}")
                ollama.pull(model_name)
            
            # Test embedding generation
            test_result = ollama.embeddings(model=model_name, prompt='test')
            
            self.client = ollama
            self.model_name = model_name
            self.dimensions = len(test_result['embedding'])
            return True
            
        except Exception as e:
            log.warning(f"Ollama initialization failed: {e}")
            return False
    
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
        
        Returns:
            Numpy array of embeddings (n_texts, dimensions)
        """
        log.info(f"Generating embeddings for {len(texts)} texts...")
        log.info(f"Provider: {self.provider}")
        log.info(f"Model: {self.model_name}")
        
        # Estimate tokens and cost
        total_tokens = sum(count_tokens(text) for text in texts)
        cost_per_1k = self.MODELS[self.provider][self.model_name]['cost_per_1k']
        estimated_cost = (total_tokens / 1000) * cost_per_1k
        
        log.info(f"Estimated tokens: {total_tokens:,}")
        log.info(f"Estimated cost: ${estimated_cost:.4f}")
        
        # Generate embeddings in batches
        all_embeddings = []
        batches = list(batch_items(texts, self.batch_size))
        
        iterator = tqdm(batches, desc="Generating embeddings") if show_progress else batches
        
        for batch in iterator:
            try:
                if self.provider == 'openai':
                    batch_embeddings = self._generate_openai_batch(batch)
                elif self.provider == 'watsonx':
                    batch_embeddings = self._generate_watsonx_batch(batch)
                elif self.provider == 'ollama':
                    batch_embeddings = self._generate_ollama_batch(batch)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                log.error(f"Error generating embeddings for batch: {e}")
                raise
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        log.info(f"✓ Generated embeddings: {embeddings.shape}")
        
        return embeddings
    
    def _generate_openai_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def _generate_watsonx_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using WatsonX AI
        WatsonX has a 512 token limit PER REQUEST (not per text)
        Need to process texts one at a time to avoid batch token limit
        """
        MAX_CHARS = 1800  # Very conservative: ~450 tokens (512 limit with safety margin)
        embeddings = []
        
        for text in texts:
            # Aggressive character-based truncation to ensure we stay under limit
            # WatsonX tokenizer may count differently than our estimate
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS]
                log.debug(f"Truncated text to {MAX_CHARS} characters")
            
            # Process one text at a time
            result = self.client.embed_documents([text])
            embeddings.append(result[0])
        
        return embeddings
    
    def _generate_ollama_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama"""
        embeddings = []
        for text in texts:
            result = self.client.embeddings(model=self.model_name, prompt=text)
            embeddings.append(result['embedding'])
        return embeddings
    
    def generate_for_chunks(
        self,
        chunks: List[Chunk],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for chunks
        
        Args:
            chunks: List of Chunk objects
            show_progress: Show progress bar
        
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk.text for chunk in chunks]
        return self.generate_embeddings(texts, show_progress=show_progress)
    
    def build_faiss_index(
        self,
        embeddings: np.ndarray,
        index_type: str = 'IndexHNSWFlat'
    ):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings
            index_type: Type of FAISS index
        
        Returns:
            FAISS index
        """
        import faiss
        
        log.info(f"Building FAISS index: {index_type}")
        
        dimension = embeddings.shape[1]
        
        if index_type == 'IndexFlatL2':
            index = faiss.IndexFlatL2(dimension)
        elif index_type == 'IndexHNSWFlat':
            M = int(os.getenv('HNSW_M', 32))
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = int(os.getenv('HNSW_EF_CONSTRUCTION', 200))
        elif index_type == 'IndexIVFFlat':
            nlist = 100
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        index.add(embeddings)
        log.info(f"✓ Built index with {index.ntotal} vectors")
        
        return index
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepath: Path,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save embeddings to file"""
        data = {
            'embeddings': embeddings,
            'metadata': metadata or {},
            'provider': self.provider,
            'model': self.model_name,
            'dimensions': self.dimensions
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        log.info(f"✓ Saved embeddings to: {filepath}")
    
    def save_index(self, index, filepath: Path):
        """Save FAISS index to file"""
        import faiss
        faiss.write_index(index, str(filepath))
        log.info(f"✓ Saved index to: {filepath}")
    
    @staticmethod
    def load_embeddings(filepath: Path) -> Dict[str, Any]:
        """Load embeddings from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    
    @staticmethod
    def load_index(filepath: Path):
        """Load FAISS index from file"""
        import faiss
        return faiss.read_index(str(filepath))

