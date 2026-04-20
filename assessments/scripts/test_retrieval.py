#!/usr/bin/env python3
"""
Test Retrieval System with Sample Query
Tests the hybrid retrieval without LLM generation
"""

import os
import sys
import pickle
import faiss
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.embeddings.multi_provider_generator import MultiProviderEmbeddingGenerator
from src.utils import log

def test_retrieval():
    """Test the retrieval system with a sample question"""
    
    # Sample question about EU AI Act
    question = "What are the prohibited AI practices according to the EU AI Act?"
    
    print("=" * 80)
    print("TESTING RETRIEVAL SYSTEM")
    print("=" * 80)
    print(f"\nQuestion: {question}\n")
    
    try:
        # Load chunks
        log.info("Loading chunks...")
        index_dir = os.getenv('INDEX_DIR', 'data/indexes')
        chunks_path = Path(f"{index_dir}/chunks.pkl")
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        log.info(f"Loaded {len(chunks)} chunks")
        
        # Initialize embedding generator
        log.info("Initializing embedding generator...")
        embedding_generator = MultiProviderEmbeddingGenerator()
        
        # Load FAISS index
        log.info("Loading FAISS index...")
        index_path = Path(f"{index_dir}/faiss_index.bin")
        index = faiss.read_index(str(index_path))
        log.info(f"Loaded FAISS index with {index.ntotal} vectors")
        
        # Initialize dense retriever
        log.info("Initializing dense retriever...")
        dense_retriever = DenseRetriever(
            index=index,
            chunks=chunks,
            embedding_generator=embedding_generator
        )
        
        # Initialize sparse retriever
        log.info("Initializing sparse retriever...")
        sparse_retriever = SparseRetriever(
            chunks=chunks,
            k1=float(os.getenv('BM25_K1', '1.5')),
            b=float(os.getenv('BM25_B', '0.75'))
        )
        
        # Initialize hybrid retriever
        log.info("Initializing hybrid retriever...")
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion_method=os.getenv('FUSION_METHOD', 'rrf'),  # type: ignore
            rrf_k=int(os.getenv('RRF_K', '60')),
            alpha=float(os.getenv('ALPHA', '0.5'))
        )
        
        # Retrieve documents
        log.info("Retrieving relevant documents...")
        results = hybrid_retriever.retrieve(
            query=question,
            top_k=int(os.getenv('FINAL_TOP_K', '5')),
            dense_k=int(os.getenv('DENSE_TOP_K', '20')),
            sparse_k=int(os.getenv('SPARSE_TOP_K', '20')),
            dense_weight=float(os.getenv('DENSE_WEIGHT', '0.5')),
            sparse_weight=float(os.getenv('SPARSE_WEIGHT', '0.5'))
        )
        
        # Display results
        print("\n" + "=" * 80)
        print(f"RETRIEVED {len(results)} DOCUMENTS")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{'─' * 80}")
            print(f"RESULT #{i}")
            print(f"{'─' * 80}")
            print(f"Chunk ID: {result.chunk_id}")
            print(f"Score: {result.score:.4f}")
            print(f"Metadata: {result.metadata}")
            print(f"\nContent Preview:")
            preview_len = int(os.getenv('CONTEXT_PREVIEW_LENGTH', '200'))
            print(result.text[:preview_len] + "..." if len(result.text) > preview_len else result.text)
        
        print("\n" + "=" * 80)
        print("RETRIEVAL TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        log.error(f"Error during retrieval test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_retrieval()
    if results:
        print("\n✅ Retrieval test passed!")
        sys.exit(0)
    else:
        print("\n❌ Retrieval test failed!")
        sys.exit(1)

