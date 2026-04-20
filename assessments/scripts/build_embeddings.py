#!/usr/bin/env python3
"""
Build Embeddings from Processed Chunks
Generates embeddings and FAISS index from pre-processed chunks
"""

import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.multi_provider_generator import MultiProviderEmbeddingGenerator
from src.ingestion.chunker import Chunk
from src.utils import log


def load_chunks_from_json(chunks_file: str) -> List[Chunk]:
    """
    Load chunks from JSON and convert to Chunk objects
    
    Args:
        chunks_file: Path to chunks JSON file
    
    Returns:
        List of Chunk objects
    """
    log.info(f"Loading chunks from: {chunks_file}")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    chunks = []
    for chunk_data in chunks_data:
        chunk = Chunk(
            text=chunk_data['content'],
            metadata=chunk_data['metadata'],
            chunk_id=chunk_data['chunk_id'],
            token_count=chunk_data['token_count'],
            start_char=chunk_data.get('start_char', 0),
            end_char=chunk_data.get('end_char', 0)
        )
        chunks.append(chunk)
    
    log.info(f"✓ Loaded {len(chunks)} chunks")
    return chunks


def build_embeddings(
    chunks_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None
):
    """
    Build embeddings and FAISS index from processed chunks
    
    Args:
        chunks_file: Path to chunks JSON file
        output_dir: Directory to save outputs
        model_name: Embedding model name (from .env if None)
        batch_size: Batch size for embedding generation
    """
    # Get values from environment if not provided
    if chunks_file is None:
        data_dir = os.getenv('DATA_DIR', 'data')
        chunks_file = f"{data_dir}/processed/chunks.json"
    if output_dir is None:
        output_dir = os.getenv('INDEX_DIR', 'data/indexes')
    if batch_size is None:
        batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '100'))
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 80)
    log.info("BUILDING EMBEDDINGS AND FAISS INDEX")
    log.info("=" * 80)
    
    # Get model name from environment if not provided
    if model_name is None:
        model_name = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')
    
    log.info(f"Model: {model_name}")
    log.info(f"Batch size: {batch_size}")
    
    # Step 1: Load chunks
    log.info("\n[1/4] Loading chunks...")
    chunks = load_chunks_from_json(chunks_file)
    
    # Calculate statistics
    avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
    min_tokens = min(c.token_count for c in chunks)
    max_tokens = max(c.token_count for c in chunks)
    log.info(f"  - Average tokens: {avg_tokens:.0f}")
    log.info(f"  - Min tokens: {min_tokens}")
    log.info(f"  - Max tokens: {max_tokens}")
    
    # Save chunks as pickle for retrieval
    chunks_pkl = output_path / "chunks.pkl"
    with open(chunks_pkl, 'wb') as f:
        pickle.dump(chunks, f)
    log.info(f"✓ Saved chunks to: {chunks_pkl}")
    
    # Step 2: Initialize embedding generator with multi-provider support
    log.info(f"\n[2/4] Initializing multi-provider embedding generator...")
    log.info("Trying providers in order: OpenAI → WatsonX AI → Ollama")
    
    generator = MultiProviderEmbeddingGenerator(
        model_name=model_name,
        batch_size=batch_size,
        provider_priority=['openai', 'watsonx', 'ollama']
    )
    log.info(f"✓ Initialized with provider: {generator.provider}, model: {generator.model_name}")
    
    # Step 3: Generate embeddings
    log.info(f"\n[3/4] Generating embeddings...")
    log.info(f"  This may take a few minutes for {len(chunks)} chunks...")
    
    embeddings = generator.generate_for_chunks(chunks, show_progress=True)
    log.info(f"✓ Generated embeddings: {embeddings.shape}")
    
    # Save embeddings
    embeddings_file = output_path / "embeddings.pkl"
    generator.save_embeddings(
        embeddings,
        embeddings_file,
        metadata={
            'num_chunks': len(chunks),
            'model': model_name,
            'embedding_dim': embeddings.shape[1],
            'source_file': chunks_file
        }
    )
    log.info(f"✓ Saved embeddings to: {embeddings_file}")
    
    # Step 4: Build FAISS index
    log.info(f"\n[4/4] Building FAISS index...")
    
    # Get index type from environment
    index_type = os.getenv('FAISS_INDEX_TYPE', 'IndexHNSWFlat')
    log.info(f"  Index type: {index_type}")
    
    index = generator.build_faiss_index(embeddings)
    
    # Save index
    index_file = output_path / "faiss_index.bin"
    generator.save_index(index, index_file)
    log.info(f"✓ Saved FAISS index to: {index_file}")
    
    # Summary
    log.info("\n" + "=" * 80)
    log.info("BUILD COMPLETE")
    log.info("=" * 80)
    log.info(f"Chunks: {len(chunks)}")
    log.info(f"Embeddings: {embeddings.shape}")
    log.info(f"Index vectors: {index.ntotal}")
    log.info(f"\nOutput files:")
    log.info(f"  - {chunks_pkl}")
    log.info(f"  - {embeddings_file}")
    log.info(f"  - {index_file}")
    log.info("=" * 80)
    
    return {
        'num_chunks': len(chunks),
        'embedding_shape': embeddings.shape,
        'index_vectors': index.ntotal,
        'output_dir': str(output_path)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build embeddings from processed chunks")
    parser.add_argument(
        "--chunks",
        default="data/processed/chunks.json",
        help="Path to chunks JSON file"
    )
    parser.add_argument(
        "--output",
        default="data/indexes",
        help="Output directory for embeddings and index"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model name (default: from .env)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation"
    )
    
    args = parser.parse_args()
    
    try:
        stats = build_embeddings(
            chunks_file=args.chunks,
            output_dir=args.output,
            model_name=args.model,
            batch_size=args.batch_size
        )
        log.info("\n✅ Embeddings and index built successfully!")
        sys.exit(0)
    except Exception as e:
        log.error(f"\n❌ Error building embeddings: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

