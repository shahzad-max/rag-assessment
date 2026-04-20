#!/usr/bin/env python3
"""
Test RAG System with All 20 EU AI Act Queries
Generates comprehensive evaluation report
"""

import os
import sys
import pickle
import faiss
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict
import time

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

# Test queries organized by category
TEST_QUERIES = {
    "fact_based": [
        "What are the prohibited AI practices according to the EU AI Act?",
        "What is the definition of an AI system under the EU AI Act?",
        "What are the obligations of providers of high-risk AI systems?",
        "What is the role of notified bodies in the EU AI Act?",
        "What constitutes a high-risk AI system?",
        "What are the transparency obligations for AI systems?",
        "What penalties can be imposed for violations of the EU AI Act?",
        "What are the requirements for technical documentation?",
        "What is the AI Office and what are its responsibilities?",
        "What are the conformity assessment procedures?",
    ],
    "abstract": [
        "How does the EU AI Act define risk-based approach?",
        "Why is the EU AI Act considered landmark legislation?",
        "What are the key differences between prohibited and high-risk AI systems?",
        "How does the EU AI Act address fundamental rights?",
    ],
    "reasoning": [
        "If a company deploys an AI system for social scoring, which articles does it violate?",
        "Can a company use an AI system for recruitment without human oversight?",
        "If an AI system causes harm, who is liable under the EU AI Act?",
    ],
    "comparative": [
        "How does the EU AI Act differ from the GDPR in terms of scope?",
        "How does the EU AI Act handle biometric identification compared to GDPR?",
        "What are the key similarities between the EU AI Act and product safety regulations?",
    ]
}

def initialize_system():
    """Initialize the RAG system components"""
    log.info("Initializing RAG system...")
    
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
    
    return hybrid_retriever

def test_query(retriever, query: str, query_id: int, category: str) -> Dict:
    """Test a single query and return results"""
    start_time = time.time()
    
    try:
        results = retriever.retrieve(
            query=query,
            top_k=int(os.getenv('FINAL_TOP_K', '5')),
            dense_k=int(os.getenv('DENSE_TOP_K', '20')),
            sparse_k=int(os.getenv('SPARSE_TOP_K', '20')),
            dense_weight=float(os.getenv('DENSE_WEIGHT', '0.5')),
            sparse_weight=float(os.getenv('SPARSE_WEIGHT', '0.5'))
        )
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "query_id": query_id,
            "query": query,
            "category": category,
            "success": True,
            "latency_ms": round(latency, 2),
            "num_results": len(results),
            "results": [
                {
                    "rank": i + 1,
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "text_preview": r.text[:int(os.getenv('CONTEXT_PREVIEW_LENGTH', '200'))] + "..." if len(r.text) > int(os.getenv('CONTEXT_PREVIEW_LENGTH', '200')) else r.text,
                    "metadata": r.metadata
                }
                for i, r in enumerate(results)
            ]
        }
    except Exception as e:
        log.error(f"Error testing query {query_id}: {e}")
        return {
            "query_id": query_id,
            "query": query,
            "category": category,
            "success": False,
            "error": str(e)
        }

def run_all_tests():
    """Run all 20 test queries"""
    print("=" * 100)
    print("EU AI ACT RAG SYSTEM - COMPREHENSIVE EVALUATION")
    print("=" * 100)
    print()
    
    # Initialize system
    retriever = initialize_system()
    
    # Run tests
    all_results = []
    query_id = 1
    
    for category, queries in TEST_QUERIES.items():
        print(f"\n{'=' * 100}")
        print(f"CATEGORY: {category.upper().replace('_', ' ')}")
        print(f"{'=' * 100}\n")
        
        for query in queries:
            print(f"Query {query_id}: {query}")
            print("-" * 100)
            
            result = test_query(retriever, query, query_id, category)
            all_results.append(result)
            
            if result["success"]:
                print(f"✓ Retrieved {result['num_results']} results in {result['latency_ms']}ms\n")
                
                for r in result["results"][:3]:  # Show top 3
                    print(f"  {r['rank']}. {r['chunk_id']} (Score: {r['score']})")
                    print(f"     Pages: {r['metadata'].get('page_numbers', 'N/A')}")
                    print(f"     Type: {r['metadata'].get('doc_type', 'N/A')}")
                    print(f"     Preview: {r['text_preview'][:150]}...")
                    print()
            else:
                print(f"✗ Error: {result['error']}\n")
            
            query_id += 1
    
    # Generate summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    
    successful = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]
    
    print(f"\nTotal Queries: {len(all_results)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(all_results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(all_results)*100:.1f}%)")
    
    avg_latency = 0.0
    if successful:
        avg_latency = sum(r["latency_ms"] for r in successful) / len(successful)
        print(f"\nAverage Latency: {avg_latency:.2f}ms")
        print(f"Min Latency: {min(r['latency_ms'] for r in successful):.2f}ms")
        print(f"Max Latency: {max(r['latency_ms'] for r in successful):.2f}ms")
    
    # Category breakdown
    print("\nResults by Category:")
    for category in TEST_QUERIES.keys():
        cat_results = [r for r in successful if r["category"] == category]
        print(f"  {category.replace('_', ' ').title()}: {len(cat_results)}/{len(TEST_QUERIES[category])} successful")
    
    # Save results to JSON
    output_dir = os.getenv('REPORT_OUTPUT_DIR', 'results')
    output_path = Path(f"{output_dir}/all_queries_evaluation.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_queries": len(all_results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful)/len(all_results)*100,
                "avg_latency_ms": avg_latency if successful else 0
            },
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Detailed results saved to: {output_path}")
    
    print("\n" + "=" * 100)
    print("EVALUATION COMPLETE")
    print("=" * 100)
    
    return all_results

if __name__ == "__main__":
    try:
        results = run_all_tests()
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    except Exception as e:
        log.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        print("\n❌ Tests failed!")
        sys.exit(1)

