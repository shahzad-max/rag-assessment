#!/usr/bin/env python3
"""
Document Ingestion Script
Processes EU AI Act documents (HTML or PDF) and creates chunks with metadata
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.parser import EURLexParser
from src.ingestion.chunker import DocumentChunker
from src.utils import log

# Get settings from environment
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '512'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))


def ingest_document(file_path: str, output_dir: Optional[str] = None) -> Dict:
    """
    Ingest a document and create chunks
    
    Args:
        file_path: Path to HTML or PDF file
        output_dir: Directory to save output files
    
    Returns:
        Dictionary with ingestion statistics
    """
    log.info(f"Starting document ingestion: {file_path}")
    
    # Create output directory
    if output_dir is None:
        output_dir = os.getenv('DATA_DIR', 'data')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse document
    log.info("Parsing document...")
    parser = EURLexParser(file_path)
    documents = parser.parse()
    log.info(f"Parsed {len(documents)} sections")
    
    # Save parsed documents
    parsed_file = output_path / "parsed_documents.json"
    with open(parsed_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    log.info(f"Saved parsed documents to: {parsed_file}")
    
    # Create chunks
    log.info("Creating chunks...")
    chunker = DocumentChunker(
        strategy='semantic',
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )
    
    chunks = chunker.chunk_documents(documents)
    
    # Convert Chunk objects to dictionaries
    all_chunks = []
    for chunk in chunks:
        all_chunks.append({
            'chunk_id': chunk.chunk_id,
            'content': chunk.text,
            'metadata': chunk.metadata,
            'token_count': chunk.token_count,
            'start_char': chunk.start_char,
            'end_char': chunk.end_char
        })
    
    log.info(f"Created {len(all_chunks)} chunks")
    
    # Save chunks
    chunks_file = output_path / "chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    log.info(f"Saved chunks to: {chunks_file}")
    
    # Calculate statistics
    stats = {
        'total_documents': len(documents),
        'total_chunks': len(all_chunks),
        'document_types': {},
        'avg_chunk_size': sum(len(c['content']) for c in all_chunks) / len(all_chunks) if all_chunks else 0,
        'chunks_by_type': {}
    }
    
    # Count by document type
    for doc in documents:
        doc_type = doc['type']
        stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
    
    # Count chunks by type
    for chunk in all_chunks:
        chunk_type = chunk['metadata'].get('type', 'unknown')
        stats['chunks_by_type'][chunk_type] = stats['chunks_by_type'].get(chunk_type, 0) + 1
    
    # Save statistics
    stats_file = output_path / "ingestion_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    log.info(f"Saved statistics to: {stats_file}")
    
    # Print summary
    log.info("\n" + "="*60)
    log.info("INGESTION SUMMARY")
    log.info("="*60)
    log.info(f"Total Documents: {stats['total_documents']}")
    log.info(f"Total Chunks: {stats['total_chunks']}")
    log.info(f"Average Chunk Size: {stats['avg_chunk_size']:.0f} characters")
    log.info("\nDocuments by Type:")
    for doc_type, count in stats['document_types'].items():
        log.info(f"  {doc_type}: {count}")
    log.info("\nChunks by Type:")
    for chunk_type, count in stats['chunks_by_type'].items():
        log.info(f"  {chunk_type}: {count}")
    log.info("="*60)
    
    return stats


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest EU AI Act documents")
    parser.add_argument(
        "file_path",
        help="Path to HTML or PDF file"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for processed files (default: data)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file_path).exists():
        log.error(f"File not found: {args.file_path}")
        sys.exit(1)
    
    try:
        stats = ingest_document(args.file_path, args.output_dir)
        log.info("\n✅ Document ingestion completed successfully!")
        sys.exit(0)
    except Exception as e:
        log.error(f"\n❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

