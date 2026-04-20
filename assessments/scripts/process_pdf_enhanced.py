"""
Enhanced script to process EU AI Act PDF with page numbers and section detection
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import re
from typing import List, Dict, Optional, Tuple
import PyPDF2
from dataclasses import dataclass, asdict


@dataclass
class EnhancedChunk:
    """Enhanced chunk with page and section metadata"""
    text: str
    chunk_id: str
    metadata: Dict
    
    def to_dict(self):
        return asdict(self)


def detect_section(text: str) -> Optional[str]:
    """
    Detect section name from text (Article, Recital, Annex, Chapter, etc.)
    
    Args:
        text: Text to analyze
        
    Returns:
        Section name or None
    """
    # Patterns for different sections
    patterns = [
        (r'^Article\s+(\d+[a-z]?)', 'Article'),
        (r'^ARTICLE\s+(\d+[a-z]?)', 'Article'),
        (r'Recital\s+\((\d+)\)', 'Recital'),
        (r'RECITAL\s+\((\d+)\)', 'Recital'),
        (r'^ANNEX\s+([IVX]+|[A-Z])', 'Annex'),
        (r'^Annex\s+([IVX]+|[A-Z])', 'Annex'),
        (r'^CHAPTER\s+([IVX]+)', 'Chapter'),
        (r'^Chapter\s+([IVX]+)', 'Chapter'),
        (r'^SECTION\s+(\d+)', 'Section'),
        (r'^Section\s+(\d+)', 'Section'),
    ]
    
    # Check first 200 characters for section markers
    text_start = text[:200].strip()
    
    for pattern, section_type in patterns:
        match = re.search(pattern, text_start, re.MULTILINE | re.IGNORECASE)
        if match:
            section_num = match.group(1)
            return f"{section_type} {section_num}"
    
    return None


def extract_text_with_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract text from PDF with page numbers
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of (page_number, text) tuples
    """
    print(f"📄 Extracting text from: {pdf_path}")
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        print(f"   Total pages: {num_pages}")
        
        pages_text = []
        for i, page in enumerate(pdf_reader.pages, 1):
            if i % 50 == 0:
                print(f"   Processing page {i}/{num_pages}...")
            
            page_text = page.extract_text()
            pages_text.append((i, page_text))
        
        total_chars = sum(len(text) for _, text in pages_text)
        print(f"✓ Extracted {total_chars:,} characters from {num_pages} pages\n")
        
        return pages_text


def enhanced_chunk_text(
    pages_text: List[Tuple[int, str]], 
    chunk_size: int = 1000, 
    overlap: int = 100
) -> List[EnhancedChunk]:
    """
    Enhanced chunking with page numbers and section detection
    
    Args:
        pages_text: List of (page_number, text) tuples
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        
    Returns:
        List of enhanced chunks
    """
    print(f"✂️  Chunking text (size={chunk_size}, overlap={overlap})...")
    
    chunks = []
    chunk_id = 0
    current_section = None
    
    # Process each page
    for page_num, page_text in pages_text:
        # Detect section at start of page
        page_section = detect_section(page_text)
        if page_section:
            current_section = page_section
        
        # Split page into chunks
        start = 0
        while start < len(page_text):
            end = start + chunk_size
            chunk_text = page_text[start:end]
            
            # Try to break at sentence boundary
            if end < len(page_text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > chunk_size * 0.7:
                    end = start + break_point + 1
                    chunk_text = page_text[start:end]
            
            # Detect section in this chunk
            chunk_section = detect_section(chunk_text) or current_section
            
            chunk = EnhancedChunk(
                text=chunk_text.strip(),
                chunk_id=f"chunk_{chunk_id:04d}",
                metadata={
                    'page_number': page_num,
                    'section': chunk_section,
                    'start_char': start,
                    'end_char': end,
                    'length': len(chunk_text),
                    'has_section_marker': detect_section(chunk_text) is not None
                }
            )
            
            if chunk.text:  # Only add non-empty chunks
                chunks.append(chunk)
                chunk_id += 1
            
            start = end - overlap
    
    print(f"✓ Created {len(chunks)} chunks")
    print(f"   Avg length: {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars")
    
    # Count sections
    sections = {}
    for chunk in chunks:
        section = chunk.metadata.get('section', 'Unknown')
        sections[section] = sections.get(section, 0) + 1
    
    print(f"   Sections found: {len(sections)}")
    for section, count in sorted(sections.items())[:10]:
        print(f"     - {section}: {count} chunks")
    if len(sections) > 10:
        print(f"     ... and {len(sections) - 10} more sections")
    print()
    
    return chunks


def save_enhanced_chunks(chunks: List[EnhancedChunk], output_dir: str):
    """Save enhanced chunks with metadata"""
    print(f"💾 Saving enhanced chunks to: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save chunks
    chunks_file = output_path / 'chunks.json'
    chunks_data = [c.to_dict() for c in chunks]
    
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(chunks)} chunks to: {chunks_file}")
    
    # Analyze metadata
    page_range = (
        min(c.metadata['page_number'] for c in chunks),
        max(c.metadata['page_number'] for c in chunks)
    )
    
    sections = set(c.metadata.get('section') for c in chunks if c.metadata.get('section'))
    
    # Save metadata
    metadata = {
        'num_chunks': len(chunks),
        'total_chars': sum(len(c.text) for c in chunks),
        'avg_chunk_size': sum(len(c.text) for c in chunks) / len(chunks),
        'page_range': page_range,
        'num_pages': page_range[1] - page_range[0] + 1,
        'num_sections': len(sections),
        'sections': sorted(list(sections)),
        'source': 'OJ_L_202401689_EN_TXT.pdf',
        'processing': 'enhanced_with_page_and_section'
    }
    
    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to: {metadata_file}")
    print(f"   Page range: {page_range[0]}-{page_range[1]}")
    print(f"   Sections: {len(sections)}")
    print()
    
    return metadata


def main():
    print("=" * 80)
    print("EU AI ACT RAG SYSTEM - ENHANCED PDF PROCESSING")
    print("=" * 80)
    print()
    
    # Configuration
    pdf_path = 'OJ_L_202401689_EN_TXT.pdf'
    output_dir = 'data'
    chunk_size = 1000  # characters
    overlap = 100  # characters
    
    # Check if PDF exists
    if not Path(pdf_path).exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        print("Please ensure the PDF is in the project root directory")
        return 1
    
    # Step 1: Extract text with page numbers
    pages_text = extract_text_with_pages(pdf_path)
    
    # Save full text with page markers
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    full_text_file = output_path / 'extracted_text_with_pages.txt'
    with open(full_text_file, 'w', encoding='utf-8') as f:
        for page_num, text in pages_text:
            f.write(f"\n{'='*80}\n")
            f.write(f"PAGE {page_num}\n")
            f.write(f"{'='*80}\n\n")
            f.write(text)
            f.write("\n")
    
    print(f"✓ Saved text with page markers to: {full_text_file}\n")
    
    # Step 2: Enhanced chunking
    chunks = enhanced_chunk_text(pages_text, chunk_size, overlap)
    
    # Step 3: Save enhanced chunks
    metadata = save_enhanced_chunks(chunks, output_dir)
    
    # Summary
    print("=" * 80)
    print("✅ ENHANCED PROCESSING COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  📄 Source: {pdf_path}")
    print(f"  📝 Pages: {metadata['num_pages']} (pages {metadata['page_range'][0]}-{metadata['page_range'][1]})")
    print(f"  ✂️  Chunks: {metadata['num_chunks']}")
    print(f"  📊 Avg chunk size: {metadata['avg_chunk_size']:.0f} chars")
    print(f"  🏷️  Sections detected: {metadata['num_sections']}")
    print(f"  💾 Output: {output_dir}/")
    print()
    print("Enhanced features:")
    print("  ✓ Page numbers tracked for each chunk")
    print("  ✓ Section names detected (Articles, Recitals, Annexes)")
    print("  ✓ Metadata enriched for better retrieval")
    print()
    print("Files created:")
    print(f"  - {output_dir}/extracted_text_with_pages.txt")
    print(f"  - {output_dir}/chunks.json (with page & section metadata)")
    print(f"  - {output_dir}/metadata.json")
    print()
    print("Next steps:")
    print("  1. Set OpenAI API key in .env file")
    print("  2. Build embeddings: python scripts/build_embeddings.py")
    print("  3. Run evaluation: python scripts/run_evaluation.py")
    print()
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


