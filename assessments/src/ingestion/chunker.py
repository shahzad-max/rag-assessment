"""
Chunking strategies for document processing
"""

from typing import List, Dict, Literal
import tiktoken
from dataclasses import dataclass, field
from src.utils import log, count_tokens


@dataclass
class Chunk:
    """Represents a document chunk"""
    text: str
    metadata: Dict
    chunk_id: str
    token_count: int
    start_char: int = 0
    end_char: int = 0


class DocumentChunker:
    """Implements multiple chunking strategies"""
    
    def __init__(
        self,
        strategy: Literal['fixed', 'semantic', 'sliding', 'hierarchical'] = 'semantic',
        chunk_size: int = 512,
        overlap: int = 50,
        encoding_name: str = 'cl100k_base'
    ):
        """
        Initialize chunker
        
        Args:
            strategy: Chunking strategy to use
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            encoding_name: Tokenizer encoding name
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        log.info(f"Initialized chunker: strategy={strategy}, size={chunk_size}, overlap={overlap}")
    
    def chunk_documents(self, documents: List[Dict]) -> List[Chunk]:
        """
        Chunk documents using selected strategy
        
        Args:
            documents: List of parsed documents
            
        Returns:
            List of Chunk objects
        """
        log.info(f"Chunking {len(documents)} documents using {self.strategy} strategy...")
        
        if self.strategy == 'fixed':
            chunks = self._fixed_size_chunking(documents)
        elif self.strategy == 'semantic':
            chunks = self._semantic_chunking(documents)
        elif self.strategy == 'sliding':
            chunks = self._sliding_window_chunking(documents)
        elif self.strategy == 'hierarchical':
            chunks = self._hierarchical_chunking(documents)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        log.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _fixed_size_chunking(self, documents: List[Dict]) -> List[Chunk]:
        """Fixed-size chunking with overlap"""
        chunks = []
        
        for doc in documents:
            text = doc['content']
            tokens = self.encoding.encode(text)
            
            # Create overlapping chunks
            start = 0
            chunk_idx = 0
            
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.encoding.decode(chunk_tokens)
                
                # Calculate character positions (approximate)
                start_char = len(self.encoding.decode(tokens[:start]))
                end_char = len(self.encoding.decode(tokens[:end]))
                
                chunk = Chunk(
                    text=chunk_text,
                    metadata={
                        **doc['metadata'],
                        'doc_type': doc['type'],
                        'doc_number': doc.get('number', ''),
                        'doc_title': doc.get('title', ''),
                        'chunk_index': chunk_idx,
                        'chunking_strategy': 'fixed',
                        'total_chunks': -1  # Will be updated later
                    },
                    chunk_id=f"{doc.get('number', 'unknown').replace(' ', '_')}_{chunk_idx}",
                    token_count=len(chunk_tokens),
                    start_char=start_char,
                    end_char=end_char
                )
                chunks.append(chunk)
                
                # Move to next chunk with overlap
                start += self.chunk_size - self.overlap
                chunk_idx += 1
            
            # Update total_chunks for all chunks from this document
            for i in range(len(chunks) - chunk_idx, len(chunks)):
                chunks[i].metadata['total_chunks'] = chunk_idx
        
        return chunks
    
    def _semantic_chunking(self, documents: List[Dict]) -> List[Chunk]:
        """
        Semantic chunking - preserve article boundaries
        Split only on paragraph boundaries if article is too large
        """
        chunks = []
        
        for doc in documents:
            # If document is small enough, keep as single chunk
            tokens = self.encoding.encode(doc['content'])
            
            if len(tokens) <= self.chunk_size:
                chunk = Chunk(
                    text=doc['content'],
                    metadata={
                        **doc['metadata'],
                        'doc_type': doc['type'],
                        'doc_number': doc.get('number', ''),
                        'doc_title': doc.get('title', ''),
                        'chunk_index': 0,
                        'chunking_strategy': 'semantic',
                        'total_chunks': 1
                    },
                    chunk_id=f"{doc.get('number', 'unknown').replace(' ', '_')}_0",
                    token_count=len(tokens),
                    start_char=0,
                    end_char=len(doc['content'])
                )
                chunks.append(chunk)
            else:
                # Split on paragraphs
                paragraphs = doc.get('paragraphs', [doc['content']])
                current_chunk = []
                current_tokens = 0
                chunk_idx = 0
                start_char = 0
                
                for para in paragraphs:
                    para_tokens = self.encoding.encode(para)
                    
                    # If single paragraph exceeds chunk size, split it
                    if len(para_tokens) > self.chunk_size:
                        # Save current chunk if any
                        if current_chunk:
                            chunk_text = '\n\n'.join(current_chunk)
                            chunk = Chunk(
                                text=chunk_text,
                                metadata={
                                    **doc['metadata'],
                                    'doc_type': doc['type'],
                                    'doc_number': doc.get('number', ''),
                                    'doc_title': doc.get('title', ''),
                                    'chunk_index': chunk_idx,
                                    'chunking_strategy': 'semantic',
                                    'total_chunks': -1
                                },
                                chunk_id=f"{doc.get('number', 'unknown').replace(' ', '_')}_{chunk_idx}",
                                token_count=current_tokens,
                                start_char=start_char,
                                end_char=start_char + len(chunk_text)
                            )
                            chunks.append(chunk)
                            chunk_idx += 1
                            start_char += len(chunk_text) + 2  # +2 for \n\n
                            current_chunk = []
                            current_tokens = 0
                        
                        # Split large paragraph using fixed-size chunking
                        para_start = 0
                        while para_start < len(para_tokens):
                            para_end = min(para_start + self.chunk_size, len(para_tokens))
                            para_chunk_tokens = para_tokens[para_start:para_end]
                            para_chunk_text = self.encoding.decode(para_chunk_tokens)
                            
                            chunk = Chunk(
                                text=para_chunk_text,
                                metadata={
                                    **doc['metadata'],
                                    'doc_type': doc['type'],
                                    'doc_number': doc.get('number', ''),
                                    'doc_title': doc.get('title', ''),
                                    'chunk_index': chunk_idx,
                                    'chunking_strategy': 'semantic',
                                    'total_chunks': -1
                                },
                                chunk_id=f"{doc.get('number', 'unknown').replace(' ', '_')}_{chunk_idx}",
                                token_count=len(para_chunk_tokens),
                                start_char=start_char,
                                end_char=start_char + len(para_chunk_text)
                            )
                            chunks.append(chunk)
                            chunk_idx += 1
                            start_char += len(para_chunk_text)
                            para_start += self.chunk_size - self.overlap
                    
                    elif current_tokens + len(para_tokens) > self.chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = '\n\n'.join(current_chunk)
                        chunk = Chunk(
                            text=chunk_text,
                            metadata={
                                **doc['metadata'],
                                'doc_type': doc['type'],
                                'doc_number': doc.get('number', ''),
                                'doc_title': doc.get('title', ''),
                                'chunk_index': chunk_idx,
                                'chunking_strategy': 'semantic',
                                'total_chunks': -1
                            },
                            chunk_id=f"{doc.get('number', 'unknown').replace(' ', '_')}_{chunk_idx}",
                            token_count=current_tokens,
                            start_char=start_char,
                            end_char=start_char + len(chunk_text)
                        )
                        chunks.append(chunk)
                        chunk_idx += 1
                        start_char += len(chunk_text) + 2
                        
                        # Start new chunk with current paragraph
                        current_chunk = [para]
                        current_tokens = len(para_tokens)
                    else:
                        current_chunk.append(para)
                        current_tokens += len(para_tokens)
                
                # Add remaining chunk
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = Chunk(
                        text=chunk_text,
                        metadata={
                            **doc['metadata'],
                            'doc_type': doc['type'],
                            'doc_number': doc.get('number', ''),
                            'doc_title': doc.get('title', ''),
                            'chunk_index': chunk_idx,
                            'chunking_strategy': 'semantic',
                            'total_chunks': chunk_idx + 1
                        },
                        chunk_id=f"{doc.get('number', 'unknown').replace(' ', '_')}_{chunk_idx}",
                        token_count=current_tokens,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                
                # Update total_chunks for all chunks from this document
                total = chunk_idx
                for chunk in chunks[-total:]:
                    chunk.metadata['total_chunks'] = total
        
        return chunks
    
    def _sliding_window_chunking(self, documents: List[Dict]) -> List[Chunk]:
        """Sliding window with configurable stride"""
        chunks = []
        stride = self.chunk_size - self.overlap
        
        for doc in documents:
            text = doc['content']
            tokens = self.encoding.encode(text)
            
            start = 0
            chunk_idx = 0
            
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.encoding.decode(chunk_tokens)
                
                start_char = len(self.encoding.decode(tokens[:start]))
                end_char = len(self.encoding.decode(tokens[:end]))
                
                chunk = Chunk(
                    text=chunk_text,
                    metadata={
                        **doc['metadata'],
                        'doc_type': doc['type'],
                        'doc_number': doc.get('number', ''),
                        'doc_title': doc.get('title', ''),
                        'chunk_index': chunk_idx,
                        'chunking_strategy': 'sliding',
                        'total_chunks': -1
                    },
                    chunk_id=f"{doc.get('number', 'unknown').replace(' ', '_')}_{chunk_idx}",
                    token_count=len(chunk_tokens),
                    start_char=start_char,
                    end_char=end_char
                )
                chunks.append(chunk)
                
                start += stride
                chunk_idx += 1
            
            # Update total_chunks
            for i in range(len(chunks) - chunk_idx, len(chunks)):
                chunks[i].metadata['total_chunks'] = chunk_idx
        
        return chunks
    
    def _hierarchical_chunking(self, documents: List[Dict]) -> List[Chunk]:
        """
        Hierarchical chunking with parent-child relationships
        Creates multiple levels of granularity
        """
        chunks = []
        
        for doc in documents:
            # Level 1: Full document (if not too large)
            full_tokens = self.encoding.encode(doc['content'])
            
            if len(full_tokens) <= self.chunk_size * 2:
                # Create parent chunk
                parent_chunk = Chunk(
                    text=doc['content'],
                    metadata={
                        **doc['metadata'],
                        'doc_type': doc['type'],
                        'doc_number': doc.get('number', ''),
                        'doc_title': doc.get('title', ''),
                        'chunk_index': 0,
                        'chunking_strategy': 'hierarchical',
                        'level': 'parent',
                        'total_chunks': 1
                    },
                    chunk_id=f"{doc.get('number', 'unknown').replace(' ', '_')}_parent",
                    token_count=len(full_tokens),
                    start_char=0,
                    end_char=len(doc['content'])
                )
                chunks.append(parent_chunk)
            
            # Level 2: Paragraph-level chunks (children)
            paragraphs = doc.get('paragraphs', [doc['content']])
            for para_idx, para in enumerate(paragraphs):
                para_tokens = self.encoding.encode(para)
                
                if len(para_tokens) > 10:  # Skip very short paragraphs
                    child_chunk = Chunk(
                        text=para,
                        metadata={
                            **doc['metadata'],
                            'doc_type': doc['type'],
                            'doc_number': doc.get('number', ''),
                            'doc_title': doc.get('title', ''),
                            'chunk_index': para_idx,
                            'chunking_strategy': 'hierarchical',
                            'level': 'child',
                            'parent_id': f"{doc.get('number', 'unknown').replace(' ', '_')}_parent",
                            'total_chunks': len(paragraphs)
                        },
                        chunk_id=f"{doc.get('number', 'unknown').replace(' ', '_')}_child_{para_idx}",
                        token_count=len(para_tokens),
                        start_char=0,
                        end_char=len(para)
                    )
                    chunks.append(child_chunk)
        
        return chunks

