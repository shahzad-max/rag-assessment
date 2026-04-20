"""
Citation extraction and verification for EU AI Act responses
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

from src.utils import log


@dataclass
class Citation:
    """Represents a citation to the EU AI Act"""
    article: Optional[str] = None
    recital: Optional[int] = None
    annex: Optional[str] = None
    paragraph: Optional[str] = None
    text: str = ""
    
    def __str__(self) -> str:
        """String representation of citation"""
        parts = []
        if self.article:
            parts.append(f"Article {self.article}")
        if self.recital:
            parts.append(f"Recital {self.recital}")
        if self.annex:
            parts.append(f"Annex {self.annex}")
        if self.paragraph:
            parts.append(f"paragraph {self.paragraph}")
        return ", ".join(parts) if parts else "Unknown citation"


class CitationTracker:
    """Tracks and verifies citations in generated responses"""
    
    # Regex patterns for citation extraction
    ARTICLE_PATTERN = r'Article\s+(\d+[a-z]?)'
    RECITAL_PATTERN = r'Recital\s+(\d+)'
    ANNEX_PATTERN = r'Annex\s+([IVX]+|[A-Z])'
    PARAGRAPH_PATTERN = r'paragraph\s+(\d+[a-z]?)'
    
    def __init__(self):
        """Initialize citation tracker"""
        self.citations: List[Citation] = []
        self.verified_citations: Set[str] = set()
        self.unverified_citations: Set[str] = set()
    
    def extract_citations(self, text: str) -> List[Citation]:
        """
        Extract citations from text
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        # Find all article references
        for match in re.finditer(self.ARTICLE_PATTERN, text, re.IGNORECASE):
            citation = Citation(
                article=match.group(1),
                text=match.group(0)
            )
            citations.append(citation)
        
        # Find all recital references
        for match in re.finditer(self.RECITAL_PATTERN, text, re.IGNORECASE):
            citation = Citation(
                recital=int(match.group(1)),
                text=match.group(0)
            )
            citations.append(citation)
        
        # Find all annex references
        for match in re.finditer(self.ANNEX_PATTERN, text, re.IGNORECASE):
            citation = Citation(
                annex=match.group(1),
                text=match.group(0)
            )
            citations.append(citation)
        
        self.citations.extend(citations)
        log.debug(f"Extracted {len(citations)} citations from text")
        
        return citations
    
    def verify_citations(
        self,
        citations: List[Citation],
        context_chunks: List[str]
    ) -> Tuple[List[Citation], List[Citation]]:
        """
        Verify citations against context chunks
        
        Args:
            citations: List of citations to verify
            context_chunks: Retrieved context chunks
            
        Returns:
            Tuple of (verified_citations, unverified_citations)
        """
        verified = []
        unverified = []
        
        # Create searchable context
        context_text = "\n".join(context_chunks).lower()
        
        for citation in citations:
            citation_str = str(citation).lower()
            
            # Check if citation appears in context
            if self._citation_in_context(citation, context_text):
                verified.append(citation)
                self.verified_citations.add(citation_str)
            else:
                unverified.append(citation)
                self.unverified_citations.add(citation_str)
        
        log.info(f"Verified {len(verified)}/{len(citations)} citations")
        
        return verified, unverified
    
    def _citation_in_context(self, citation: Citation, context: str) -> bool:
        """
        Check if citation appears in context
        
        Args:
            citation: Citation to check
            context: Context text (lowercase)
            
        Returns:
            True if citation found in context
        """
        # Check article
        if citation.article:
            pattern = f"article {citation.article}".lower()
            if pattern in context:
                return True
        
        # Check recital
        if citation.recital:
            pattern = f"recital {citation.recital}".lower()
            if pattern in context:
                return True
        
        # Check annex
        if citation.annex:
            pattern = f"annex {citation.annex}".lower()
            if pattern in context:
                return True
        
        return False
    
    def format_citations(
        self,
        citations: List[Citation],
        style: str = 'inline'
    ) -> str:
        """
        Format citations for display
        
        Args:
            citations: List of citations
            style: 'inline', 'footnote', or 'bibliography'
            
        Returns:
            Formatted citation string
        """
        if not citations:
            return ""
        
        if style == 'inline':
            return self._format_inline(citations)
        elif style == 'footnote':
            return self._format_footnote(citations)
        elif style == 'bibliography':
            return self._format_bibliography(citations)
        else:
            return self._format_inline(citations)
    
    def _format_inline(self, citations: List[Citation]) -> str:
        """Format citations inline"""
        unique_citations = list({str(c): c for c in citations}.values())
        return ", ".join(str(c) for c in unique_citations)
    
    def _format_footnote(self, citations: List[Citation]) -> str:
        """Format citations as footnotes"""
        unique_citations = list({str(c): c for c in citations}.values())
        footnotes = []
        for i, citation in enumerate(unique_citations, 1):
            footnotes.append(f"[{i}] {citation}")
        return "\n".join(footnotes)
    
    def _format_bibliography(self, citations: List[Citation]) -> str:
        """Format citations as bibliography"""
        unique_citations = list({str(c): c for c in citations}.values())
        bibliography = ["References:"]
        for citation in sorted(unique_citations, key=str):
            bibliography.append(f"- {citation}")
        return "\n".join(bibliography)
    
    def get_citation_stats(self) -> Dict:
        """
        Get citation statistics
        
        Returns:
            Dictionary with citation counts
        """
        return {
            'total_citations': len(self.citations),
            'verified_citations': len(self.verified_citations),
            'unverified_citations': len(self.unverified_citations),
            'verification_rate': (
                len(self.verified_citations) / len(self.citations) * 100
                if self.citations else 0.0
            )
        }
    
    def reset(self):
        """Reset citation tracker"""
        self.citations.clear()
        self.verified_citations.clear()
        self.unverified_citations.clear()
        log.debug("Citation tracker reset")

