"""
Document Parser for EUR-Lex documents
Supports HTML and PDF formats
Extracts structured content from EU AI Act documents
"""

from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple
import re
from pathlib import Path
from src.utils import log

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    log.warning("PyPDF2 not installed. PDF parsing will not be available.")


class EURLexParser:
    """Parse EUR-Lex documents (HTML and PDF)"""
    
    def __init__(self, file_path: str):
        """
        Initialize parser
        
        Args:
            file_path: Path to HTML or PDF file
        """
        self.file_path = Path(file_path)
        self.file_type = self._detect_file_type()
        self.soup: Optional[BeautifulSoup] = None
        log.info(f"Initialized parser for {self.file_type.upper()}: {file_path}")
    
    def _detect_file_type(self) -> str:
        """Detect file type from extension"""
        suffix = self.file_path.suffix.lower()
        if suffix == '.pdf':
            if not PDF_SUPPORT:
                raise ImportError("PyPDF2 is required for PDF parsing. Install with: pip install PyPDF2")
            return 'pdf'
        elif suffix in ['.html', '.htm']:
            return 'html'
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .html, .htm")
        
    def parse(self) -> List[Dict]:
        """
        Parse document and extract structured content
        
        Returns:
            List of dictionaries with:
                - type: 'article', 'recital', 'annex', 'chapter'
                - number: Article/recital number
                - title: Section title
                - content: Text content
                - metadata: Additional metadata (includes page_number for PDFs)
        """
        if self.file_type == 'pdf':
            return self._parse_pdf()
        else:
            return self._parse_html()
    
    def _parse_html(self) -> List[Dict]:
        """Parse HTML document"""
        log.info("Starting HTML parsing...")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.soup = BeautifulSoup(f.read(), 'lxml')
        
        documents = []
        
        # Extract recitals (preamble)
        log.info("Extracting recitals...")
        recitals = self._extract_recitals()
        documents.extend(recitals)
        log.info(f"Extracted {len(recitals)} recitals")
        
        # Extract chapters and articles
        log.info("Extracting articles...")
        articles = self._extract_articles()
        documents.extend(articles)
        log.info(f"Extracted {len(articles)} articles")
        
        # Extract annexes
        log.info("Extracting annexes...")
        annexes = self._extract_annexes()
        documents.extend(annexes)
        log.info(f"Extracted {len(annexes)} annexes")
        
        log.info(f"Total documents extracted: {len(documents)}")
        return documents
    
    def _parse_pdf(self) -> List[Dict]:
        """
        Parse PDF document with page tracking and section detection
        
        Returns:
            List of structured documents with page numbers
        """
        log.info("Starting PDF parsing...")
        
        # Extract text with page numbers
        pages_text = self._extract_pdf_text_with_pages()
        log.info(f"Extracted text from {len(pages_text)} pages")
        
        # Detect sections across pages
        documents = self._detect_pdf_sections(pages_text)
        log.info(f"Detected {len(documents)} sections")
        
        return documents
    
    def _extract_pdf_text_with_pages(self) -> List[Tuple[int, str]]:
        """
        Extract text from PDF with page numbers
        
        Returns:
            List of tuples (page_number, text)
        """
        pages_text = []
        
        with open(self.file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            total_pages = len(pdf_reader.pages)
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                pages_text.append((page_num + 1, text))  # 1-based page numbers
        
        return pages_text
    
    def _detect_pdf_sections(self, pages_text: List[Tuple[int, str]]) -> List[Dict]:
        """
        Detect sections (Articles, Recitals, Annexes) in PDF text
        
        Args:
            pages_text: List of (page_number, text) tuples
        
        Returns:
            List of structured documents with metadata
        """
        documents = []
        current_section = None
        current_content = []
        current_pages = []
        
        # Patterns for section detection
        article_pattern = re.compile(r'^Article\s+(\d+[a-z]?)\s*[-–—]?\s*(.*?)$', re.MULTILINE | re.IGNORECASE)
        recital_pattern = re.compile(r'^\((\d+)\)\s+', re.MULTILINE)
        annex_pattern = re.compile(r'^ANNEX\s+([IVX]+|[A-Z])\s*[-–—]?\s*(.*?)$', re.MULTILINE | re.IGNORECASE)
        chapter_pattern = re.compile(r'^CHAPTER\s+([IVX]+|[A-Z]|\d+)\s*[-–—]?\s*(.*?)$', re.MULTILINE | re.IGNORECASE)
        
        for page_num, text in pages_text:
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for new section
                new_section = None
                
                # Check for Article
                article_match = article_pattern.match(line)
                if article_match:
                    new_section = {
                        'type': 'article',
                        'number': article_match.group(1),
                        'title': article_match.group(2).strip() if article_match.group(2) else f"Article {article_match.group(1)}",
                        'pages': [page_num]
                    }
                
                # Check for Recital
                elif recital_pattern.match(line):
                    recital_match = recital_pattern.match(line)
                    new_section = {
                        'type': 'recital',
                        'number': recital_match.group(1),
                        'title': f"Recital {recital_match.group(1)}",
                        'pages': [page_num]
                    }
                
                # Check for Annex
                elif annex_pattern.match(line):
                    annex_match = annex_pattern.match(line)
                    new_section = {
                        'type': 'annex',
                        'number': annex_match.group(1),
                        'title': annex_match.group(2).strip() if annex_match.group(2) else f"Annex {annex_match.group(1)}",
                        'pages': [page_num]
                    }
                
                # Check for Chapter
                elif chapter_pattern.match(line):
                    chapter_match = chapter_pattern.match(line)
                    new_section = {
                        'type': 'chapter',
                        'number': chapter_match.group(1),
                        'title': chapter_match.group(2).strip() if chapter_match.group(2) else f"Chapter {chapter_match.group(1)}",
                        'pages': [page_num]
                    }
                
                # If new section found, save previous section
                if new_section:
                    if current_section and current_content:
                        documents.append({
                            'type': current_section['type'],
                            'number': f"{current_section['type'].title()} {current_section['number']}",
                            'title': current_section['title'],
                            'content': '\n'.join(current_content),
                            'paragraphs': current_content,
                            'metadata': {
                                'source': 'EU AI Act',
                                'regulation': '2024/1689',
                                'page_numbers': current_pages,
                                'page_start': min(current_pages),
                                'page_end': max(current_pages),
                                f"{current_section['type']}_number": current_section['number']
                            }
                        })
                    
                    current_section = new_section
                    current_content = [line]
                    current_pages = [page_num]
                else:
                    # Continue current section
                    if current_section:
                        current_content.append(line)
                        if page_num not in current_pages:
                            current_pages.append(page_num)
        
        # Add last section
        if current_section and current_content:
            documents.append({
                'type': current_section['type'],
                'number': f"{current_section['type'].title()} {current_section['number']}",
                'title': current_section['title'],
                'content': '\n'.join(current_content),
                'paragraphs': current_content,
                'metadata': {
                    'source': 'EU AI Act',
                    'regulation': '2024/1689',
                    'page_numbers': current_pages,
                    'page_start': min(current_pages),
                    'page_end': max(current_pages),
                    f"{current_section['type']}_number": current_section['number']
                }
            })
        
        return documents
    
    def _extract_recitals(self) -> List[Dict]:
        """Extract all recitals from the preamble"""
        recitals = []
        
        # Find recital elements - EUR-Lex uses specific patterns
        # Look for elements with "recital" in class or id
        recital_elements = self.soup.find_all(
            ['div', 'p', 'table'],
            class_=re.compile(r'recital', re.IGNORECASE)
        )
        
        # Also try finding by text pattern "Whereas:"
        if not recital_elements:
            # Alternative: find preamble section
            preamble = self.soup.find(
                ['div', 'section'],
                class_=re.compile(r'preamble|whereas', re.IGNORECASE)
            )
            if preamble:
                recital_elements = preamble.find_all(['p', 'div'])
        
        for idx, elem in enumerate(recital_elements, 1):
            text = elem.get_text(separator=' ', strip=True)
            
            # Skip empty or very short texts
            if len(text) < 20:
                continue
            
            # Extract recital number if present
            number_match = re.match(r'\((\d+)\)', text)
            number = number_match.group(1) if number_match else str(idx)
            
            recitals.append({
                'type': 'recital',
                'number': f"Recital {number}",
                'title': f"Recital {number}",
                'content': text,
                'paragraphs': [text],
                'metadata': {
                    'source': 'EU AI Act',
                    'regulation': '2024/1689',
                    'section': 'preamble'
                }
            })
        
        return recitals
    
    def _extract_articles(self) -> List[Dict]:
        """Extract all articles from the document"""
        articles = []
        
        # Find all article elements
        # EUR-Lex uses specific class names for articles
        article_elements = self.soup.find_all(
            ['div', 'article', 'section'],
            class_=re.compile(r'eli-subdivision.*article|^article', re.IGNORECASE)
        )
        
        # Alternative: find by article title pattern
        if not article_elements:
            article_elements = self.soup.find_all(
                text=re.compile(r'^Article\s+\d+', re.IGNORECASE)
            )
            article_elements = [elem.parent for elem in article_elements if elem.parent]
        
        for elem in article_elements:
            article = self._parse_article(elem)
            if article and article['content']:
                articles.append(article)
        
        return articles
    
    def _parse_article(self, element) -> Optional[Dict]:
        """Parse a single article element"""
        try:
            # Extract article number and title
            # Look for title element
            title_elem = element.find(
                ['h1', 'h2', 'h3', 'h4', 'p', 'div'],
                class_=re.compile(r'title|heading|eli-title', re.IGNORECASE)
            )
            
            if not title_elem:
                # Try to find by text pattern
                title_text = element.get_text(strip=True)[:200]
                title_match = re.search(r'Article\s+(\d+[a-z]?)\s*[-–—]?\s*(.*?)(?:\n|$)', title_text, re.IGNORECASE)
                if title_match:
                    number = title_match.group(1)
                    title = title_match.group(2).strip() if title_match.group(2) else f"Article {number}"
                else:
                    return None
            else:
                title_text = title_elem.get_text(strip=True)
                title_match = re.search(r'Article\s+(\d+[a-z]?)\s*[-–—]?\s*(.*)', title_text, re.IGNORECASE)
                if title_match:
                    number = title_match.group(1)
                    title = title_match.group(2).strip() if title_match.group(2) else f"Article {number}"
                else:
                    number = "Unknown"
                    title = title_text
            
            # Extract content
            # Remove title from content
            content_elem = element
            if title_elem and title_elem.parent == element:
                content_elem = element.find_next_sibling()
                if not content_elem:
                    content_elem = element
            
            content = content_elem.get_text(separator='\n', strip=True)
            
            # Extract paragraphs
            paragraphs = []
            para_elements = element.find_all(
                ['p', 'div'],
                class_=re.compile(r'paragraph|para|point', re.IGNORECASE)
            )
            
            if para_elements:
                for para in para_elements:
                    para_text = para.get_text(strip=True)
                    if para_text and len(para_text) > 10:
                        paragraphs.append(para_text)
            else:
                # Split content by newlines as fallback
                paragraphs = [p.strip() for p in content.split('\n') if p.strip() and len(p.strip()) > 10]
            
            return {
                'type': 'article',
                'number': f"Article {number}",
                'title': title,
                'content': content,
                'paragraphs': paragraphs,
                'metadata': {
                    'source': 'EU AI Act',
                    'regulation': '2024/1689',
                    'article_number': number
                }
            }
        except Exception as e:
            log.warning(f"Error parsing article: {e}")
            return None
    
    def _extract_annexes(self) -> List[Dict]:
        """Extract all annexes from the document"""
        annexes = []
        
        # Find annex elements
        annex_elements = self.soup.find_all(
            ['div', 'section'],
            class_=re.compile(r'annex', re.IGNORECASE)
        )
        
        # Alternative: find by text pattern
        if not annex_elements:
            annex_titles = self.soup.find_all(
                text=re.compile(r'^ANNEX\s+[IVX]+', re.IGNORECASE)
            )
            annex_elements = [elem.parent for elem in annex_titles if elem.parent]
        
        for elem in annex_elements:
            # Extract annex number
            title_elem = elem.find(['h1', 'h2', 'h3', 'p'])
            if title_elem:
                title_text = title_elem.get_text(strip=True)
                annex_match = re.search(r'ANNEX\s+([IVX]+|[A-Z])', title_text, re.IGNORECASE)
                if annex_match:
                    number = annex_match.group(1)
                    title = title_text
                else:
                    continue
            else:
                continue
            
            # Extract content
            content = elem.get_text(separator='\n', strip=True)
            
            # Extract sections/paragraphs
            paragraphs = []
            para_elements = elem.find_all(['p', 'div', 'li'])
            for para in para_elements:
                para_text = para.get_text(strip=True)
                if para_text and len(para_text) > 10:
                    paragraphs.append(para_text)
            
            annexes.append({
                'type': 'annex',
                'number': f"Annex {number}",
                'title': title,
                'content': content,
                'paragraphs': paragraphs,
                'metadata': {
                    'source': 'EU AI Act',
                    'regulation': '2024/1689',
                    'annex_number': number
                }
            })
        
        return annexes
    
    def extract_text_only(self) -> str:
        """
        Extract all text from document without structure
        
        Returns:
            Plain text content
        """
        if self.file_type == 'pdf':
            # For PDF, extract all text from all pages
            pages_text = self._extract_pdf_text_with_pages()
            return '\n\n'.join([text for _, text in pages_text])
        else:
            # For HTML
            if not self.soup:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.soup = BeautifulSoup(f.read(), 'lxml')
            
            # Remove script and style elements
            for script in self.soup(["script", "style"]):
                script.decompose()
        
        text = self.soup.get_text(separator='\n', strip=True)
        
        # Clean up multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text

