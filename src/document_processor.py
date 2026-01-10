"""
Document Processing Module
Handles ingestion, chunking, and indexing of long-form narratives

Note: Uses Pathway when available (Linux/MacOS), falls back to 
standalone implementation on Windows.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import re

# Try to import Pathway - it's primarily Linux/MacOS
try:
    import pathway as pw
    from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
    from pathway.xpacks.llm.splitters import TokenCountSplitter
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False
    print("Note: Pathway not available on this platform. Using standalone implementation.")


class NarrativeDocumentProcessor:
    """
    Document processor for long-form narratives.
    Handles chunking, embedding preparation, and text extraction.
    
    Uses Pathway when available, otherwise falls back to standalone methods.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Initialize Pathway components if available
        if PATHWAY_AVAILABLE:
            try:
                self.embedder = SentenceTransformerEmbedder(model=embedding_model)
                self.splitter = TokenCountSplitter(
                    min_token=100,
                    max_token=chunk_size,
                    encoding_name="cl100k_base"
                )
            except Exception as e:
                print(f"Pathway initialization error: {e}")
                self.embedder = None
                self.splitter = None
        else:
            self.embedder = None
            self.splitter = None
        
    def load_book(self, book_path: Path) -> str:
        """Load and clean a book text file."""
        with open(book_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # Clean the text - remove Project Gutenberg headers/footers
        text = self._clean_gutenberg_text(text)
        return text
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """Remove Project Gutenberg boilerplate text."""
        # Find start of actual content
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "*** START OF THIS PROJECT GUTENBERG EBOOK",
            "*END*THE SMALL PRINT",
        ]
        for marker in start_markers:
            if marker in text:
                idx = text.find(marker)
                # Find the next line after the marker
                next_newline = text.find('\n', idx)
                if next_newline != -1:
                    text = text[next_newline + 1:]
                break
        
        # Find end of actual content
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG EBOOK",
            "*** END OF THIS PROJECT GUTENBERG EBOOK",
            "End of Project Gutenberg",
            "End of the Project Gutenberg"
        ]
        for marker in end_markers:
            if marker in text:
                idx = text.find(marker)
                text = text[:idx]
                break
        
        return text.strip()
    
    def chunk_text(self, text: str, book_name: str) -> List[Dict]:
        """
        Split text into overlapping chunks with metadata.
        Uses a sliding window approach for better context preservation.
        """
        chunks = []
        
        # Split by chapters first if possible
        chapter_pattern = r'(CHAPTER|Chapter|BOOK|Book|PART|Part)\s+[IVXLCDM\d]+[.\s:]'
        chapter_splits = re.split(f'({chapter_pattern})', text)
        
        if len(chapter_splits) > 1:
            # Process chapter by chapter
            current_chapter = ""
            chapter_num = 0
            
            for i, part in enumerate(chapter_splits):
                if re.match(chapter_pattern, part):
                    if current_chapter.strip():
                        chunks.extend(self._chunk_section(
                            current_chapter, book_name, f"chapter_{chapter_num}"
                        ))
                    current_chapter = part
                    chapter_num += 1
                else:
                    current_chapter += part
            
            # Don't forget the last chapter
            if current_chapter.strip():
                chunks.extend(self._chunk_section(
                    current_chapter, book_name, f"chapter_{chapter_num}"
                ))
        else:
            # No clear chapter structure, use sliding window
            chunks = self._chunk_section(text, book_name, "full_text")
        
        return chunks
    
    def _chunk_section(
        self, 
        text: str, 
        book_name: str, 
        section_id: str
    ) -> List[Dict]:
        """Chunk a section of text using sliding window."""
        chunks = []
        
        # Clean whitespace
        text = ' '.join(text.split())
        
        if len(text) < self.chunk_size:
            if text.strip():
                chunks.append({
                    "text": text.strip(),
                    "book_name": book_name,
                    "section_id": section_id,
                    "chunk_id": 0
                })
            return chunks
        
        # Sliding window chunking
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                search_start = max(start + self.chunk_size - 100, start)
                search_end = min(start + self.chunk_size + 100, len(text))
                search_text = text[search_start:search_end]
                
                # Find sentence boundaries
                sentence_ends = []
                for pattern in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                    idx = search_text.rfind(pattern)
                    if idx != -1:
                        sentence_ends.append(search_start + idx + len(pattern))
                
                if sentence_ends:
                    end = max(sentence_ends)
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "book_name": book_name,
                    "section_id": section_id,
                    "chunk_id": chunk_id
                })
                chunk_id += 1
            
            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text) - self.chunk_overlap:
                break
        
        return chunks
    
    def extract_character_mentions(
        self, 
        text: str, 
        character_name: str
    ) -> List[str]:
        """
        Extract passages that mention a specific character.
        Useful for focused evidence retrieval.
        """
        mentions = []
        
        # Create variations of the character name
        name_parts = character_name.split()
        search_terms = [character_name]
        search_terms.extend(name_parts)
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for i, sentence in enumerate(sentences):
            for term in search_terms:
                if term.lower() in sentence.lower():
                    # Get surrounding context (previous and next sentence)
                    start_idx = max(0, i - 1)
                    end_idx = min(len(sentences), i + 2)
                    context = ' '.join(sentences[start_idx:end_idx])
                    mentions.append(context)
                    break
        
        return mentions


def create_document_dataframe(chunks: List[Dict]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from document chunks.
    Works on all platforms.
    """
    return pd.DataFrame(chunks)


# Pathway-specific functions (only used when Pathway is available)
if PATHWAY_AVAILABLE:
    def create_pathway_document_table(chunks: List[Dict]) -> 'pw.Table':
        """
        Create a Pathway table from document chunks.
        This enables real-time indexing and retrieval.
        """
        df = pd.DataFrame(chunks)
        table = pw.debug.table_from_pandas(df)
        return table

    def build_vector_index(
        table: 'pw.Table',
        embedder: 'SentenceTransformerEmbedder'
    ) -> 'pw.Table':
        """
        Build a vector index using Pathway's indexing capabilities.
        """
        table_with_embeddings = table.select(
            text=pw.this.text,
            book_name=pw.this.book_name,
            section_id=pw.this.section_id,
            chunk_id=pw.this.chunk_id,
            embedding=embedder(pw.this.text)
        )
        return table_with_embeddings
