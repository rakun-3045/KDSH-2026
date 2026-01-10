"""
Evidence Retrieval Module using Pathway Vector Store
Implements multi-stage retrieval for narrative consistency checking
"""

import pathway as pw
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class RetrievedEvidence:
    """Container for retrieved evidence with metadata."""
    text: str
    book_name: str
    section_id: str
    chunk_id: int
    similarity_score: float
    relevance_type: str  # "character_mention", "event", "setting", etc.


class NarrativeEvidenceRetriever:
    """
    Multi-stage evidence retriever for narrative consistency checking.
    
    Uses Pathway for document management and semantic search
    with additional heuristic-based filtering for improved precision.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 15,
        similarity_threshold: float = 0.3
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.document_chunks: Dict[str, List[Dict]] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
    def index_documents(self, book_name: str, chunks: List[Dict]) -> None:
        """
        Index document chunks for a book.
        Computes and caches embeddings for efficient retrieval.
        """
        self.document_chunks[book_name] = chunks
        
        # Compute embeddings for all chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.embeddings_cache[book_name] = embeddings
        print(f"Indexed {len(chunks)} chunks for '{book_name}'")
    
    def retrieve_evidence(
        self,
        query: str,
        book_name: str,
        character_name: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[RetrievedEvidence]:
        """
        Retrieve relevant evidence chunks for a query.
        
        Uses a multi-stage approach:
        1. Semantic similarity search
        2. Character mention filtering (if character specified)
        3. Relevance re-ranking
        """
        if book_name not in self.document_chunks:
            raise ValueError(f"Book '{book_name}' not indexed")
        
        top_k = top_k or self.top_k
        
        # Stage 1: Semantic similarity search
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        book_embeddings = self.embeddings_cache[book_name]
        
        # Compute cosine similarities
        similarities = self._cosine_similarity(
            query_embedding.reshape(1, -1), 
            book_embeddings
        )[0]
        
        # Get top candidates (more than top_k for filtering)
        candidate_indices = np.argsort(similarities)[::-1][:top_k * 3]
        
        # Stage 2: Filter and score candidates
        evidence_list = []
        chunks = self.document_chunks[book_name]
        
        for idx in candidate_indices:
            if similarities[idx] < self.similarity_threshold:
                continue
            
            chunk = chunks[idx]
            relevance_type = self._classify_relevance(
                chunk["text"], 
                query, 
                character_name
            )
            
            # Boost score for character mentions
            score = similarities[idx]
            if character_name and self._contains_character(
                chunk["text"], 
                character_name
            ):
                score *= 1.2
            
            evidence_list.append(RetrievedEvidence(
                text=chunk["text"],
                book_name=chunk["book_name"],
                section_id=chunk["section_id"],
                chunk_id=chunk["chunk_id"],
                similarity_score=float(score),
                relevance_type=relevance_type
            ))
        
        # Stage 3: Re-rank and deduplicate
        evidence_list = self._rerank_evidence(evidence_list)
        
        return evidence_list[:top_k]
    
    def retrieve_multi_query(
        self,
        queries: List[str],
        book_name: str,
        character_name: Optional[str] = None,
        top_k_per_query: int = 5
    ) -> List[RetrievedEvidence]:
        """
        Retrieve evidence using multiple queries for better coverage.
        Useful for complex backstories with multiple claims.
        """
        all_evidence = []
        seen_chunks = set()
        
        for query in queries:
            evidence = self.retrieve_evidence(
                query, 
                book_name, 
                character_name,
                top_k=top_k_per_query
            )
            
            for ev in evidence:
                chunk_key = (ev.book_name, ev.section_id, ev.chunk_id)
                if chunk_key not in seen_chunks:
                    all_evidence.append(ev)
                    seen_chunks.add(chunk_key)
        
        # Final re-ranking
        all_evidence.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return all_evidence
    
    def _cosine_similarity(
        self, 
        query: np.ndarray, 
        documents: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and documents."""
        query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        doc_norm = documents / (np.linalg.norm(documents, axis=1, keepdims=True) + 1e-8)
        return np.dot(query_norm, doc_norm.T)
    
    def _classify_relevance(
        self,
        text: str,
        query: str,
        character_name: Optional[str]
    ) -> str:
        """Classify the type of relevance for a chunk."""
        text_lower = text.lower()
        
        # Check for character mentions
        if character_name and self._contains_character(text, character_name):
            return "character_mention"
        
        # Check for event-related keywords
        event_keywords = [
            "happened", "occurred", "event", "incident", 
            "killed", "died", "born", "married", "discovered"
        ]
        if any(kw in text_lower for kw in event_keywords):
            return "event"
        
        # Check for setting/location
        setting_keywords = [
            "place", "city", "country", "house", "room", 
            "location", "arrived", "traveled"
        ]
        if any(kw in text_lower for kw in setting_keywords):
            return "setting"
        
        # Check for relationship mentions
        relationship_keywords = [
            "father", "mother", "brother", "sister", "friend",
            "enemy", "lover", "wife", "husband", "son", "daughter"
        ]
        if any(kw in text_lower for kw in relationship_keywords):
            return "relationship"
        
        return "general"
    
    def _contains_character(self, text: str, character_name: str) -> bool:
        """Check if text contains character name (case-insensitive)."""
        name_parts = character_name.lower().split()
        text_lower = text.lower()
        
        # Check full name
        if character_name.lower() in text_lower:
            return True
        
        # Check individual name parts (for names like "Jacques Paganel")
        for part in name_parts:
            if len(part) > 2 and part in text_lower:
                return True
        
        return False
    
    def _rerank_evidence(
        self, 
        evidence_list: List[RetrievedEvidence]
    ) -> List[RetrievedEvidence]:
        """
        Re-rank evidence based on multiple factors.
        Prioritizes diverse evidence types and sections.
        """
        if not evidence_list:
            return []
        
        # Group by section to ensure diversity
        sections_seen = {}
        reranked = []
        
        # First pass: pick best from each section
        for ev in sorted(evidence_list, key=lambda x: x.similarity_score, reverse=True):
            if ev.section_id not in sections_seen:
                sections_seen[ev.section_id] = 0
            
            if sections_seen[ev.section_id] < 2:  # Max 2 per section
                reranked.append(ev)
                sections_seen[ev.section_id] += 1
        
        return reranked


def extract_backstory_claims(backstory_content: str) -> List[str]:
    """
    Extract individual claims from a backstory for multi-query retrieval.
    Each claim becomes a separate retrieval query.
    """
    claims = []
    
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', backstory_content)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:  # Skip very short fragments
            claims.append(sentence)
    
    # Also extract key phrases with entities
    # Look for patterns like "his father", "her mother", specific events
    entity_patterns = [
        r"his (?:father|mother|brother|sister|wife|family)",
        r"her (?:father|mother|brother|sister|husband|family)",
        r"born (?:in|to|on)",
        r"died (?:in|when|after)",
        r"killed (?:by|in|when)",
        r"at (?:age|the age of) \d+",
    ]
    
    for pattern in entity_patterns:
        matches = re.findall(f"[^.]*{pattern}[^.]*[.]", backstory_content, re.IGNORECASE)
        claims.extend(matches)
    
    # Deduplicate while preserving order
    seen = set()
    unique_claims = []
    for claim in claims:
        if claim not in seen:
            seen.add(claim)
            unique_claims.append(claim)
    
    return unique_claims
