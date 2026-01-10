"""
Enhanced Evidence Retrieval Module
Implements multiple retrieval strategies for better evidence coverage.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    """Retrieved evidence chunk with metadata."""
    text: str
    book_name: str
    section_id: str
    score: float
    retrieval_type: str  # "semantic", "keyword", "character"


class EnhancedEvidenceRetriever:
    """
    Multi-strategy evidence retriever.
    Combines semantic search, keyword matching, and character-focused retrieval.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 20
    ):
        self.model = SentenceTransformer(embedding_model)
        self.top_k = top_k
        self.chunks: Dict[str, List[Dict]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
    
    def index_book(self, book_name: str, chunks: List[Dict]) -> None:
        """Index a book's chunks."""
        self.chunks[book_name] = chunks
        texts = [c["text"] for c in chunks]
        self.embeddings[book_name] = self.model.encode(texts, show_progress_bar=True)
        print(f"Indexed {len(chunks)} chunks for '{book_name}'")
    
    def retrieve_comprehensive(
        self,
        backstory: str,
        character_name: str,
        book_name: str
    ) -> List[RetrievedChunk]:
        """
        Comprehensive retrieval using multiple strategies.
        """
        if book_name not in self.chunks:
            return []
        
        all_chunks = []
        seen_indices: Set[int] = set()
        
        # Strategy 1: Semantic search on full backstory
        semantic_chunks = self._semantic_search(
            backstory, book_name, top_k=self.top_k // 3
        )
        for chunk, score, idx in semantic_chunks:
            if idx not in seen_indices:
                all_chunks.append(RetrievedChunk(
                    text=chunk["text"],
                    book_name=book_name,
                    section_id=chunk.get("section_id", ""),
                    score=score,
                    retrieval_type="semantic"
                ))
                seen_indices.add(idx)
        
        # Strategy 2: Character-focused retrieval
        character_chunks = self._character_search(
            character_name, book_name, top_k=self.top_k // 3
        )
        for chunk, score, idx in character_chunks:
            if idx not in seen_indices:
                all_chunks.append(RetrievedChunk(
                    text=chunk["text"],
                    book_name=book_name,
                    section_id=chunk.get("section_id", ""),
                    score=score,
                    retrieval_type="character"
                ))
                seen_indices.add(idx)
        
        # Strategy 3: Keyword-based retrieval for key claims
        keywords = self._extract_key_terms(backstory, character_name)
        for keyword in keywords[:5]:
            keyword_chunks = self._keyword_search(
                keyword, book_name, top_k=3
            )
            for chunk, score, idx in keyword_chunks:
                if idx not in seen_indices:
                    all_chunks.append(RetrievedChunk(
                        text=chunk["text"],
                        book_name=book_name,
                        section_id=chunk.get("section_id", ""),
                        score=score,
                        retrieval_type="keyword"
                    ))
                    seen_indices.add(idx)
        
        # Strategy 4: Retrieve passages about family if mentioned
        if self._mentions_family(backstory):
            family_chunks = self._family_search(
                character_name, book_name, top_k=self.top_k // 4
            )
            for chunk, score, idx in family_chunks:
                if idx not in seen_indices:
                    all_chunks.append(RetrievedChunk(
                        text=chunk["text"],
                        book_name=book_name,
                        section_id=chunk.get("section_id", ""),
                        score=score,
                        retrieval_type="family"
                    ))
                    seen_indices.add(idx)
        
        # Sort by score and return
        all_chunks.sort(key=lambda x: x.score, reverse=True)
        return all_chunks[:self.top_k]
    
    def _semantic_search(
        self,
        query: str,
        book_name: str,
        top_k: int
    ) -> List[Tuple[Dict, float, int]]:
        """Semantic similarity search."""
        query_emb = self.model.encode(query)
        book_embs = self.embeddings[book_name]
        
        similarities = self._cosine_similarity(
            query_emb.reshape(1, -1), book_embs
        )[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                self.chunks[book_name][idx],
                float(similarities[idx]),
                int(idx)
            ))
        return results
    
    def _character_search(
        self,
        character_name: str,
        book_name: str,
        top_k: int
    ) -> List[Tuple[Dict, float, int]]:
        """Find passages mentioning the character."""
        results = []
        name_parts = character_name.lower().split()
        
        for idx, chunk in enumerate(self.chunks[book_name]):
            text_lower = chunk["text"].lower()
            
            # Check for character name mentions
            mention_count = 0
            for part in name_parts:
                if len(part) > 2:
                    mention_count += text_lower.count(part)
            
            if mention_count > 0:
                # Score based on mention density
                score = min(1.0, mention_count / 10.0)
                results.append((chunk, score, idx))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _keyword_search(
        self,
        keyword: str,
        book_name: str,
        top_k: int
    ) -> List[Tuple[Dict, float, int]]:
        """Keyword-based search."""
        results = []
        keyword_lower = keyword.lower()
        
        for idx, chunk in enumerate(self.chunks[book_name]):
            text_lower = chunk["text"].lower()
            
            if keyword_lower in text_lower:
                # Score based on position and frequency
                count = text_lower.count(keyword_lower)
                score = min(1.0, count / 5.0)
                results.append((chunk, score, idx))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _family_search(
        self,
        character_name: str,
        book_name: str,
        top_k: int
    ) -> List[Tuple[Dict, float, int]]:
        """Search for family-related passages."""
        family_terms = [
            "father", "mother", "brother", "sister", "son", "daughter",
            "wife", "husband", "child", "parent", "family", "born",
            "died", "death", "killed", "orphan"
        ]
        
        results = []
        name_parts = [p.lower() for p in character_name.split() if len(p) > 2]
        
        for idx, chunk in enumerate(self.chunks[book_name]):
            text_lower = chunk["text"].lower()
            
            # Check if mentions character AND family terms
            has_character = any(part in text_lower for part in name_parts)
            family_count = sum(1 for term in family_terms if term in text_lower)
            
            if has_character and family_count > 0:
                score = min(1.0, family_count / 5.0)
                results.append((chunk, score, idx))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _extract_key_terms(self, backstory: str, character_name: str) -> List[str]:
        """Extract key terms from backstory for targeted search."""
        terms = []
        
        # Extract proper nouns (potential names, places)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', backstory)
        terms.extend([n for n in proper_nouns if n != character_name][:3])
        
        # Extract years
        years = re.findall(r'\b1[789]\d{2}\b', backstory)
        terms.extend(years[:2])
        
        # Extract key action verbs context
        action_phrases = re.findall(
            r'(?:killed|murdered|died|born|married|discovered|escaped|joined)\s+\w+(?:\s+\w+)?',
            backstory.lower()
        )
        terms.extend(action_phrases[:3])
        
        return terms
    
    def _mentions_family(self, text: str) -> bool:
        """Check if text mentions family relationships."""
        family_keywords = [
            "father", "mother", "brother", "sister", "parent",
            "son", "daughter", "wife", "husband", "family",
            "orphan", "born", "birth"
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in family_keywords)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(a_norm, b_norm.T)
