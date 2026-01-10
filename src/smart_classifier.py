"""
Smart Classifier for Narrative Consistency
Uses semantic similarity and targeted pattern matching
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from collections import defaultdict


class SmartConsistencyClassifier:
    """
    Smart classifier that uses multiple signals:
    1. Semantic similarity between backstory and evidence
    2. Character mention analysis
    3. Targeted pattern detection
    4. Cross-reference validation
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.chunks: Dict[str, List[Dict]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.character_chunks: Dict[str, Dict[str, List[int]]] = {}
        
        # Known character relationships from novels
        self.monte_cristo_characters = {
            "faria", "dantes", "edmond", "villefort", "noirtier", "mercedes",
            "fernand", "danglars", "caderousse", "valentine", "maximilian",
            "monte cristo", "count"
        }
        
        self.castaways_characters = {
            "paganel", "jacques", "glenarvan", "ayrton", "grant", "mary",
            "robert", "thalcave", "kai-koumou", "mulrady", "mcnabbs"
        }
    
    def index_book(self, book_name: str, chunks: List[Dict]) -> None:
        """Index a book for retrieval."""
        self.chunks[book_name] = chunks
        texts = [c["text"] for c in chunks]
        self.embeddings[book_name] = self.model.encode(texts, show_progress_bar=True)
        
        # Index character mentions
        self.character_chunks[book_name] = defaultdict(list)
        for idx, chunk in enumerate(chunks):
            text_lower = chunk["text"].lower()
            for char in self.monte_cristo_characters | self.castaways_characters:
                if char in text_lower:
                    self.character_chunks[book_name][char].append(idx)
        
        print(f"Indexed {len(chunks)} chunks for '{book_name}'")
    
    def classify(
        self,
        backstory: str,
        character_name: str,
        book_name: str
    ) -> Tuple[int, float, str]:
        """
        Classify backstory as consistent (1) or contradicting (0).
        """
        book_name = self._normalize_book_name(book_name)
        
        if book_name not in self.chunks:
            return 1, 0.5, "Book not indexed"
        
        # Get relevant evidence
        evidence_texts = self._get_evidence(backstory, character_name, book_name)
        
        # Calculate multiple signals
        signals = {}
        
        # Signal 1: Semantic coherence
        signals['semantic'] = self._compute_semantic_score(backstory, evidence_texts)
        
        # Signal 2: Character presence in evidence
        signals['char_presence'] = self._compute_character_presence(
            character_name, evidence_texts, book_name
        )
        
        # Signal 3: Cross-book contamination check
        signals['contamination'] = self._check_cross_contamination(backstory, book_name)
        
        # Signal 4: Event consistency
        signals['event_consistency'] = self._check_event_consistency(
            backstory, evidence_texts, character_name
        )
        
        # Signal 5: Timeline plausibility
        signals['timeline'] = self._check_timeline(backstory, evidence_texts)
        
        # Combine signals
        prediction, confidence, rationale = self._combine_signals(signals, backstory)
        
        return prediction, confidence, rationale
    
    def _get_evidence(
        self,
        backstory: str,
        character_name: str,
        book_name: str,
        top_k: int = 20
    ) -> List[str]:
        """Get relevant evidence passages."""
        # Semantic search
        query_emb = self.model.encode(backstory)
        book_embs = self.embeddings[book_name]
        
        similarities = np.dot(query_emb, book_embs.T) / (
            np.linalg.norm(query_emb) * np.linalg.norm(book_embs, axis=1) + 1e-8
        )
        
        # Get top semantic matches
        top_indices = set(np.argsort(similarities)[::-1][:top_k])
        
        # Add character-specific chunks
        char_parts = character_name.lower().split()
        for part in char_parts:
            if part in self.character_chunks[book_name]:
                for idx in self.character_chunks[book_name][part][:10]:
                    top_indices.add(idx)
        
        # Collect evidence
        evidence = []
        for idx in top_indices:
            if idx < len(self.chunks[book_name]):
                evidence.append(self.chunks[book_name][idx]["text"])
        
        return evidence[:top_k]
    
    def _compute_semantic_score(
        self,
        backstory: str,
        evidence_texts: List[str]
    ) -> float:
        """Compute semantic similarity score."""
        if not evidence_texts:
            return 0.5
        
        # Split backstory into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', backstory) if len(s.strip()) > 20]
        
        if not sentences:
            return 0.5
        
        # Get embeddings
        backstory_embs = self.model.encode(sentences)
        evidence_embs = self.model.encode(evidence_texts[:10])
        
        # Compute max similarity for each backstory sentence
        scores = []
        for bs_emb in backstory_embs:
            sims = np.dot(bs_emb, evidence_embs.T) / (
                np.linalg.norm(bs_emb) * np.linalg.norm(evidence_embs, axis=1) + 1e-8
            )
            scores.append(np.max(sims))
        
        return float(np.mean(scores))
    
    def _compute_character_presence(
        self,
        character_name: str,
        evidence_texts: List[str],
        book_name: str
    ) -> float:
        """Check if character is well-represented in evidence."""
        combined = " ".join(evidence_texts).lower()
        
        name_parts = character_name.lower().split()
        mentions = sum(combined.count(part) for part in name_parts if len(part) > 2)
        
        # Normalize
        return min(1.0, mentions / 20)
    
    def _check_cross_contamination(
        self,
        backstory: str,
        book_name: str
    ) -> float:
        """Check if backstory mentions characters from wrong book."""
        backstory_lower = backstory.lower()
        
        if "monte cristo" in book_name.lower():
            # Check for Castaways characters in Monte Cristo backstory
            wrong_chars = ["paganel", "glenarvan", "ayrton", "thalcave", "kai-koumou", "britannia"]
        else:
            # Check for Monte Cristo characters in Castaways backstory
            wrong_chars = ["villefort", "dantes", "chateau d'if", "monte cristo", "mercedes", "danglars"]
        
        contamination_score = sum(1 for char in wrong_chars if char in backstory_lower)
        
        return min(1.0, contamination_score * 0.5)
    
    def _check_event_consistency(
        self,
        backstory: str,
        evidence_texts: List[str],
        character_name: str
    ) -> float:
        """Check for event-level contradictions."""
        backstory_lower = backstory.lower()
        evidence_combined = " ".join(evidence_texts).lower()
        
        contradiction_score = 0.0
        
        # Check death claims
        death_patterns = [
            (r"(?:his|her)\s+(\w+)\s+(?:was\s+)?(?:killed|died|murdered)", "claims death"),
            (r"(\w+)\s+died\s+(?:when|while|before)", "claims death"),
        ]
        
        for pattern, desc in death_patterns:
            matches = re.findall(pattern, backstory_lower)
            for match in matches:
                # If someone died in backstory, check if they're alive in evidence
                person = match if isinstance(match, str) else match[0]
                if len(person) > 2:
                    # Check for alive indicators
                    alive_patterns = [
                        f"{person}\\s+(?:said|spoke|told|asked)",
                        f"{person}\\s+(?:looked|smiled|nodded)",
                        f"{person}\\s+(?:came|went|walked|arrived)",
                    ]
                    for alive_pattern in alive_patterns:
                        if re.search(alive_pattern, evidence_combined):
                            contradiction_score += 0.3
                            break
        
        # Check for meeting claims between characters who shouldn't meet
        meeting_patterns = [
            r"(?:he|she)\s+met\s+(\w+)",
            r"together\s+with\s+(\w+)",
            r"alongside\s+(\w+)",
        ]
        
        for pattern in meeting_patterns:
            matches = re.findall(pattern, backstory_lower)
            for match in matches:
                if match in self.monte_cristo_characters and "castaways" in character_name.lower():
                    contradiction_score += 0.4
                elif match in self.castaways_characters and "monte" in character_name.lower():
                    contradiction_score += 0.4
        
        return min(1.0, contradiction_score)
    
    def _check_timeline(
        self,
        backstory: str,
        evidence_texts: List[str]
    ) -> float:
        """Check for timeline issues."""
        # Extract years from backstory
        backstory_years = set(re.findall(r'\b1[789]\d{2}\b', backstory))
        evidence_years = set()
        for text in evidence_texts:
            evidence_years.update(re.findall(r'\b1[789]\d{2}\b', text))
        
        # Check for obvious timeline conflicts
        if backstory_years and evidence_years:
            backstory_max = max(int(y) for y in backstory_years)
            backstory_min = min(int(y) for y in backstory_years)
            evidence_max = max(int(y) for y in evidence_years) if evidence_years else 1900
            evidence_min = min(int(y) for y in evidence_years) if evidence_years else 1800
            
            # Check if backstory years are way outside evidence range
            if backstory_max > evidence_max + 20 or backstory_min < evidence_min - 50:
                return 0.3
        
        return 0.0
    
    def _combine_signals(
        self,
        signals: Dict[str, float],
        backstory: str
    ) -> Tuple[int, float, str]:
        """Combine signals into final prediction."""
        
        # Strong contradiction indicators
        if signals['contamination'] > 0.3:
            return 0, 0.85, "Cross-book character contamination detected"
        
        if signals['event_consistency'] > 0.5:
            return 0, 0.80, "Event-level contradiction detected"
        
        # Moderate contradiction indicators
        contradiction_score = (
            signals['contamination'] * 0.4 +
            signals['event_consistency'] * 0.3 +
            (1 - signals['semantic']) * 0.2 +
            signals['timeline'] * 0.1
        )
        
        # Very low semantic similarity is suspicious
        if signals['semantic'] < 0.25 and signals['char_presence'] < 0.3:
            contradiction_score += 0.2
        
        # Make decision
        if contradiction_score > 0.35:
            confidence = min(0.9, 0.5 + contradiction_score)
            return 0, confidence, f"Contradiction signals: {contradiction_score:.2f}"
        else:
            confidence = min(0.9, 0.5 + (1 - contradiction_score) * 0.5)
            return 1, confidence, f"Consistent - semantic score: {signals['semantic']:.2f}"
    
    def _normalize_book_name(self, book_name: str) -> str:
        """Normalize book name."""
        book_name_lower = book_name.lower()
        
        for indexed_name in self.chunks.keys():
            if indexed_name.lower() in book_name_lower or book_name_lower in indexed_name.lower():
                return indexed_name
        
        if "castaway" in book_name_lower or "search" in book_name_lower:
            return "In Search of the Castaways"
        if "monte cristo" in book_name_lower or "count" in book_name_lower:
            return "The Count of Monte Cristo"
        
        return book_name
