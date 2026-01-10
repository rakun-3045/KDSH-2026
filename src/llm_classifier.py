"""
LLM-Based Classifier for Narrative Consistency
Uses HuggingFace API with carefully crafted prompts to detect contradictions.
"""

import os
import re
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer


class LLMConsistencyClassifier:
    """
    LLM-based classifier that uses the HuggingFace API
    with carefully crafted prompts to detect narrative contradictions.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY", "")
        self.model = SentenceTransformer(embedding_model)
        self.chunks: Dict[str, List[Dict]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # API endpoint
        self.api_url = "https://router.huggingface.co/hf-inference/models/google/gemma-2-27b-it"
    
    def index_book(self, book_name: str, chunks: List[Dict]) -> None:
        """Index a book for retrieval."""
        self.chunks[book_name] = chunks
        texts = [c["text"] for c in chunks]
        self.embeddings[book_name] = self.model.encode(texts, show_progress_bar=True)
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
        evidence_texts = self._get_evidence(backstory, character_name, book_name, top_k=8)
        
        # Use LLM for classification
        prediction, confidence, rationale = self._llm_classify(
            backstory, character_name, book_name, evidence_texts
        )
        
        return prediction, confidence, rationale
    
    def _get_evidence(
        self,
        backstory: str,
        character_name: str,
        book_name: str,
        top_k: int = 10
    ) -> List[str]:
        """Get relevant evidence passages using semantic search."""
        # Semantic search
        query_emb = self.model.encode(backstory)
        book_embs = self.embeddings[book_name]
        
        similarities = np.dot(query_emb, book_embs.T) / (
            np.linalg.norm(query_emb) * np.linalg.norm(book_embs, axis=1) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        evidence = []
        for idx in top_indices:
            evidence.append(self.chunks[book_name][idx]["text"])
        
        return evidence
    
    def _llm_classify(
        self,
        backstory: str,
        character_name: str,
        book_name: str,
        evidence_texts: List[str]
    ) -> Tuple[int, float, str]:
        """Use LLM for classification."""
        
        evidence_str = "\n---\n".join(evidence_texts[:6])
        
        prompt = f"""Analyze if this hypothetical backstory for a character CONTRADICTS the original novel.

CHARACTER: {character_name}
NOVEL: {book_name}

HYPOTHETICAL BACKSTORY:
{backstory}

RELEVANT PASSAGES FROM THE NOVEL:
{evidence_str}

CONTRADICTION TYPES TO CHECK:
1. FACTUAL ERRORS: Historical facts wrong (e.g., "Napoleon won at Waterloo" - he lost)
2. IMPOSSIBLE MEETINGS: Character meets someone they couldn't have met in the novel
3. TIMELINE ERRORS: Events at wrong times or in wrong order
4. CHARACTER FACTS: Wrong info about character's family, origin, profession
5. PLOT CONTRADICTIONS: Events that contradict the actual story

IMPORTANT NOTES:
- Napoleon LOST at Waterloo in 1815 (any backstory saying he won is CONTRADICTING)
- Château d'If is where Dantès was imprisoned (Faria was already there when Dantès arrived)
- Noirtier was paralyzed and communicated by eye movements
- Ayrton was a criminal who was marooned, not a hero
- Characters from different novels meeting is usually a contradiction

Based on the above, is this backstory:
- CONSISTENT (1): No clear contradictions with the novel
- CONTRADICTING (0): Contains factual errors, impossible events, or contradicts the plot

Respond with ONLY: 0 or 1
Then a brief reason."""

        if not self.api_key:
            return 1, 0.5, "No API key"
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.1,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get("generated_text", "")
                    
                    # Parse response
                    text = text.strip()
                    
                    # Look for 0 or 1 at the start
                    if text.startswith("0"):
                        return 0, 0.85, text[:200]
                    elif text.startswith("1"):
                        return 1, 0.85, text[:200]
                    elif "contradict" in text.lower() or "0" in text[:10]:
                        return 0, 0.75, text[:200]
                    else:
                        return 1, 0.65, text[:200]
            
            return 1, 0.5, f"API error: {response.status_code}"
            
        except Exception as e:
            return 1, 0.5, f"Error: {str(e)}"
    
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
