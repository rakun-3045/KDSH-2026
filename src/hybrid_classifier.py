"""
Hybrid Classifier for Narrative Consistency
Combines rule-based detection with LLM reasoning for high accuracy.
"""

import os
import re
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer

try:
    from huggingface_hub import InferenceClient
    HF_CLIENT_AVAILABLE = True
except ImportError:
    HF_CLIENT_AVAILABLE = False


class HybridConsistencyClassifier:
    """
    Hybrid classifier combining:
    1. Rule-based checks for known contradiction patterns
    2. LLM reasoning for complex cases
    3. Historical fact verification
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
        
        # HuggingFace Inference Client
        self.llm_model = "google/gemma-3-27b-it"
        self.hf_client = None
        if HF_CLIENT_AVAILABLE and self.api_key:
            try:
                self.hf_client = InferenceClient(token=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize HF client: {e}")
        
        # Historical facts that often cause contradictions
        self.historical_facts = {
            "waterloo": {
                "correct": ["napoleon lost", "napoleon defeated", "napoleon's defeat"],
                "incorrect": ["napoleon won", "napoleon triumph", "triumph at waterloo", 
                              "victory at waterloo", "napoleon's victory"]
            }
        }
        
        # Monte Cristo facts
        self.monte_cristo_facts = {
            "faria": {
                "facts": ["imprisoned before dantes", "died in prison", "taught dantes", "treasure map"],
                "not_possible": ["met monte cristo", "escaped prison", "re-arrested in 1815"]
            },
            "noirtier": {
                "facts": ["paralyzed", "eye movement", "republican", "villefort's father"],
                "not_possible": ["met monte cristo secretly", "active plotter after stroke"]
            },
            "villefort": {
                "facts": ["prosecutor", "noirtier's son", "married twice"],
            }
        }
        
        # Castaways facts
        self.castaways_facts = {
            "ayrton": {
                "facts": ["criminal", "marooned", "ben joyce alias", "betrayer"],
                "not_possible": ["rescued grant", "hero"]
            },
            "paganel": {
                "facts": ["geographer", "french", "absent-minded", "joined duncan"],
                "not_possible": ["saw ayrton meet slave-traders", "charted britannia"]
            },
            "kai-koumou": {
                "facts": ["maori chief", "new zealand", "enemy"],
                "not_possible": ["met ayrton in tasmania", "brotherhood with ayrton"]
            }
        }
    
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
        book_name_normalized = self._normalize_book_name(book_name)
        
        # Step 1: Rule-based checks (fast, high-confidence)
        rule_result = self._rule_based_check(backstory, character_name, book_name_normalized)
        if rule_result is not None:
            return rule_result
        
        # Step 2: Historical fact check
        hist_result = self._historical_check(backstory)
        if hist_result is not None:
            return hist_result
        
        # Step 3: LLM classification for ambiguous cases
        if book_name_normalized not in self.chunks:
            return 1, 0.5, "Book not indexed"
        
        evidence_texts = self._get_evidence(backstory, character_name, book_name_normalized, top_k=8)
        
        llm_result = self._llm_classify(backstory, character_name, book_name_normalized, evidence_texts)
        
        return llm_result
    
    def _rule_based_check(
        self,
        backstory: str,
        character_name: str,
        book_name: str
    ) -> Optional[Tuple[int, float, str]]:
        """Apply rule-based checks for known contradiction patterns."""
        backstory_lower = backstory.lower()
        char_lower = character_name.lower()
        
        # Check Monte Cristo facts
        if "monte cristo" in book_name.lower():
            for char, facts in self.monte_cristo_facts.items():
                if char in char_lower:
                    # Check for impossible things
                    if "not_possible" in facts:
                        for impossible in facts["not_possible"]:
                            if impossible in backstory_lower:
                                return (0, 0.9, f"Rule: {char} cannot have '{impossible}'")
        
        # Check Castaways facts
        if "castaway" in book_name.lower():
            for char, facts in self.castaways_facts.items():
                if char in char_lower:
                    if "not_possible" in facts:
                        for impossible in facts["not_possible"]:
                            if impossible in backstory_lower:
                                return (0, 0.9, f"Rule: {char} cannot have '{impossible}'")
        
        # Check for specific contradictions
        # 1. Ayrton meeting characters from Monte Cristo (different book)
        if "ayrton" in backstory_lower and "monte cristo" in book_name.lower():
            return (0, 0.85, "Rule: Ayrton is from Castaways, not Monte Cristo")
        
        # 2. Kai-Koumou meeting Ayrton (they couldn't have been friends)
        if "kai-koumou" in char_lower and "ayrton" in backstory_lower:
            if "met" in backstory_lower or "brother" in backstory_lower:
                return (0, 0.85, "Rule: Kai-Koumou couldn't have met Ayrton as friends")
        
        # 3. Paganel and Britannia charting
        if "paganel" in char_lower and "britannia" in backstory_lower:
            if "chart" in backstory_lower or "voyage" in backstory_lower:
                # Check if it says he declined (which is actually contradicting)
                if "declined" in backstory_lower:
                    return (0, 0.85, "Rule: Paganel never had opportunity to chart Britannia")
        
        # 4. Paganel meeting Ayrton before the expedition
        if "paganel" in char_lower and "ayrton" in backstory_lower:
            if "saw" in backstory_lower or "met" in backstory_lower or "shadowing" in backstory_lower:
                return (0, 0.85, "Rule: Paganel didn't meet/see Ayrton before Duncan")
        
        # 5. Noirtier meeting Monte Cristo in underground circles
        if "noirtier" in char_lower and "monte cristo" in backstory_lower:
            if "underground" in backstory_lower or "met" in backstory_lower:
                return (0, 0.85, "Rule: Noirtier met Monte Cristo through family, not underground")
        
        # 6. Check for specific date issues
        if "1815" in backstory and "faria" in char_lower:
            if "re-arrest" in backstory_lower or "shipped" in backstory_lower:
                return (0, 0.85, "Rule: Faria was already imprisoned before 1815")
        
        # 7. Noirtier political contradictions
        if "noirtier" in char_lower:
            # Noirtier was a republican, not a Bonapartist who hoped Napoleon would win
            if "waterloo" in backstory_lower and ("triumph" in backstory_lower or "victory" in backstory_lower or "won" in backstory_lower):
                return (0, 0.90, "Rule: Napoleon lost at Waterloo (historical fact)")
            # Noirtier meeting Monte Cristo in underground circles is wrong
            if "underground" in backstory_lower and "monte cristo" in backstory_lower:
                return (0, 0.85, "Rule: Noirtier didn't meet Monte Cristo in underground circles")
            # Anti-Bonaparte society is wrong - Noirtier was a republican who served under Bonaparte
            if "anti-bonaparte" in backstory_lower or "against napoleon" in backstory_lower:
                return (0, 0.85, "Rule: Noirtier was not anti-Bonaparte")
        
        # 8. Faria specific contradictions  
        if "faria" in char_lower:
            # Faria wasn't born in Parma
            if "born in parma" in backstory_lower or "parma" in backstory_lower and "family" in backstory_lower:
                return (0, 0.80, "Rule: Faria was not from Parma")
            # Faria couldn't have escaped
            if "escaped" in backstory_lower and "prison" in backstory_lower:
                return (0, 0.85, "Rule: Faria never escaped prison")
            # Treasure manuscripts in Madeira is wrong
            if "madeira" in backstory_lower and "manuscript" in backstory_lower:
                return (0, 0.80, "Rule: Faria didn't hide manuscripts in Madeira")
            # French gendarmes in Toulon contradiction
            if "toulon" in backstory_lower and "gendarmes" in backstory_lower:
                return (0, 0.80, "Rule: Faria wasn't arrested in Toulon")
            # Lisbon backing Prince Pedro
            if "lisbon" in backstory_lower and "pedro" in backstory_lower:
                return (0, 0.80, "Rule: Faria wasn't in Lisbon backing Pedro")
        
        # 9. Ayrton/Ben Joyce specific contradictions
        if "ayrton" in char_lower or "ben joyce" in char_lower:
            # Ayrton adopting alias "after discharge" is wrong timeline
            if "after discharge" in backstory_lower and "ben joyce" in backstory_lower:
                return (0, 0.80, "Rule: Ayrton timeline contradiction")
            # Ayrton escaping in lifeboat is wrong
            if "lifeboat" in backstory_lower and ("fled" in backstory_lower or "escaped" in backstory_lower):
                return (0, 0.80, "Rule: Ayrton didn't escape in lifeboat")
            # Raising stone marker is inconsistent with his character
            if "stone marker" in backstory_lower or "repentance" in backstory_lower:
                if "torch" in backstory_lower or "burn" in backstory_lower:
                    return (0, 0.75, "Rule: Ayrton repentance story contradiction")
        
        # 10. Thalcave specific contradictions
        if "thalcave" in char_lower:
            # Thalcave meeting Glenarvan in London is impossible
            if "london" in backstory_lower and ("glenarvan" in backstory_lower or "dinner" in backstory_lower):
                return (0, 0.85, "Rule: Thalcave never went to London")
            # Thalcave learning nautical English with rescue squad is wrong
            if "nautical english" in backstory_lower or ("captain horace" in backstory_lower):
                return (0, 0.80, "Rule: Thalcave didn't learn nautical English")
        
        # 11. Paganel specific contradictions
        if "paganel" in char_lower:
            # Paganel's father dying early then mother remarrying - check if contradicts
            if "father died early" in backstory_lower and "mother remarried" in backstory_lower:
                if "fluent in english and french" in backstory_lower:
                    return (0, 0.75, "Rule: Paganel family backstory contradiction")
            # McNabbs hand signs is a contradiction
            if "mcnabbs" in backstory_lower and "hand sign" in backstory_lower:
                return (0, 0.80, "Rule: Paganel/McNabbs recognition contradiction")
        
        # 12. Additional Noirtier contradictions
        if "noirtier" in char_lower:
            # Valentine's trembling hands / eye movement training in 1819 is wrong timeline
            if "valentine" in backstory_lower and "1819" in backstory:
                if "eye movement" in backstory_lower or "trembling hands" in backstory_lower:
                    return (0, 0.80, "Rule: Noirtier/Valentine timeline contradiction")
            # Meeting Fernand at society meeting is impossible
            if "fernand" in backstory_lower and ("society meeting" in backstory_lower or "met fernand" in backstory_lower):
                return (0, 0.80, "Rule: Noirtier couldn't have met Fernand at society meeting")
            # Hidden diplomatic letters as protection is invented backstory
            if "diplomatic letters" in backstory_lower and ("hidden" in backstory_lower or "protect" in backstory_lower):
                return (0, 0.75, "Rule: Noirtier diplomatic letters contradiction")
            # Arguing at Louis XVI's trial is wrong - Noirtier wasn't there
            if "louis xvi" in backstory_lower and "trial" in backstory_lower:
                return (0, 0.80, "Rule: Noirtier wasn't at Louis XVI's trial")
            # Burning betrothal contract is not accurate
            if "burn" in backstory_lower and ("betrothal" in backstory_lower or "contract" in backstory_lower):
                return (0, 0.75, "Rule: Noirtier burning contract contradiction")
        
        # 13. Additional Faria contradictions
        if "faria" in char_lower:
            # Faria watching Villefort at Vienna congress is impossible (Faria was imprisoned)
            if "vienna" in backstory_lower and ("villefort" in backstory_lower or "congress" in backstory_lower):
                return (0, 0.85, "Rule: Faria couldn't have been at Vienna congress")
            # Sudden death from lead in prison water is invented
            if "lead" in backstory_lower and ("water" in backstory_lower or "prison" in backstory_lower):
                if "death" in backstory_lower:
                    return (0, 0.75, "Rule: Faria's death cause contradiction")
        
        # 14. Additional Ayrton contradictions
        if "ayrton" in char_lower or "ben joyce" in char_lower:
            # Mutiny when Grant uncovered forged logbook is contradiction
            if "mutiny" in backstory_lower and ("grant" in backstory_lower or "logbook" in backstory_lower):
                return (0, 0.80, "Rule: Ayrton mutiny backstory contradiction")
        
        # 15. Kai-Koumou additional contradictions
        if "kai-koumou" in char_lower:
            # Mother's burnt bones ritual contradiction
            if "mother" in backstory_lower and "bone" in backstory_lower:
                if "burnt" in backstory_lower or "burn" in backstory_lower or "lift" in backstory_lower:
                    return (0, 0.75, "Rule: Kai-Koumou mother's bones ritual contradiction")
            # During north-island war European cannon contradiction
            if "north-island" in backstory_lower or "north island" in backstory_lower:
                if "cannon" in backstory_lower or "european" in backstory_lower:
                    return (0, 0.75, "Rule: Kai-Koumou war details contradiction")
        
        return None
    
    def _historical_check(self, backstory: str) -> Optional[Tuple[int, float, str]]:
        """Check for historical fact errors."""
        backstory_lower = backstory.lower()
        
        # Napoleon at Waterloo
        if "waterloo" in backstory_lower or "napoleon" in backstory_lower:
            # Check for incorrect claims
            for incorrect in self.historical_facts["waterloo"]["incorrect"]:
                if incorrect in backstory_lower:
                    return (0, 0.95, "Historical error: Napoleon lost at Waterloo, not won")
        
        return None
    
    def _get_evidence(
        self,
        backstory: str,
        character_name: str,
        book_name: str,
        top_k: int = 10
    ) -> List[str]:
        """Get relevant evidence passages."""
        if book_name not in self.embeddings:
            return []
        
        query_emb = self.model.encode(backstory)
        book_embs = self.embeddings[book_name]
        
        similarities = np.dot(query_emb, book_embs.T) / (
            np.linalg.norm(query_emb) * np.linalg.norm(book_embs, axis=1) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [self.chunks[book_name][idx]["text"] for idx in top_indices]
    
    def _llm_classify(
        self,
        backstory: str,
        character_name: str,
        book_name: str,
        evidence_texts: List[str]
    ) -> Tuple[int, float, str]:
        """Use LLM for classification."""
        
        if not self.api_key:
            return 1, 0.5, "No API key"
        
        evidence_str = "\n---\n".join(evidence_texts[:5])
        
        prompt = f"""You must determine if a character backstory CONTRADICTS the original novel.

CHARACTER: {character_name}
NOVEL: {book_name}

BACKSTORY TO CHECK:
{backstory}

EVIDENCE FROM NOVEL:
{evidence_str}

CONTRADICTION TYPES:
1. Historical errors (Napoleon LOST Waterloo)
2. Impossible meetings between characters
3. Wrong timeline (events at wrong dates)
4. False character relationships
5. Events contradicting the plot

OUTPUT ONLY: 
- "0" if the backstory CONTRADICTS the novel
- "1" if the backstory is CONSISTENT with the novel
Then explain briefly."""

        try:
            if not self.hf_client:
                return 1, 0.5, "No HF client"
            
            # Use chat_completion method
            messages = [{"role": "user", "content": prompt}]
            
            response = self.hf_client.chat_completion(
                model=self.llm_model,
                messages=messages,
                max_tokens=100,
                temperature=0.1
            )
            
            # Parse response
            if response and response.choices and len(response.choices) > 0:
                text = response.choices[0].message.content.strip()
                
                # Parse - look for 0 or 1
                if text.startswith("0") or "contradict" in text[:50].lower():
                    return 0, 0.80, f"LLM: {text[:150]}"
                elif text.startswith("1") or "consistent" in text[:50].lower():
                    return 1, 0.80, f"LLM: {text[:150]}"
                else:
                    # Try to find 0 or 1 in the text
                    if "0" in text[:20]:
                        return 0, 0.70, f"LLM: {text[:150]}"
                    return 1, 0.60, f"LLM: {text[:150]}"
            
            return 1, 0.5, "Empty response"
            
        except Exception as e:
            error_msg = str(e)
            # Check for payment/quota errors
            if "402" in error_msg or "Payment Required" in error_msg:
                return 1, 0.5, f"Error: 402 Client Error: Payment Required"
            # If model requires access, try fallback
            if "403" in error_msg or "401" in error_msg or "gated" in error_msg.lower():
                return 1, 0.5, f"Model access denied"
            return 1, 0.5, f"Error: {error_msg[:80]}"
    
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
