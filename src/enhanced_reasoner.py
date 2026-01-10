"""
Enhanced Consistency Reasoning Module
Implements multiple advanced techniques to improve accuracy:
1. Multi-strategy retrieval
2. Pattern-based contradiction detection
3. Semantic similarity scoring
4. Ensemble voting
5. Character-focused analysis
"""

import re
import os
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
import requests


@dataclass
class EnhancedResult:
    """Enhanced result with multiple signals."""
    prediction: int
    confidence: float
    rationale: str
    contradiction_signals: List[str]
    support_signals: List[str]
    semantic_score: float
    pattern_score: float
    llm_score: float


class ContradictionPatternDetector:
    """
    Rule-based contradiction detection using linguistic patterns.
    Highly effective for common contradiction types.
    """
    
    def __init__(self):
        # Death/Life contradictions
        self.death_patterns = [
            (r"(?:his|her)\s+(\w+)\s+(?:was\s+)?killed", "death_claim"),
            (r"(\w+)\s+died\s+(?:when|while|before|after)", "death_claim"),
            (r"(\w+)'s?\s+death", "death_reference"),
            (r"murdered\s+(?:his|her)\s+(\w+)", "murder_claim"),
        ]
        
        # Family relationship patterns
        self.family_patterns = [
            (r"(?:his|her)\s+(father|mother|brother|sister|son|daughter|wife|husband)", "family_relation"),
            (r"(orphan|orphaned)", "orphan_status"),
            (r"only\s+child", "only_child"),
            (r"no\s+(?:brothers?|sisters?|siblings?)", "no_siblings"),
        ]
        
        # Birth/Origin patterns
        self.origin_patterns = [
            (r"born\s+(?:in|at|on)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", "birthplace"),
            (r"native\s+of\s+([A-Z][a-z]+)", "native_place"),
            (r"(?:from|came\s+from)\s+([A-Z][a-z]+)", "origin_place"),
        ]
        
        # Event/Time patterns
        self.time_patterns = [
            (r"at\s+(?:age|the\s+age\s+of)\s+(\d+)", "age_reference"),
            (r"(\d{4})", "year_reference"),
            (r"(before|after|during)\s+the\s+(\w+)", "temporal_marker"),
        ]
        
    def extract_claims(self, text: str) -> Dict[str, List[str]]:
        """Extract structured claims from text."""
        claims = {
            "deaths": [],
            "family": [],
            "origins": [],
            "times": [],
            "entities": []
        }
        
        text_lower = text.lower()
        
        for pattern, claim_type in self.death_patterns:
            matches = re.findall(pattern, text_lower)
            claims["deaths"].extend(matches)
        
        for pattern, claim_type in self.family_patterns:
            matches = re.findall(pattern, text_lower)
            claims["family"].extend(matches)
        
        for pattern, claim_type in self.origin_patterns:
            matches = re.findall(pattern, text)
            claims["origins"].extend(matches)
        
        for pattern, claim_type in self.time_patterns:
            matches = re.findall(pattern, text)
            claims["times"].extend(matches)
        
        return claims
    
    def find_contradictions(
        self, 
        backstory: str, 
        evidence_texts: List[str]
    ) -> Tuple[List[str], float]:
        """
        Find explicit contradictions between backstory and evidence.
        Returns list of contradictions and a contradiction score.
        """
        contradictions = []
        
        backstory_claims = self.extract_claims(backstory)
        evidence_combined = " ".join(evidence_texts)
        evidence_claims = self.extract_claims(evidence_combined)
        
        backstory_lower = backstory.lower()
        evidence_lower = evidence_combined.lower()
        
        # Check death contradictions
        # If backstory says someone died, check if evidence shows them alive later
        for death in backstory_claims["deaths"]:
            if isinstance(death, str):
                person = death
                # Check if person is mentioned as alive/speaking/acting in evidence
                alive_patterns = [
                    f"{person}\\s+(?:said|spoke|told|asked|replied|answered)",
                    f"{person}\\s+(?:walked|went|came|arrived|left)",
                    f"{person}\\s+(?:smiled|laughed|cried|nodded)",
                    f"{person}\\s+(?:gave|handed|showed|pointed)",
                ]
                for pattern in alive_patterns:
                    if re.search(pattern, evidence_lower):
                        contradictions.append(
                            f"Backstory claims {person} died, but evidence shows them alive/active"
                        )
                        break
        
        # Check family contradictions
        if "orphan" in backstory_claims["family"] or "orphaned" in backstory_claims["family"]:
            # If backstory says orphan, check if parents appear alive in evidence
            parent_alive = [
                r"(?:his|her)\s+(?:father|mother)\s+(?:said|told|gave|showed)",
                r"(?:father|mother)\s+(?:smiled|nodded|replied|answered)",
            ]
            for pattern in parent_alive:
                if re.search(pattern, evidence_lower):
                    contradictions.append(
                        "Backstory claims character is orphan, but evidence shows parent alive"
                    )
                    break
        
        # Check "only child" contradiction
        if "only_child" in str(backstory_claims["family"]):
            sibling_patterns = [
                r"(?:his|her)\s+(?:brother|sister)",
                r"(?:brothers?|sisters?)\s+(?:said|told|came)",
            ]
            for pattern in sibling_patterns:
                if re.search(pattern, evidence_lower):
                    contradictions.append(
                        "Backstory claims only child, but evidence mentions siblings"
                    )
                    break
        
        # Check if backstory mentions character meeting someone who doesn't exist
        # or events that couldn't have happened
        
        # Calculate contradiction score
        if contradictions:
            score = min(1.0, len(contradictions) * 0.3)
        else:
            score = 0.0
        
        return contradictions, score


class SemanticConsistencyAnalyzer:
    """
    Analyzes semantic consistency between backstory and evidence
    using embedding similarity.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def compute_consistency_score(
        self,
        backstory: str,
        evidence_texts: List[str],
        character_name: str
    ) -> Tuple[float, List[str]]:
        """
        Compute semantic consistency score.
        Returns score (0-1, higher = more consistent) and supporting evidence.
        """
        if not evidence_texts:
            return 0.5, []
        
        # Split backstory into sentences
        backstory_sentences = self._split_sentences(backstory)
        
        # Embed all texts
        backstory_embeddings = self.model.encode(backstory_sentences)
        evidence_embeddings = self.model.encode(evidence_texts)
        
        # For each backstory sentence, find best matching evidence
        sentence_scores = []
        supporting = []
        
        for i, bs_emb in enumerate(backstory_embeddings):
            similarities = self._cosine_similarity(
                bs_emb.reshape(1, -1),
                evidence_embeddings
            )[0]
            
            max_sim = np.max(similarities)
            best_idx = np.argmax(similarities)
            
            sentence_scores.append(max_sim)
            
            if max_sim > 0.5:
                supporting.append(f"'{backstory_sentences[i][:50]}...' matches evidence")
        
        # Overall score - average of sentence scores
        overall_score = np.mean(sentence_scores) if sentence_scores else 0.5
        
        return float(overall_score), supporting
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(a_norm, b_norm.T)


class EnhancedHuggingFaceReasoner:
    """
    Enhanced LLM reasoner with better prompts and structured analysis.
    """
    
    def __init__(
        self,
        model: str = "google/gemma-2-27b-it",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY", "")
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model}"
    
    def analyze_consistency(
        self,
        backstory: str,
        character_name: str,
        evidence_texts: List[str],
        book_name: str
    ) -> Tuple[int, float, str]:
        """
        Analyze consistency with enhanced prompting.
        Returns (prediction, confidence, rationale).
        """
        evidence_str = "\n---\n".join(evidence_texts[:6])
        
        prompt = f"""You are analyzing whether a character backstory is CONSISTENT or CONTRADICTS the original novel.

TASK: Determine if this backstory for "{character_name}" from "{book_name}" contradicts the original text.

BACKSTORY TO ANALYZE:
{backstory}

EVIDENCE FROM THE ORIGINAL NOVEL:
{evidence_str}

ANALYSIS INSTRUCTIONS:
1. Look for DIRECT CONTRADICTIONS - where the backstory explicitly states something different from the evidence
2. Check for IMPOSSIBLE EVENTS - things that couldn't happen given what we know
3. Verify FAMILY RELATIONSHIPS - deaths, births, relatives mentioned
4. Check TIMELINE CONSISTENCY - events in wrong order or impossible timing

IMPORTANT RULES:
- A backstory CONTRADICTS (0) if ANY claim directly conflicts with the evidence
- A backstory is CONSISTENT (1) if no claims conflict, even if some are unverifiable
- Focus on FACTS, not interpretations or feelings
- Death claims are critical: if backstory says someone died but evidence shows them alive, it's a contradiction

Respond with ONLY a JSON object:
{{"prediction": 0 or 1, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

        response = self._call_api(prompt)
        
        # Parse response
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                import json
                result = json.loads(json_match.group())
                return (
                    int(result.get("prediction", 1)),
                    float(result.get("confidence", 0.5)),
                    result.get("reason", "")
                )
        except:
            pass
        
        # Fallback: look for keywords
        response_lower = response.lower()
        if "contradict" in response_lower or "inconsistent" in response_lower:
            return 0, 0.6, "LLM detected contradiction"
        
        return 1, 0.5, "No clear contradiction found"
    
    def _call_api(self, prompt: str, max_tokens: int = 300) -> str:
        """Call HuggingFace API."""
        if not self.api_key:
            return ""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.1,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
            else:
                return ""
        except Exception as e:
            return ""
        
        return ""


class KeywordContradictionChecker:
    """
    Fast keyword-based contradiction checker.
    Very effective for catching obvious contradictions.
    """
    
    def __init__(self):
        # Key contradiction indicators
        self.contradiction_keywords = {
            "death": {
                "backstory": ["killed", "died", "dead", "death", "murdered", "slain"],
                "evidence_negates": ["said", "spoke", "told", "walked", "came", "smiled", "nodded", "replied"]
            },
            "orphan": {
                "backstory": ["orphan", "orphaned", "parents died", "mother died", "father died"],
                "evidence_negates": ["his father said", "her father said", "his mother said", "her mother said",
                                    "father told", "mother told", "father gave", "mother gave"]
            },
            "only_child": {
                "backstory": ["only child", "no siblings", "no brothers", "no sisters"],
                "evidence_negates": ["his brother", "her brother", "his sister", "her sister", 
                                    "brothers", "sisters", "sibling"]
            }
        }
    
    def check(self, backstory: str, evidence_texts: List[str]) -> Tuple[bool, List[str]]:
        """
        Check for keyword-based contradictions.
        Returns (has_contradiction, list_of_issues).
        """
        issues = []
        backstory_lower = backstory.lower()
        evidence_lower = " ".join(evidence_texts).lower()
        
        for category, patterns in self.contradiction_keywords.items():
            backstory_match = any(kw in backstory_lower for kw in patterns["backstory"])
            if backstory_match:
                evidence_negates = any(kw in evidence_lower for kw in patterns["evidence_negates"])
                if evidence_negates:
                    issues.append(f"{category}_contradiction")
        
        return len(issues) > 0, issues


class EnsembleConsistencyClassifier:
    """
    Ensemble classifier that combines multiple strategies.
    """
    
    def __init__(self, hf_api_key: Optional[str] = None):
        self.pattern_detector = ContradictionPatternDetector()
        self.semantic_analyzer = SemanticConsistencyAnalyzer()
        self.keyword_checker = KeywordContradictionChecker()
        self.llm_reasoner = EnhancedHuggingFaceReasoner(api_key=hf_api_key)
        
        # Weights for ensemble (tuned for best accuracy)
        self.weights = {
            "pattern": 0.35,      # Pattern detection is highly reliable
            "keyword": 0.30,      # Keyword matching is fast and accurate
            "semantic": 0.15,     # Semantic similarity provides soft signal
            "llm": 0.20           # LLM provides reasoning but can be noisy
        }
    
    def predict(
        self,
        backstory: str,
        character_name: str,
        evidence_texts: List[str],
        book_name: str
    ) -> EnhancedResult:
        """
        Make ensemble prediction using multiple strategies.
        """
        signals = {
            "pattern": {"score": 0.0, "contradiction": False, "details": []},
            "keyword": {"score": 0.0, "contradiction": False, "details": []},
            "semantic": {"score": 0.5, "details": []},
            "llm": {"score": 0.5, "prediction": 1, "rationale": ""}
        }
        
        # 1. Pattern-based detection
        contradictions, pattern_score = self.pattern_detector.find_contradictions(
            backstory, evidence_texts
        )
        signals["pattern"]["score"] = pattern_score
        signals["pattern"]["contradiction"] = len(contradictions) > 0
        signals["pattern"]["details"] = contradictions
        
        # 2. Keyword-based detection
        has_keyword_contradiction, keyword_issues = self.keyword_checker.check(
            backstory, evidence_texts
        )
        signals["keyword"]["contradiction"] = has_keyword_contradiction
        signals["keyword"]["score"] = 1.0 if has_keyword_contradiction else 0.0
        signals["keyword"]["details"] = keyword_issues
        
        # 3. Semantic consistency
        semantic_score, semantic_support = self.semantic_analyzer.compute_consistency_score(
            backstory, evidence_texts, character_name
        )
        signals["semantic"]["score"] = semantic_score
        signals["semantic"]["details"] = semantic_support
        
        # 4. LLM reasoning (if API key available)
        if self.llm_reasoner.api_key:
            llm_pred, llm_conf, llm_rationale = self.llm_reasoner.analyze_consistency(
                backstory, character_name, evidence_texts, book_name
            )
            signals["llm"]["prediction"] = llm_pred
            signals["llm"]["score"] = llm_conf
            signals["llm"]["rationale"] = llm_rationale
        
        # Ensemble decision
        # Strong contradiction signals override everything
        if signals["pattern"]["contradiction"] or signals["keyword"]["contradiction"]:
            prediction = 0
            confidence = 0.85
            rationale = "Strong contradiction detected: " + "; ".join(
                signals["pattern"]["details"] + signals["keyword"]["details"]
            )[:200]
        else:
            # Weighted voting
            contradiction_score = (
                signals["pattern"]["score"] * self.weights["pattern"] +
                signals["keyword"]["score"] * self.weights["keyword"] +
                (1 - signals["semantic"]["score"]) * self.weights["semantic"] +
                (1 - signals["llm"]["prediction"]) * self.weights["llm"]
            )
            
            if contradiction_score > 0.4:
                prediction = 0
                confidence = min(0.9, 0.5 + contradiction_score)
            else:
                prediction = 1
                confidence = min(0.9, 0.5 + (1 - contradiction_score))
            
            rationale = f"Ensemble scores - Pattern: {signals['pattern']['score']:.2f}, " \
                       f"Semantic: {signals['semantic']['score']:.2f}"
            if signals["llm"]["rationale"]:
                rationale += f". LLM: {signals['llm']['rationale'][:100]}"
        
        return EnhancedResult(
            prediction=prediction,
            confidence=confidence,
            rationale=rationale,
            contradiction_signals=signals["pattern"]["details"] + signals["keyword"]["details"],
            support_signals=signals["semantic"]["details"],
            semantic_score=signals["semantic"]["score"],
            pattern_score=signals["pattern"]["score"],
            llm_score=signals["llm"]["score"]
        )
