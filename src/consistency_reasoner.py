"""
Consistency Reasoning Module
Uses LLMs for causal and logical reasoning about narrative consistency
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import re
import requests

# Support multiple LLM providers
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Hugging Face Inference API
HUGGINGFACE_AVAILABLE = True  # Uses requests, always available


@dataclass
class ConsistencyResult:
    """Result of consistency analysis."""
    prediction: int  # 1 = consistent, 0 = contradict
    confidence: float
    rationale: str
    supporting_evidence: List[str]
    contradicting_evidence: List[str]


class NarrativeConsistencyReasoner:
    """
    LLM-based reasoner for narrative consistency checking.
    
    Implements a multi-step reasoning approach:
    1. Evidence analysis - examine retrieved passages
    2. Claim verification - check each backstory claim
    3. Causal reasoning - assess logical consistency
    4. Final judgment - synthesize evidence into prediction
    """
    
    def __init__(
        self,
        model: str = "google/gemma-2-27b-it",
        temperature: float = 0.1,
        api_key: Optional[str] = None
    ):
        self.model = model
        self.temperature = temperature
        
        # Initialize LLM client based on model
        if "gemma" in model.lower() or "huggingface" in model.lower() or "google/" in model.lower():
            # Use Hugging Face Inference API
            self.hf_api_key = api_key or os.getenv("HUGGINGFACE_API_KEY", "")
            self.hf_api_url = f"https://api-inference.huggingface.co/models/{model}"
            self.provider = "huggingface"
            print(f"Using Hugging Face model: {model}")
        elif "gpt" in model.lower() or "openai" in model.lower():
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed")
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.provider = "openai"
        elif "claude" in model.lower() or "anthropic" in model.lower():
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed")
            self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.provider = "anthropic"
        else:
            # Default to Hugging Face
            self.hf_api_key = api_key or os.getenv("HUGGINGFACE_API_KEY", "")
            self.hf_api_url = f"https://api-inference.huggingface.co/models/{model}"
            self.provider = "huggingface"
            print(f"Using Hugging Face model: {model}")
    
    def analyze_consistency(
        self,
        backstory: str,
        character_name: str,
        evidence_passages: List[str],
        book_name: str
    ) -> ConsistencyResult:
        """
        Analyze whether a backstory is consistent with narrative evidence.
        
        Uses a structured reasoning approach with explicit evidence linkage.
        """
        # Step 1: Extract key claims from backstory
        claims = self._extract_claims(backstory)
        
        # Step 2: Analyze each claim against evidence
        claim_analyses = []
        for claim in claims[:10]:  # Limit to top 10 claims
            analysis = self._analyze_claim(
                claim, 
                evidence_passages, 
                character_name,
                book_name
            )
            claim_analyses.append(analysis)
        
        # Step 3: Synthesize final judgment
        result = self._synthesize_judgment(
            backstory,
            character_name,
            claim_analyses,
            evidence_passages,
            book_name
        )
        
        return result
    
    def _extract_claims(self, backstory: str) -> List[str]:
        """Extract verifiable claims from backstory."""
        prompt = f"""Extract the key factual claims from this character backstory.
Focus on claims about:
- Birth, death, or major life events
- Family relationships
- Actions taken by the character
- Locations visited or lived in
- Character traits or beliefs formed by specific events

Backstory:
{backstory}

List each claim as a separate line, starting with a dash (-).
Only include specific, verifiable claims, not general descriptions."""

        response = self._call_llm(prompt, max_tokens=500)
        
        # Parse claims from response
        claims = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('-'):
                claims.append(line[1:].strip())
            elif line and not line.startswith('#'):
                claims.append(line)
        
        return claims
    
    def _analyze_claim(
        self,
        claim: str,
        evidence_passages: List[str],
        character_name: str,
        book_name: str
    ) -> Dict:
        """Analyze a single claim against evidence."""
        evidence_text = "\n\n---\n\n".join(evidence_passages[:8])
        
        prompt = f"""You are analyzing whether a claim about the character "{character_name}" 
from the novel "{book_name}" is consistent with the original text.

CLAIM TO VERIFY:
{claim}

EVIDENCE FROM THE ORIGINAL NOVEL:
{evidence_text}

Analyze this claim by:
1. Identifying any evidence that SUPPORTS this claim
2. Identifying any evidence that CONTRADICTS this claim
3. Noting if the claim is about something NOT MENTIONED in the evidence

Respond in this JSON format:
{{
    "claim": "<the claim being analyzed>",
    "verdict": "supported" | "contradicted" | "unverifiable",
    "supporting_evidence": ["<quote 1>", "<quote 2>"],
    "contradicting_evidence": ["<quote 1>", "<quote 2>"],
    "reasoning": "<brief explanation>"
}}

Only use "contradicted" if there is CLEAR evidence against the claim.
Use "unverifiable" if the claim is plausible but not directly addressed."""

        response = self._call_llm(prompt, max_tokens=600)
        
        # Parse JSON response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        return {
            "claim": claim,
            "verdict": "unverifiable",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "reasoning": response[:200]
        }
    
    def _synthesize_judgment(
        self,
        backstory: str,
        character_name: str,
        claim_analyses: List[Dict],
        evidence_passages: List[str],
        book_name: str
    ) -> ConsistencyResult:
        """Synthesize final consistency judgment from claim analyses."""
        
        # Count verdicts
        verdicts = [a.get("verdict", "unverifiable") for a in claim_analyses]
        contradicted_count = verdicts.count("contradicted")
        supported_count = verdicts.count("supported")
        
        # Collect evidence
        supporting = []
        contradicting = []
        for analysis in claim_analyses:
            supporting.extend(analysis.get("supporting_evidence", []))
            contradicting.extend(analysis.get("contradicting_evidence", []))
        
        # Prepare analysis summary for final judgment
        analysis_summary = "\n".join([
            f"- {a.get('claim', 'Unknown')}: {a.get('verdict', 'unknown')} - {a.get('reasoning', '')}"
            for a in claim_analyses
        ])
        
        prompt = f"""You are making a final judgment on whether a character backstory is 
CONSISTENT or CONTRADICTS the original novel "{book_name}".

CHARACTER: {character_name}

BACKSTORY BEING EVALUATED:
{backstory}

CLAIM-BY-CLAIM ANALYSIS:
{analysis_summary}

SUMMARY:
- Claims that are SUPPORTED by evidence: {supported_count}
- Claims that are CONTRADICTED by evidence: {contradicted_count}
- Claims that are UNVERIFIABLE: {len(verdicts) - supported_count - contradicted_count}

DECISION CRITERIA:
- Mark as CONTRADICT (0) if ANY claim directly contradicts the original text
- Mark as CONSISTENT (1) if no claims contradict and the backstory is plausible
- Pay special attention to: family relationships, death/birth events, character actions, timeline

Your response must be in this exact JSON format:
{{
    "prediction": 0 or 1,
    "confidence": 0.0 to 1.0,
    "rationale": "<2-3 sentence explanation of your decision>"
}}"""

        response = self._call_llm(prompt, max_tokens=400)
        
        # Parse response
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return ConsistencyResult(
                    prediction=int(result.get("prediction", 1)),
                    confidence=float(result.get("confidence", 0.5)),
                    rationale=result.get("rationale", ""),
                    supporting_evidence=supporting[:5],
                    contradicting_evidence=contradicting[:5]
                )
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: use simple heuristic
        prediction = 0 if contradicted_count > 0 else 1
        confidence = 0.6 if contradicted_count > 0 else 0.7
        
        return ConsistencyResult(
            prediction=prediction,
            confidence=confidence,
            rationale=f"Based on analysis of {len(claim_analyses)} claims: "
                      f"{contradicted_count} contradicted, {supported_count} supported.",
            supporting_evidence=supporting[:5],
            contradicting_evidence=contradicting[:5]
        )
    
    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call the LLM with the given prompt."""
        try:
            if self.provider == "huggingface":
                return self._call_huggingface(prompt, max_tokens)
            
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise literary analyst who carefully examines evidence to determine narrative consistency."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ""
        
        return ""
    
    def _call_huggingface(self, prompt: str, max_tokens: int = 500) -> str:
        """Call Hugging Face Inference API."""
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json"
        }
        
        # Format prompt for Gemma instruction-tuned model
        system_msg = "You are a precise literary analyst who carefully examines evidence to determine narrative consistency."
        formatted_prompt = f"<start_of_turn>user\n{system_msg}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": self.temperature,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(
                self.hf_api_url,
                headers=headers,
                json=payload,
                timeout=120  # 2 minute timeout for large models
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    return result.get("generated_text", "")
            elif response.status_code == 503:
                # Model is loading
                print("Model is loading, waiting...")
                import time
                time.sleep(30)
                return self._call_huggingface(prompt, max_tokens)
            else:
                print(f"HuggingFace API error: {response.status_code} - {response.text}")
                return ""
                
        except requests.exceptions.Timeout:
            print("HuggingFace API timeout")
            return ""
        except Exception as e:
            print(f"HuggingFace API call failed: {e}")
            return ""
        
        return ""


class RuleBasedConsistencyChecker:
    """
    Rule-based consistency checker as a fallback/supplement to LLM reasoning.
    Useful when API calls fail or for quick heuristic checks.
    """
    
    def __init__(self):
        # Contradiction patterns
        self.contradiction_patterns = [
            # Death-related contradictions
            (r"killed\s+(?:his|her)\s+(\w+)", r"\1\s+(?:was\s+)?alive"),
            (r"(\w+)\s+died", r"\1\s+(?:said|spoke|walked|arrived)"),
            # Birth/origin contradictions
            (r"born\s+in\s+(\w+)", r"native\s+of\s+(?!$1)"),
            # Relationship contradictions
            (r"only\s+child", r"(?:brother|sister)"),
            (r"orphan", r"(?:his|her)\s+(?:father|mother)\s+(?:said|gave|taught)"),
        ]
    
    def check_basic_consistency(
        self,
        backstory: str,
        evidence: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Perform basic rule-based consistency check.
        Returns (is_consistent, list_of_issues).
        """
        issues = []
        combined_evidence = " ".join(evidence).lower()
        backstory_lower = backstory.lower()
        
        for pattern_backstory, pattern_evidence in self.contradiction_patterns:
            backstory_matches = re.findall(pattern_backstory, backstory_lower)
            for match in backstory_matches:
                evidence_pattern = pattern_evidence.replace("$1", match)
                if re.search(evidence_pattern, combined_evidence):
                    issues.append(f"Potential contradiction: backstory says '{match}' "
                                  f"but evidence suggests otherwise")
        
        return len(issues) == 0, issues
