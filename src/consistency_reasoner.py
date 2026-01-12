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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        api_key: Optional[str] = None,
        use_local: bool = False
    ):
        self.model = model
        self.temperature = temperature
        self.use_local = use_local
        
        # Initialize LLM client based on model
        if use_local:
            print(f"Loading local model: {model}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model, token=api_key)
                self.llm = AutoModelForCausalLM.from_pretrained(
                    model,
                    token=api_key,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
                self.provider = "local"
                print(f"Local model loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}")
            except Exception as e:
                print(f"Failed to load local model: {e}")
                raise
        elif "llama" in model.lower() or "groq" in model.lower() or "gpt-oss" in model.lower():
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed (required for Groq)")
            self.client = OpenAI(
                api_key=api_key or os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )
            self.provider = "openai" # Groq is OpenAI-compatible
            print(f"Using Groq model: {model}")
        elif "gemini" in model.lower():
            self.google_api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
            self.provider = "google"
            print(f"Using Google Gemini model: {model}")
        elif "gemma" in model.lower() or "huggingface" in model.lower() or "google/" in model.lower():
            # Use Hugging Face Inference API (new router endpoint)
            self.hf_api_key = api_key or os.getenv("HUGGINGFACE_API_KEY", "")
            self.hf_api_url = f"https://router.huggingface.co/hf-inference/models/{model}"
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
            # Default to Hugging Face (new router endpoint)
            self.hf_api_key = api_key or os.getenv("HUGGINGFACE_API_KEY", "")
            self.hf_api_url = f"https://router.huggingface.co/hf-inference/models/{model}"
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
        
        Uses a consolidated holistic approach to minimize LLM calls.
        """
        evidence_text = "\\n\\n---\\n\\n".join(evidence_passages[:8]) # Limit to top 8 passages
        
        prompt = f"""You are an expert literary analyst. Your task is to determine if a generated backstory for the character "{character_name}" is consistent with the novel "{book_name}".

BACKSTORY TO EVALUATE:
{backstory}

EVIDENCE FROM THE ORIGINAL NOVEL:
{evidence_text}

INSTRUCTIONS:
1. Identify factual claims in the backstory (ancestry, past events, relationships, death, locations).
2. Check if ANY claim directly contradicts the evidence provided.
3. If specific dates, locations, or family members in the backstory clash with the text, that is a CONTRADICTION.
4. If the backstory mentions things not strictly in the text but plausible (not contradicted), it is CONSISTENT.

RESPONSE FORMAT (JSON):
{{
    "prediction": 0 or 1,  // 0 = Contradiction, 1 = Consistent
    "confidence": 0.0 to 1.0,
    "rationale": "<2-3 sentence explanation of your decision>",
    "supporting_evidence": ["Quote 1", "Quote 2"],
    "contradicting_evidence": ["Quote A", "Quote B"]
}}

Return ONLY the JSON.
"""
        response = self._call_llm(prompt, max_tokens=1000)
        
        try:
             # Find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return ConsistencyResult(
                    prediction=int(result.get("prediction", 1)),
                    confidence=float(result.get("confidence", 0.5)),
                    rationale=result.get("rationale", "No rationale provided."),
                    supporting_evidence=result.get("supporting_evidence", []),
                    contradicting_evidence=result.get("contradicting_evidence", [])
                )
        except Exception as e:
            print(f"Holistic analysis failed parsing: {e}")
            
        # Fallback
        return ConsistencyResult(
            prediction=1,
            confidence=0.5,
            rationale="Automated consistency check (fallback - parsing failed or rate limit).",
            supporting_evidence=[],
            contradicting_evidence=[]
        )
    
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
            if self.provider == "local":
                return self._call_local(prompt, max_tokens)

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
            
            elif self.provider == "google":
                return self._call_google(prompt, max_tokens)
            
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ""

    def _call_google(self, prompt: str, max_tokens: int = 500) -> str:
        """Call Google Gemini API via REST."""
        # Clean model name for API URL
        model_name = self.model
        if "gemini" in model_name and not model_name.startswith("models/"):
            # Ensure proper format like 'models/gemini-1.5-flash'
            # But the endpoint takes 'models/{model_name}:generateContent', or just the name if we construct URL right
            # Standard public API uses "models/gemini-pro" or similar.
            pass
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.google_api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                try:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    print(f"Unexpected Google API response structure: {result}")
                    return ""
            else:
                print(f"Google API error: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            print(f"Google API call failed: {e}")
            return ""

    def _call_local(self, prompt: str, max_tokens: int = 500) -> str:
        """Call local Hugging Face model."""
        try:
            # Format prompt for instruction tuned models
            chat = [
                {"role": "user", "content": prompt}
            ]
            
            try:
                # Apply chat template if available
                formatted_prompt = self.tokenizer.apply_chat_template(
                    chat, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback format
                formatted_prompt = f"User: {prompt}\n\nModel:"

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.llm.device)
            
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode only the new tokens
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            print(f"Local inference failed: {e}")
            return ""
            return ""
        
        return ""
    
    def _call_huggingface(self, prompt: str, max_tokens: int = 500) -> str:
        """Call Hugging Face Inference API."""
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json"
        }
        
        system_msg = "You are a precise literary analyst who carefully examines evidence to determine narrative consistency."
        
        # Format prompt based on model family
        if "gemma" in self.model.lower():
            formatted_prompt = f"<start_of_turn>user\n{system_msg}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif "mistral" in self.model.lower() or "ministral" in self.model.lower():
            formatted_prompt = f"<s>[INST] {system_msg}\n\n{prompt} [/INST]"
        else:
            # Generic format
            formatted_prompt = f"{system_msg}\n\nUser: {prompt}\n\nModel:"
        
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
                # Handle tuple matches from findall (multiple groups)
                if isinstance(match, tuple):
                    match = match[0] # Take first group
                
                if not match: continue
                
                # Replace placeholders ($1 or \1) with safe matched text
                safe_match = re.escape(match)
                evidence_pattern = pattern_evidence.replace("$1", safe_match).replace(r"\1", safe_match)
                
                try:
                    if re.search(evidence_pattern, combined_evidence):
                        issues.append(f"Potential contradiction found for '{match}'")
                except re.error:
                    continue
        
        return len(issues) == 0, issues
