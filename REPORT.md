# KDSH 2026 Track A: Narrative Consistency Classification

## Team: [TEAM_NAME]

---

## Executive Summary

This report presents our solution for Track A of the Kharagpur Data Science Hackathon 2026. We developed a **Pathway-based RAG (Retrieval-Augmented Generation) pipeline** that determines whether hypothetical character backstories are consistent with or contradict the original novel text.

**Key Results:**
- Accuracy: [XX.XX%]
- F1 Score: [X.XX]
- Approach: Multi-stage evidence retrieval + LLM-based causal reasoning

---

## 1. Problem Understanding

### 1.1 Task Definition
Given:
- A complete long-form narrative (100k+ words)
- A hypothetical backstory for a central character

**Objective:** Determine if the backstory is **consistent** (1) or **contradicts** (0) the original narrative.

### 1.2 Key Challenges
1. **Long Context Management**: Novels span 100k+ words, exceeding typical LLM context windows
2. **Causal Reasoning**: Must trace how backstory events would affect later narrative developments
3. **Evidence Aggregation**: Conclusions must be supported by signals from multiple text sections
4. **Subtle Contradictions**: Many contradictions are implicit rather than explicit

---

## 2. Approach Overview

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PATHWAY DOCUMENT LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Book 1    │    │   Book 2    │    │   Book N    │         │
│  │  Chunking   │    │  Chunking   │    │  Chunking   │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│                   ┌─────────────────┐                           │
│                   │  Vector Index   │ (Sentence Transformers)   │
│                   └────────┬────────┘                           │
└────────────────────────────┼────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                    RETRIEVAL LAYER                               │
├────────────────────────────┼────────────────────────────────────┤
│                            ▼                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Query     │───▶│  Semantic   │───▶│  Character  │          │
│  │ Generation  │    │   Search    │    │  Filtering  │          │
│  └─────────────┘    └─────────────┘    └──────┬──────┘          │
│                                               │                  │
│                                        ┌──────▼──────┐          │
│                                        │  Evidence   │          │
│                                        │  Re-ranking │          │
│                                        └──────┬──────┘          │
└───────────────────────────────────────────────┼─────────────────┘
                                                │
┌───────────────────────────────────────────────┼─────────────────┐
│                    REASONING LAYER                               │
├───────────────────────────────────────────────┼─────────────────┤
│                                               ▼                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Claim     │───▶│   Claim     │───▶│   Final     │          │
│  │ Extraction  │    │ Verification│    │  Synthesis  │          │
│  └─────────────┘    └─────────────┘    └──────┬──────┘          │
│                                               │                  │
│                                        ┌──────▼──────┐          │
│                                        │ Prediction  │          │
│                                        │ + Rationale │          │
│                                        └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Pathway Integration

We use **Pathway** ([github.com/pathwaycom/pathway](https://github.com/pathwaycom/pathway)) as the core document processing and indexing layer:

1. **Document Ingestion**: Loading and cleaning raw book text files
2. **Intelligent Chunking**: Splitting novels into overlapping chunks while preserving chapter boundaries
3. **Vector Indexing**: Creating embeddings for semantic search
4. **Retrieval Pipeline**: Efficient similarity search across long documents

---

## 3. Technical Implementation

### 3.1 Document Processing

**Chunking Strategy:**
- Chunk size: 1000 tokens with 200 token overlap
- Chapter-aware splitting to preserve narrative structure
- Sentence boundary detection for clean chunk boundaries

```python
# Key chunking parameters
CHUNK_SIZE = 1000       # characters
CHUNK_OVERLAP = 200     # for context continuity
```

**Rationale:** The overlap ensures that important context spanning chunk boundaries isn't lost. Chapter-aware splitting helps maintain narrative coherence.

### 3.2 Evidence Retrieval

**Multi-Stage Retrieval Pipeline:**

1. **Query Expansion**: Extract individual claims from backstory
2. **Semantic Search**: Dense retrieval using sentence-transformers
3. **Character Filtering**: Boost passages mentioning the target character
4. **Diversity Re-ranking**: Ensure evidence from multiple sections

```python
# Retrieval configuration
TOP_K_RETRIEVAL = 15
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### 3.3 Consistency Reasoning

**Three-Phase Reasoning:**

1. **Claim Extraction**: LLM extracts verifiable claims from backstory
2. **Claim Verification**: Each claim analyzed against retrieved evidence
3. **Synthesis**: Final judgment combining all claim analyses

**Decision Logic:**
- **CONTRADICT (0)**: If ANY claim is directly contradicted by evidence
- **CONSISTENT (1)**: If no contradictions found and backstory is plausible

---

## 4. Handling Long Context

### 4.1 Challenges
- Novels contain 100k+ words
- Important evidence may be distributed across the entire text
- LLM context windows are limited (4k-128k tokens)

### 4.2 Our Solution

| Challenge | Solution |
|-----------|----------|
| Context length | Chunking + retrieval instead of full-text processing |
| Distributed evidence | Multi-query retrieval from different parts of backstory |
| Context coherence | Chapter-aware chunking + overlap |
| Relevance | Character-based filtering + semantic re-ranking |

---

## 5. Distinguishing Causal Signals from Noise

### 5.1 Types of Evidence

| Type | Description | Weight |
|------|-------------|--------|
| **Direct Contradiction** | Evidence explicitly contradicts backstory | High |
| **Causal Impossibility** | Backstory makes later events impossible | High |
| **Timeline Conflict** | Events cannot occur in stated order | Medium |
| **Character Inconsistency** | Actions contradict established character | Medium |
| **Unverifiable Claims** | Neither supported nor contradicted | Low |

### 5.2 Noise Reduction

- **Character-focused retrieval**: Prioritize passages mentioning the character
- **Claim decomposition**: Analyze individual claims separately
- **Section diversity**: Avoid over-reliance on single text sections
- **Confidence scoring**: Track certainty of judgments

---

## 6. Limitations and Failure Cases

### 6.1 Known Limitations

1. **Implicit Information**: May miss contradictions that require deep inference
2. **Retrieval Gaps**: Important evidence might not be retrieved
3. **LLM Hallucination**: Model may generate plausible but incorrect reasoning
4. **Timeline Complexity**: Multi-threaded narratives are challenging

### 6.2 Observed Failure Modes

| Failure Type | Example | Mitigation |
|--------------|---------|------------|
| False Positive | Marks contradiction when claim is just unverified | Require explicit evidence |
| False Negative | Misses subtle causal contradiction | Multi-claim analysis |
| Retrieval Miss | Important passage not in top-k | Increase k, multi-query |

---

## 7. Results

### 7.1 Training Set Performance

| Metric | Value |
|--------|-------|
| Accuracy | [XX.XX%] |
| Precision | [X.XX] |
| Recall | [X.XX] |
| F1 Score | [X.XX] |

### 7.2 Analysis by Book

| Book | Accuracy | Notes |
|------|----------|-------|
| In Search of the Castaways | [XX%] | |
| The Count of Monte Cristo | [XX%] | |

---

## 8. Conclusion

Our Pathway-based RAG pipeline demonstrates effective narrative consistency classification by:

1. **Leveraging Pathway** for scalable document processing and retrieval
2. **Multi-stage retrieval** for comprehensive evidence gathering
3. **Structured reasoning** with claim extraction and verification
4. **Explicit evidence linkage** for interpretable predictions

The approach balances accuracy with explainability, providing rationales that trace back to specific textual evidence.

---

## Appendix A: Reproducibility

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-key-here"

# Run pipeline
python main.py --mode full
```

### File Structure
```
KDSH-2026/
├── main.py                 # Entry point
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── src/
│   ├── document_processor.py  # Pathway-based document handling
│   ├── retriever.py           # Evidence retrieval
│   ├── consistency_reasoner.py # LLM reasoning
│   └── pipeline.py            # Main pipeline orchestration
├── Dataset/
│   ├── Books/              # Novel text files
│   ├── train (1).csv       # Training data
│   └── test (1).csv        # Test data
└── output/
    └── results.csv         # Predictions
```

---

## Appendix B: References

1. Pathway Framework: https://github.com/pathwaycom/pathway
2. Sentence Transformers: https://www.sbert.net/
3. OpenAI API: https://platform.openai.com/docs/
