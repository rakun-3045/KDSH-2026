"""
Main Pipeline Module
Orchestrates the complete narrative consistency classification workflow
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import json
import os

# Try to import Pathway (Linux/MacOS only)
try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False

from .document_processor import NarrativeDocumentProcessor
from .retriever import NarrativeEvidenceRetriever, extract_backstory_claims
from .consistency_reasoner import (
    NarrativeConsistencyReasoner, 
    ConsistencyResult,
    RuleBasedConsistencyChecker
)


class NarrativeConsistencyPipeline:
    """
    End-to-end pipeline for narrative consistency classification.
    
    Workflow:
    1. Load and index books using Pathway
    2. For each test example:
       a. Extract character backstory claims
       b. Retrieve relevant evidence from the book
       c. Reason about consistency using LLM
       d. Generate prediction and rationale
    """
    
    def __init__(
        self,
        books_dir: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "google/gemma-2-27b-it",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k_retrieval: int = 15,
        use_rule_based_fallback: bool = True,
        use_local_llm: bool = False,
        api_key: Optional[str] = None
    ):
        self.books_dir = books_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.use_local_llm = use_local_llm
        self.api_key = api_key
        
        # Initialize components
        self.doc_processor = NarrativeDocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model
        )
        
        self.retriever = NarrativeEvidenceRetriever(
            embedding_model=embedding_model,
            top_k=top_k_retrieval
        )
        
        self.reasoner = None  # Initialized lazily
        self.rule_checker = RuleBasedConsistencyChecker() if use_rule_based_fallback else None
        
        # Book data
        self.books_indexed = False
        self.book_chunks: Dict[str, List[Dict]] = {}
        
    def initialize(self) -> None:
        """Initialize the pipeline by loading and indexing books."""
        print("=" * 60)
        print("Initializing Narrative Consistency Pipeline")
        print("=" * 60)
        
        # Define book mappings
        book_files = {
            "In Search of the Castaways": self.books_dir / "In search of the castaways.txt",
            "The Count of Monte Cristo": self.books_dir / "The Count of Monte Cristo.txt"
        }
        
        for book_name, book_path in book_files.items():
            if book_path.exists():
                print(f"\nProcessing: {book_name}")
                
                # Load and chunk the book
                text = self.doc_processor.load_book(book_path)
                print(f"  - Loaded {len(text):,} characters")
                
                chunks = self.doc_processor.chunk_text(text, book_name)
                print(f"  - Created {len(chunks)} chunks")
                
                self.book_chunks[book_name] = chunks
                
                # Index for retrieval
                self.retriever.index_documents(book_name, chunks)
            else:
                print(f"Warning: Book file not found: {book_path}")
        
        self.books_indexed = True
        print("\n" + "=" * 60)
        print("Initialization complete!")
        print("=" * 60)
    
    def _ensure_initialized(self) -> None:
        """Ensure pipeline is initialized before processing."""
        if not self.books_indexed:
            self.initialize()
        
        if self.reasoner is None:
            try:
                self.reasoner = NarrativeConsistencyReasoner(
                    model=self.llm_model,
                    use_local=self.use_local_llm,
                    api_key=self.api_key
                )
            except Exception as e:
                print(f"Warning: Could not initialize LLM reasoner: {e}")
                print("Will use rule-based fallback only.")
    
    def process_single_example(
        self,
        example_id: int,
        book_name: str,
        character_name: str,
        backstory_content: str,
        caption: Optional[str] = None
    ) -> Tuple[int, str, float]:
        """
        Process a single example and return prediction.
        
        Returns:
            Tuple of (prediction, rationale, confidence)
        """
        self._ensure_initialized()
        
        # Normalize book name
        book_name_normalized = self._normalize_book_name(book_name)
        
        if book_name_normalized not in self.book_chunks:
            print(f"Warning: Book '{book_name}' not found in index")
            return 1, "Book not found - defaulting to consistent", 0.0
        
        # Step 1: Extract claims from backstory for targeted retrieval
        claims = extract_backstory_claims(backstory_content)
        
        # Step 2: Retrieve relevant evidence
        # Use both the full backstory and individual claims as queries
        queries = [backstory_content] + claims[:5]
        
        evidence_list = self.retriever.retrieve_multi_query(
            queries=queries,
            book_name=book_name_normalized,
            character_name=character_name,
            top_k_per_query=5
        )
        
        evidence_texts = [ev.text for ev in evidence_list]
        
        # Step 3: Apply LLM reasoning (if available)
        if self.reasoner is not None:
            try:
                result = self.reasoner.analyze_consistency(
                    backstory=backstory_content,
                    character_name=character_name,
                    evidence_passages=evidence_texts,
                    book_name=book_name_normalized
                )
                return result.prediction, result.rationale, result.confidence
            except Exception as e:
                print(f"LLM reasoning failed for example {example_id}: {e}")
        
        # Step 4: Fallback to rule-based checking
        if self.rule_checker is not None:
            is_consistent, issues = self.rule_checker.check_basic_consistency(
                backstory_content, 
                evidence_texts
            )
            if issues:
                rationale = f"Rule-based check found issues: {'; '.join(issues[:2])}"
                return 0, rationale, 0.5
            else:
                return 1, "No contradictions found by rule-based checker", 0.5
        
        # Default fallback
        return 1, "Unable to determine - defaulting to consistent", 0.0
    
    def process_dataset(
        self,
        data_df: pd.DataFrame,
        output_path: Optional[Path] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Process a full dataset and generate predictions.
        
        Args:
            data_df: DataFrame with columns [id, book_name, char, caption, content]
            output_path: Optional path to save results CSV
            verbose: Whether to show progress
        
        Returns:
            DataFrame with predictions
        """
        self._ensure_initialized()
        
        results = []
        iterator = tqdm(data_df.iterrows(), total=len(data_df)) if verbose else data_df.iterrows()
        
        for idx, row in iterator:
            example_id = row.get('id', idx)
            book_name = row.get('book_name', '')
            character_name = row.get('char', '')
            backstory_content = row.get('content', '')
            caption = row.get('caption', '')
            
            if verbose:
                iterator.set_description(f"Processing {character_name[:20]}...")
            
            prediction, rationale, confidence = self.process_single_example(
                example_id=example_id,
                book_name=book_name,
                character_name=character_name,
                backstory_content=backstory_content,
                caption=caption
            )
            
            results.append({
                'id': example_id,
                'prediction': prediction,
                'rationale': rationale,
                'confidence': confidence
            })
        
        results_df = pd.DataFrame(results)
        
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
        
        return results_df
    
    def _normalize_book_name(self, book_name: str) -> str:
        """Normalize book name to match indexed name."""
        book_name_lower = book_name.lower()
        
        for indexed_name in self.book_chunks.keys():
            if indexed_name.lower() in book_name_lower or book_name_lower in indexed_name.lower():
                return indexed_name
        
        # Try partial matching
        if "castaway" in book_name_lower or "search" in book_name_lower:
            return "In Search of the Castaways"
        if "monte cristo" in book_name_lower or "count" in book_name_lower:
            return "The Count of Monte Cristo"
        
        return book_name
    
    def evaluate(
        self,
        predictions_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> Dict:
        """
        Evaluate predictions against ground truth labels.
        
        Returns:
            Dictionary with accuracy, precision, recall, F1 metrics
        """
        # Merge predictions with labels
        merged = predictions_df.merge(
            labels_df[['id', 'label']], 
            on='id', 
            how='inner'
        )
        
        # Convert labels to binary
        merged['label_binary'] = merged['label'].apply(
            lambda x: 1 if x == 'consistent' else 0
        )
        
        # Calculate metrics
        correct = (merged['prediction'] == merged['label_binary']).sum()
        total = len(merged)
        accuracy = correct / total if total > 0 else 0
        
        # Precision, recall, F1 for "contradict" class (0)
        true_positives = ((merged['prediction'] == 0) & (merged['label_binary'] == 0)).sum()
        false_positives = ((merged['prediction'] == 0) & (merged['label_binary'] == 1)).sum()
        false_negatives = ((merged['prediction'] == 1) & (merged['label_binary'] == 0)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_examples': total,
            'correct_predictions': correct
        }


# Pathway streaming pipeline (only available on Linux/MacOS)
if PATHWAY_AVAILABLE:
    def create_pathway_streaming_pipeline(
        books_dir: Path,
        input_stream: 'pw.Table'
    ) -> 'pw.Table':
        """
        Create a Pathway streaming pipeline for real-time consistency checking.
        
        This demonstrates Pathway's streaming capabilities for continuous
        data processing.
        """
        from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
        
        embedder = SentenceTransformerEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Add embeddings to input stream
        embedded_stream = input_stream.select(
            id=pw.this.id,
            book_name=pw.this.book_name,
            character=pw.this.char,
            content=pw.this.content,
            embedding=embedder(pw.this.content)
        )
        
        return embedded_stream
