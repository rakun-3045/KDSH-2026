"""
Enhanced Pipeline Module
High-accuracy pipeline for narrative consistency classification.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import os

from .document_processor import NarrativeDocumentProcessor
from .enhanced_retriever import EnhancedEvidenceRetriever
from .enhanced_reasoner import EnsembleConsistencyClassifier, EnhancedResult


class EnhancedNarrativeConsistencyPipeline:
    """
    Enhanced pipeline with multiple accuracy improvements:
    1. Multi-strategy retrieval
    2. Pattern-based contradiction detection
    3. Ensemble classification
    4. Character-focused evidence
    """
    
    def __init__(
        self,
        books_dir: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 800,  # Smaller chunks for better precision
        chunk_overlap: int = 200,
        top_k_retrieval: int = 25,  # More evidence
        hf_api_key: Optional[str] = None
    ):
        self.books_dir = books_dir
        
        # Initialize components
        self.doc_processor = NarrativeDocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model
        )
        
        self.retriever = EnhancedEvidenceRetriever(
            embedding_model=embedding_model,
            top_k=top_k_retrieval
        )
        
        self.classifier = EnsembleConsistencyClassifier(
            hf_api_key=hf_api_key or os.getenv("HUGGINGFACE_API_KEY", "")
        )
        
        self.books_indexed = False
        self.book_chunks: Dict[str, List[Dict]] = {}
    
    def initialize(self) -> None:
        """Initialize by loading and indexing books."""
        print("=" * 60)
        print("Initializing Enhanced Narrative Consistency Pipeline")
        print("=" * 60)
        
        book_files = {
            "In Search of the Castaways": self.books_dir / "In search of the castaways.txt",
            "The Count of Monte Cristo": self.books_dir / "The Count of Monte Cristo.txt"
        }
        
        for book_name, book_path in book_files.items():
            if book_path.exists():
                print(f"\nProcessing: {book_name}")
                
                text = self.doc_processor.load_book(book_path)
                print(f"  - Loaded {len(text):,} characters")
                
                chunks = self.doc_processor.chunk_text(text, book_name)
                print(f"  - Created {len(chunks)} chunks")
                
                self.book_chunks[book_name] = chunks
                self.retriever.index_book(book_name, chunks)
        
        self.books_indexed = True
        print("\n" + "=" * 60)
        print("Enhanced initialization complete!")
        print("=" * 60)
    
    def process_single(
        self,
        example_id: int,
        book_name: str,
        character_name: str,
        backstory: str
    ) -> Tuple[int, str, float]:
        """Process a single example."""
        if not self.books_indexed:
            self.initialize()
        
        # Normalize book name
        book_name = self._normalize_book_name(book_name)
        
        if book_name not in self.book_chunks:
            return 1, "Book not found", 0.0
        
        # Get comprehensive evidence
        evidence_chunks = self.retriever.retrieve_comprehensive(
            backstory, character_name, book_name
        )
        
        evidence_texts = [chunk.text for chunk in evidence_chunks]
        
        # Run ensemble classifier
        result = self.classifier.predict(
            backstory=backstory,
            character_name=character_name,
            evidence_texts=evidence_texts,
            book_name=book_name
        )
        
        return result.prediction, result.rationale, result.confidence
    
    def process_dataset(
        self,
        data_df: pd.DataFrame,
        output_path: Optional[Path] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Process a full dataset."""
        if not self.books_indexed:
            self.initialize()
        
        results = []
        iterator = tqdm(data_df.iterrows(), total=len(data_df)) if verbose else data_df.iterrows()
        
        for idx, row in iterator:
            example_id = row.get('id', idx)
            book_name = row.get('book_name', '')
            character_name = row.get('char', '')
            backstory = row.get('content', '')
            
            if verbose:
                iterator.set_description(f"Processing {character_name[:15]}...")
            
            prediction, rationale, confidence = self.process_single(
                example_id=example_id,
                book_name=book_name,
                character_name=character_name,
                backstory=backstory
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
    
    def evaluate(
        self,
        predictions_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> Dict:
        """Evaluate predictions."""
        merged = predictions_df.merge(
            labels_df[['id', 'label']],
            on='id',
            how='inner'
        )
        
        merged['label_binary'] = merged['label'].apply(
            lambda x: 1 if x == 'consistent' else 0
        )
        
        correct = (merged['prediction'] == merged['label_binary']).sum()
        total = len(merged)
        accuracy = correct / total if total > 0 else 0
        
        # Detailed metrics
        tp = ((merged['prediction'] == 0) & (merged['label_binary'] == 0)).sum()
        fp = ((merged['prediction'] == 0) & (merged['label_binary'] == 1)).sum()
        fn = ((merged['prediction'] == 1) & (merged['label_binary'] == 0)).sum()
        tn = ((merged['prediction'] == 1) & (merged['label_binary'] == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total': total,
            'correct': correct,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    
    def _normalize_book_name(self, book_name: str) -> str:
        """Normalize book name."""
        book_name_lower = book_name.lower()
        
        for indexed_name in self.book_chunks.keys():
            if indexed_name.lower() in book_name_lower or book_name_lower in indexed_name.lower():
                return indexed_name
        
        if "castaway" in book_name_lower or "search" in book_name_lower:
            return "In Search of the Castaways"
        if "monte cristo" in book_name_lower or "count" in book_name_lower:
            return "The Count of Monte Cristo"
        
        return book_name
