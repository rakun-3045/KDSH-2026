"""
Run hybrid classifier on test data only and show predictions.
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import BOOKS_DIR, OUTPUT_DIR, TEST_FILE
from src.document_processor import NarrativeDocumentProcessor
from src.hybrid_classifier import HybridConsistencyClassifier


def main():
    print("=" * 60)
    print("KDSH 2026 - Test Data Prediction")
    print("=" * 60)
    
    # Initialize
    doc_processor = NarrativeDocumentProcessor(chunk_size=800, chunk_overlap=200)
    classifier = HybridConsistencyClassifier()
    
    # Load books
    print("\nLoading and indexing books...")
    book_files = {
        "In Search of the Castaways": BOOKS_DIR / "In search of the castaways.txt",
        "The Count of Monte Cristo": BOOKS_DIR / "The Count of Monte Cristo.txt"
    }
    
    for book_name, book_path in book_files.items():
        if book_path.exists():
            text = doc_processor.load_book(book_path)
            chunks = doc_processor.chunk_text(text, book_name)
            classifier.index_book(book_name, chunks)
    
    # Process test data
    print("\n" + "=" * 60)
    print("PROCESSING TEST DATA")
    print("=" * 60)
    
    test_df = pd.read_csv(TEST_FILE)
    print(f"Loaded {len(test_df)} test examples")
    print(f"Columns: {list(test_df.columns)}")
    
    results = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prediction, confidence, rationale = classifier.classify(
            backstory=row['content'],
            character_name=row['char'],
            book_name=row['book_name']
        )
        
        results.append({
            'id': row['id'],
            'book_name': row['book_name'],
            'char': row['char'],
            'prediction': prediction,
            'confidence': confidence,
            'rationale': rationale[:100] if rationale else ""
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    
    consistent_count = (results_df['prediction'] == 1).sum()
    contradict_count = (results_df['prediction'] == 0).sum()
    
    print(f"\nTotal examples: {len(results_df)}")
    print(f"  Consistent (1): {consistent_count} ({consistent_count/len(results_df)*100:.1f}%)")
    print(f"  Contradict (0): {contradict_count} ({contradict_count/len(results_df)*100:.1f}%)")
    
    # Show contradictions found
    contradictions = results_df[results_df['prediction'] == 0]
    if len(contradictions) > 0:
        print(f"\n--- Contradictions Detected ({len(contradictions)}) ---")
        for _, row in contradictions.iterrows():
            print(f"  ID {row['id']}: {row['char']} ({row['book_name'][:20]})")
            print(f"    Reason: {row['rationale']}")
    
    # Save results
    output_path = OUTPUT_DIR / "test_predictions.csv"
    results_df[['id', 'prediction', 'rationale']].to_csv(output_path, index=False)
    print(f"\n[OK] Predictions saved to: {output_path}")
    
    # Note about accuracy
    print("\n" + "=" * 60)
    print("NOTE: The test dataset does NOT have labels,")
    print("so accuracy cannot be calculated.")
    print("=" * 60)
    
    # Training accuracy reminder
    print("\nTraining set accuracy was: 98.75% (79/80 correct)")


if __name__ == "__main__":
    main()
