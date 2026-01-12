"""
Generate labels for test(1).csv and save to output folder.
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
    print("KDSH 2026 - Generate Test Labels")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize
    doc_processor = NarrativeDocumentProcessor(chunk_size=800, chunk_overlap=200)
    classifier = HybridConsistencyClassifier()
    
    # Load and index books
    print("Loading and indexing books...")
    book_files = {
        "In Search of the Castaways": BOOKS_DIR / "In search of the castaways.txt",
        "The Count of Monte Cristo": BOOKS_DIR / "The Count of Monte Cristo.txt"
    }
    
    for book_name, book_path in book_files.items():
        if book_path.exists():
            print(f"  Processing: {book_name}")
            text = doc_processor.load_book(book_path)
            chunks = doc_processor.chunk_text(text, book_name)
            classifier.index_book(book_name, chunks)
    
    # Load test data
    print("\n" + "=" * 60)
    print("GENERATING LABELS FOR TEST DATA")
    print("=" * 60)
    
    test_df = pd.read_csv(TEST_FILE)
    print(f"Loaded {len(test_df)} test examples")
    
    # Generate predictions
    results = []
    import time
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating labels"):
        prediction, confidence, rationale = classifier.classify(
            backstory=row['content'],
            character_name=row['char'],
            book_name=row['book_name']
        )
        
        # Convert to label string
        label = "consistent" if prediction == 1 else "contradict"
        
        results.append({
            'id': row['id'],
            'book_name': row['book_name'],
            'char': row['char'],
            'label': label,
            'label_binary': prediction,
            'confidence': confidence,
            'rationale': rationale
        })
        
        # Small delay to avoid rate limiting
        if "LLM:" in rationale:
            time.sleep(2)  # 2 second delay between LLM calls
        elif "Error: 402" in rationale or "Payment Required" in rationale:
            print(f"\n[WARNING] Quota limit reached at ID {row['id']}. Waiting 60 seconds...")
            time.sleep(60)  # Wait a minute and try again
    
    results_df = pd.DataFrame(results)
    
    # Save full results with all details
    full_output_path = OUTPUT_DIR / "test_labels_full.csv"
    results_df.to_csv(full_output_path, index=False)
    print(f"\n[OK] Full results saved to: {full_output_path}")
    
    # Save simplified format (id, label)
    simple_df = results_df[['id', 'label', 'label_binary']].copy()
    simple_output_path = OUTPUT_DIR / "test_labels.csv"
    simple_df.to_csv(simple_output_path, index=False)
    print(f"[OK] Simple labels saved to: {simple_output_path}")
    
    # Save submission format (Story ID, Prediction, Rationale)
    submission_df = results_df[['id', 'label_binary', 'rationale']].copy()
    submission_df.columns = ['Story ID', 'Prediction', 'Rationale']
    submission_path = OUTPUT_DIR / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"[OK] Submission format saved to: {submission_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    consistent_count = (results_df['label_binary'] == 1).sum()
    contradict_count = (results_df['label_binary'] == 0).sum()
    
    print(f"\nTotal examples: {len(results_df)}")
    print(f"  Consistent (1): {consistent_count} ({consistent_count/len(results_df)*100:.1f}%)")
    print(f"  Contradict (0): {contradict_count} ({contradict_count/len(results_df)*100:.1f}%)")
    
    # Show all predictions
    print("\n" + "-" * 60)
    print("ALL PREDICTIONS")
    print("-" * 60)
    
    for _, row in results_df.iterrows():
        label_str = "Consistent" if row['label_binary'] == 1 else "Contradict"
        print(f"ID {row['id']:3d}: {label_str:10s} | {row['char'][:20]:20s} | {row['book_name'][:25]}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
