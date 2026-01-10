"""
KDSH 2026 Track A - Smart Classifier Pipeline
Optimized for high accuracy on narrative consistency classification.
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, BOOKS_DIR, OUTPUT_DIR, TRAIN_FILE, TEST_FILE
from src.document_processor import NarrativeDocumentProcessor
from src.smart_classifier import SmartConsistencyClassifier


def main():
    print("=" * 60)
    print("KDSH 2026 Track A - Smart Classifier")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize
    doc_processor = NarrativeDocumentProcessor(
        chunk_size=800,
        chunk_overlap=200
    )
    
    classifier = SmartConsistencyClassifier()
    
    # Load and index books
    print("Loading books...")
    book_files = {
        "In Search of the Castaways": BOOKS_DIR / "In search of the castaways.txt",
        "The Count of Monte Cristo": BOOKS_DIR / "The Count of Monte Cristo.txt"
    }
    
    for book_name, book_path in book_files.items():
        if book_path.exists():
            print(f"\nProcessing: {book_name}")
            text = doc_processor.load_book(book_path)
            print(f"  Loaded {len(text):,} characters")
            
            chunks = doc_processor.chunk_text(text, book_name)
            print(f"  Created {len(chunks)} chunks")
            
            classifier.index_book(book_name, chunks)
    
    # Process training data
    print("\n" + "=" * 60)
    print("TRAINING DATA EVALUATION")
    print("=" * 60)
    
    train_df = pd.read_csv(TRAIN_FILE)
    print(f"Loaded {len(train_df)} examples")
    
    results = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        prediction, confidence, rationale = classifier.classify(
            backstory=row['content'],
            character_name=row['char'],
            book_name=row['book_name']
        )
        
        results.append({
            'id': row['id'],
            'prediction': prediction,
            'rationale': rationale,
            'confidence': confidence
        })
    
    results_df = pd.DataFrame(results)
    
    # Evaluate
    train_df['label_binary'] = train_df['label'].apply(lambda x: 1 if x == 'consistent' else 0)
    merged = results_df.merge(train_df[['id', 'label_binary']], on='id')
    
    correct = (merged['prediction'] == merged['label_binary']).sum()
    total = len(merged)
    accuracy = correct / total
    
    tp = ((merged['prediction'] == 0) & (merged['label_binary'] == 0)).sum()
    fp = ((merged['prediction'] == 0) & (merged['label_binary'] == 1)).sum()
    fn = ((merged['prediction'] == 1) & (merged['label_binary'] == 0)).sum()
    tn = ((merged['prediction'] == 1) & (merged['label_binary'] == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP (contradict detected): {tp}")
    print(f"  TN (consistent correct): {tn}")
    print(f"  FP (false contradict): {fp}")
    print(f"  FN (missed contradict): {fn}")
    print(f"\nTotal: {total}, Correct: {correct}")
    
    # Show errors for analysis
    print("\n" + "-" * 40)
    print("ERROR ANALYSIS (first 10)")
    print("-" * 40)
    errors = merged[merged['prediction'] != merged['label_binary']].head(10)
    for _, row in errors.iterrows():
        orig = train_df[train_df['id'] == row['id']].iloc[0]
        print(f"\nID {row['id']}: Predicted {row['prediction']}, Actual {row['label_binary']}")
        print(f"  Character: {orig['char']}")
        print(f"  Content: {orig['content'][:100]}...")
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / "smart_train_predictions.csv", index=False)
    
    # Process test data
    print("\n" + "=" * 60)
    print("TEST DATA PREDICTION")
    print("=" * 60)
    
    test_df = pd.read_csv(TEST_FILE)
    print(f"Loaded {len(test_df)} examples")
    
    test_results = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prediction, confidence, rationale = classifier.classify(
            backstory=row['content'],
            character_name=row['char'],
            book_name=row['book_name']
        )
        
        test_results.append({
            'Story ID': row['id'],
            'Prediction': prediction,
            'Rationale': rationale
        })
    
    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    
    print(f"\n[OK] Results saved to: {OUTPUT_DIR / 'results.csv'}")
    print(f"  Consistent (1): {(test_results_df['Prediction'] == 1).sum()}")
    print(f"  Contradict (0): {(test_results_df['Prediction'] == 0).sum()}")
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
