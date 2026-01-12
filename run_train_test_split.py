"""
KDSH 2026 - Train/Test Split Evaluation
Splits train(1).csv into 80:20 ratio and evaluates accuracy on the test portion.
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from config import BOOKS_DIR, OUTPUT_DIR, TRAIN_FILE
from src.document_processor import NarrativeDocumentProcessor
from src.hybrid_classifier import HybridConsistencyClassifier


def main():
    print("=" * 60)
    print("KDSH 2026 - Train/Test Split Evaluation (80:20)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load full training data
    print("Loading data...")
    full_df = pd.read_csv(TRAIN_FILE)
    print(f"Total examples: {len(full_df)}")
    
    # Convert labels to binary
    full_df['label_binary'] = full_df['label'].apply(lambda x: 1 if x == 'consistent' else 0)
    
    # Split 80:20
    train_df, test_df = train_test_split(
        full_df, 
        test_size=0.20, 
        random_state=42,  # For reproducibility
        stratify=full_df['label_binary']  # Maintain class balance
    )
    
    print(f"\nSplit Results:")
    print(f"  Training set: {len(train_df)} examples")
    print(f"  Test set:     {len(test_df)} examples")
    
    # Show class distribution
    print(f"\nClass Distribution in Test Set:")
    print(f"  Consistent (1): {(test_df['label_binary'] == 1).sum()}")
    print(f"  Contradict (0): {(test_df['label_binary'] == 0).sum()}")
    
    # Initialize classifier
    print("\n" + "=" * 60)
    print("INITIALIZING CLASSIFIER")
    print("=" * 60)
    
    doc_processor = NarrativeDocumentProcessor(chunk_size=800, chunk_overlap=200)
    classifier = HybridConsistencyClassifier()
    
    # Load and index books
    book_files = {
        "In Search of the Castaways": BOOKS_DIR / "In search of the castaways.txt",
        "The Count of Monte Cristo": BOOKS_DIR / "The Count of Monte Cristo.txt"
    }
    
    for book_name, book_path in book_files.items():
        if book_path.exists():
            print(f"\nProcessing: {book_name}")
            text = doc_processor.load_book(book_path)
            chunks = doc_processor.chunk_text(text, book_name)
            classifier.index_book(book_name, chunks)
    
    # Evaluate on TEST SET (20%)
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET (20%)")
    print("=" * 60)
    
    results = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
        prediction, confidence, rationale = classifier.classify(
            backstory=row['content'],
            character_name=row['char'],
            book_name=row['book_name']
        )
        
        results.append({
            'id': row['id'],
            'actual': row['label_binary'],
            'prediction': prediction,
            'correct': prediction == row['label_binary'],
            'rationale': rationale
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    correct = results_df['correct'].sum()
    total = len(results_df)
    accuracy = correct / total
    
    tp = ((results_df['prediction'] == 0) & (results_df['actual'] == 0)).sum()
    fp = ((results_df['prediction'] == 0) & (results_df['actual'] == 1)).sum()
    fn = ((results_df['prediction'] == 1) & (results_df['actual'] == 0)).sum()
    tn = ((results_df['prediction'] == 1) & (results_df['actual'] == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS (20% of data)")
    print("=" * 60)
    
    print(f"\n{'='*40}")
    print(f"ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"{'='*40}")
    
    print(f"\nDetailed Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives  (Contradict detected correctly): {tp}")
    print(f"  True Negatives  (Consistent detected correctly): {tn}")
    print(f"  False Positives (Wrongly marked contradict):     {fp}")
    print(f"  False Negatives (Missed contradictions):         {fn}")
    
    print(f"\nTotal: {total} | Correct: {correct} | Wrong: {total - correct}")
    
    # Show all predictions
    print("\n" + "-" * 60)
    print("DETAILED TEST SET PREDICTIONS")
    print("-" * 60)
    
    for _, row in results_df.iterrows():
        status = "[OK]" if row['correct'] else "[X]"
        orig = test_df[test_df['id'] == row['id']].iloc[0]
        print(f"\n{status} ID {row['id']}: Pred={row['prediction']}, Actual={row['actual']}")
        print(f"   Character: {orig['char']}")
        print(f"   Book: {orig['book_name']}")
        print(f"   Rationale: {row['rationale'][:60]}...")
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / "train_test_split_results.csv", index=False)
    print(f"\n[OK] Results saved to: {OUTPUT_DIR / 'train_test_split_results.csv'}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Training Set: {len(train_df)} examples (80%)")
    print(f"Test Set:     {len(test_df)} examples (20%)")
    print(f"Test Accuracy: {accuracy*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
