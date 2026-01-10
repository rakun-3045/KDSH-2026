"""
KDSH 2026 Track A - Enhanced High-Accuracy Pipeline
Runs the improved pipeline with multiple accuracy enhancements.

Usage:
    python run_enhanced.py --mode train    # Evaluate on training data
    python run_enhanced.py --mode predict  # Generate test predictions
    python run_enhanced.py --mode full     # Both
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, BOOKS_DIR, OUTPUT_DIR,
    TRAIN_FILE, TEST_FILE,
    EMBEDDING_MODEL, HUGGINGFACE_API_KEY
)
from src.enhanced_pipeline import EnhancedNarrativeConsistencyPipeline


def setup_environment():
    """Setup environment."""
    print("=" * 60)
    print("KDSH 2026 Track A - ENHANCED High-Accuracy Pipeline")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("Enhanced Features:")
    print("  - Multi-strategy evidence retrieval")
    print("  - Pattern-based contradiction detection")
    print("  - Keyword matching for common contradictions")
    print("  - Semantic similarity analysis")
    print("  - Ensemble classification")
    print()
    
    # Check API key
    if HUGGINGFACE_API_KEY:
        print("[OK] Hugging Face API key configured")
    else:
        print("[INFO] No HuggingFace API key - using pattern/keyword methods only")
    
    # Check files
    print("\nData files:")
    for name, path in [("Training", TRAIN_FILE), ("Test", TEST_FILE)]:
        status = "[OK]" if path.exists() else "[X]"
        print(f"  {status} {name}: {path}")
    
    print("\nBooks:")
    for book_file in BOOKS_DIR.glob("*.txt"):
        print(f"  [OK] {book_file.name}")
    
    print()


def run_training_evaluation(pipeline: EnhancedNarrativeConsistencyPipeline):
    """Evaluate on training data."""
    print("\n" + "=" * 60)
    print("TRAINING DATA EVALUATION")
    print("=" * 60)
    
    train_df = pd.read_csv(TRAIN_FILE)
    print(f"Loaded {len(train_df)} examples")
    
    output_path = OUTPUT_DIR / "enhanced_train_predictions.csv"
    results_df = pipeline.process_dataset(train_df, output_path=output_path)
    
    metrics = pipeline.evaluate(results_df, train_df)
    
    print("\n" + "-" * 40)
    print("EVALUATION METRICS")
    print("-" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP (correct contradict): {metrics['true_positives']}")
    print(f"  TN (correct consistent): {metrics['true_negatives']}")
    print(f"  FP (false contradict):   {metrics['false_positives']}")
    print(f"  FN (false consistent):   {metrics['false_negatives']}")
    print(f"\nTotal: {metrics['total']}, Correct: {metrics['correct']}")
    
    return metrics


def run_test_prediction(pipeline: EnhancedNarrativeConsistencyPipeline):
    """Generate test predictions."""
    print("\n" + "=" * 60)
    print("TEST DATA PREDICTION")
    print("=" * 60)
    
    test_df = pd.read_csv(TEST_FILE)
    print(f"Loaded {len(test_df)} examples")
    
    output_path = OUTPUT_DIR / "results.csv"
    results_df = pipeline.process_dataset(test_df, output_path=output_path)
    
    # Format for submission
    submission_df = results_df[['id', 'prediction', 'rationale']].copy()
    submission_df.columns = ['Story ID', 'Prediction', 'Rationale']
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n[OK] Submission file saved: {output_path}")
    print("\nPrediction Summary:")
    print(f"  - Total examples: {len(results_df)}")
    print(f"  - Consistent (1): {(results_df['prediction'] == 1).sum()}")
    print(f"  - Contradict (0): {(results_df['prediction'] == 0).sum()}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="KDSH 2026 Track A - Enhanced Pipeline"
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'predict', 'full'],
        default='full',
        help='Execution mode'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=25,
        help='Number of evidence chunks to retrieve'
    )
    
    args = parser.parse_args()
    
    setup_environment()
    
    # Initialize enhanced pipeline
    print("\nInitializing enhanced pipeline...")
    pipeline = EnhancedNarrativeConsistencyPipeline(
        books_dir=BOOKS_DIR,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=800,
        chunk_overlap=200,
        top_k_retrieval=args.top_k,
        hf_api_key=HUGGINGFACE_API_KEY
    )
    
    pipeline.initialize()
    
    # Execute
    if args.mode in ['train', 'full']:
        run_training_evaluation(pipeline)
    
    if args.mode in ['predict', 'full']:
        run_test_prediction(pipeline)
    
    print("\n" + "=" * 60)
    print("Enhanced pipeline execution complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
