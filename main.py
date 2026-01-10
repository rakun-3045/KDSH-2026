"""
KDSH 2026 Track A Solution - Main Entry Point
Narrative Consistency Classification using Pathway

This script orchestrates the complete pipeline for:
1. Loading and indexing narrative documents
2. Processing test examples
3. Generating predictions for submission

Usage:
    python main.py --mode train    # Evaluate on training data
    python main.py --mode predict  # Generate predictions for test data
    python main.py --mode full     # Full pipeline with evaluation
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, BOOKS_DIR, OUTPUT_DIR,
    TRAIN_FILE, TEST_FILE,
    EMBEDDING_MODEL, LLM_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL
)
from src.pipeline import NarrativeConsistencyPipeline


def setup_environment():
    """Setup environment and check dependencies."""
    print("=" * 60)
    print("KDSH 2026 Track A - Narrative Consistency Classification")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check for HuggingFace API key
    from config import HUGGINGFACE_API_KEY
    if HUGGINGFACE_API_KEY:
        print("[OK] Hugging Face API key configured")
        print(f"  Using model: {LLM_MODEL}")
    else:
        print("Warning: HUGGINGFACE_API_KEY not set.")
        print("Set the API key for LLM-powered reasoning:")
        print("  $env:HUGGINGFACE_API_KEY='your-hf-key'  (PowerShell)")
        print("  export HUGGINGFACE_API_KEY='your-hf-key' (Linux/Mac)")
        print("The pipeline will use rule-based fallback only.")
        print()
    
    # Check data files
    print("\nChecking data files...")
    for name, path in [("Training data", TRAIN_FILE), ("Test data", TEST_FILE)]:
        if path.exists():
            print(f"  [OK] {name}: {path}")
        else:
            print(f"  [X] {name} NOT FOUND: {path}")
    
    # Check book files
    print("\nChecking book files...")
    for book_file in BOOKS_DIR.glob("*.txt"):
        print(f"  [OK] {book_file.name}")
    
    print()


def load_data(file_path: Path) -> pd.DataFrame:
    """Load dataset from CSV file."""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} examples from {file_path.name}")
    return df


def run_training_evaluation(pipeline: NarrativeConsistencyPipeline):
    """Run evaluation on training data."""
    print("\n" + "=" * 60)
    print("TRAINING DATA EVALUATION")
    print("=" * 60)
    
    train_df = load_data(TRAIN_FILE)
    
    # Process training data
    output_path = OUTPUT_DIR / "train_predictions.csv"
    results_df = pipeline.process_dataset(train_df, output_path=output_path)
    
    # Evaluate
    metrics = pipeline.evaluate(results_df, train_df)
    
    print("\n" + "-" * 40)
    print("EVALUATION METRICS")
    print("-" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Total:     {metrics['total_examples']}")
    print(f"Correct:   {metrics['correct_predictions']}")
    
    return metrics


def run_test_prediction(pipeline: NarrativeConsistencyPipeline):
    """Generate predictions for test data."""
    print("\n" + "=" * 60)
    print("TEST DATA PREDICTION")
    print("=" * 60)
    
    test_df = load_data(TEST_FILE)
    
    # Process test data
    output_path = OUTPUT_DIR / "results.csv"
    results_df = pipeline.process_dataset(test_df, output_path=output_path)
    
    # Format for submission
    submission_df = results_df[['id', 'prediction', 'rationale']].copy()
    submission_df.columns = ['Story ID', 'Prediction', 'Rationale']
    
    submission_path = OUTPUT_DIR / "results.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n[OK] Submission file saved: {submission_path}")
    
    # Summary
    print("\nPrediction Summary:")
    print(f"  - Total examples: {len(results_df)}")
    print(f"  - Consistent (1): {(results_df['prediction'] == 1).sum()}")
    print(f"  - Contradict (0): {(results_df['prediction'] == 0).sum()}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="KDSH 2026 Track A - Narrative Consistency Classification"
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'predict', 'full'],
        default='full',
        help='Execution mode: train (evaluate), predict (test), or full (both)'
    )
    parser.add_argument(
        '--llm-model',
        default=LLM_MODEL,
        help=f'LLM model to use (default: {LLM_MODEL})'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=TOP_K_RETRIEVAL,
        help=f'Number of chunks to retrieve (default: {TOP_K_RETRIEVAL})'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_environment()
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = NarrativeConsistencyPipeline(
        books_dir=BOOKS_DIR,
        embedding_model=EMBEDDING_MODEL,
        llm_model=args.llm_model,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k_retrieval=args.top_k
    )
    
    # Initialize (load and index books)
    pipeline.initialize()
    
    # Execute based on mode
    if args.mode in ['train', 'full']:
        run_training_evaluation(pipeline)
    
    if args.mode in ['predict', 'full']:
        run_test_prediction(pipeline)
    
    print("\n" + "=" * 60)
    print("Pipeline execution complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
