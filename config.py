"""
Configuration file for KDSH 2026 - Track A Solution
Narrative Consistency Classification Pipeline
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "Dataset"
BOOKS_DIR = DATA_DIR / "Books"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset files
TRAIN_FILE = DATA_DIR / "train (1).csv"
TEST_FILE = DATA_DIR / "test (1).csv"

# Book files
BOOKS = {
    "In Search of the Castaways": BOOKS_DIR / "In search of the castaways.txt",
    "The Count of Monte Cristo": BOOKS_DIR / "The Count of Monte Cristo.txt"
}

# Embedding model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Chunking configuration for long documents
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters

# Retrieval configuration
TOP_K_RETRIEVAL = 15  # Number of chunks to retrieve per query
SIMILARITY_THRESHOLD = 0.3

# LLM configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # Default model
LLM_TEMPERATURE = 0.1  # Low temperature for consistent reasoning
MAX_TOKENS = 2000

# API Keys (set via environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Pathway Vector Store configuration
VECTOR_STORE_PATH = OUTPUT_DIR / "vector_store"
