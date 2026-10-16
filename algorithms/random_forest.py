"""Standalone entry point for the shared train/validation/test evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import run_standalone

if __name__ == "__main__":
    run_standalone("Random Forest")
