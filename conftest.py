"""Configure pytest to find src packages."""
import sys
from pathlib import Path

# Add src/ to the Python path so `from src.tglue import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
