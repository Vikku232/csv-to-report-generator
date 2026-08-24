from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
OUTPUT_DIR = BASE_DIR / "storage" / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 50
ID_LIKE_UNIQUE_RATIO = 1.0   # Column is ID-like if nunique/len == 1.0
CATEGORICAL_UNIQUE_THRESHOLD = 20  # Numeric column with <= 20 uniques is treated as categorical
RANDOM_STATE = 42



