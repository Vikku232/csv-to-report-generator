import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile
from backend.core.config import UPLOAD_DIR, OUTPUT_DIR

def save_uploaded_file(upload_file: UploadFile) -> Path:
    """Saves an uploaded file to storage/uploads with a unique name."""
    ext = Path(upload_file.filename).suffix or ".csv"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return dest


def new_output_path(filename: str = "report.pdf") -> Path:
    """Generates a unique destination path in storage/outputs."""
    return OUTPUT_DIR / f"{uuid.uuid4().hex}_{filename}"


def cleanup_file(path: Path):
    """Safely removes an ephemeral file from disk."""
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass