import os
from pathlib import Path
from typing import Dict, List

UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".txt", ".md"}


def create_uploads_directory() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file_type(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(uploaded_file) -> str:
    create_uploads_directory()
    filename = Path(uploaded_file.name).name
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as destination:
        destination.write(uploaded_file.getbuffer())
    return filename


def delete_uploaded_file(filename: str) -> bool:
    file_path = UPLOAD_DIR / Path(filename).name
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def load_text_content(filename: str) -> str:
    file_path = UPLOAD_DIR / Path(filename).name
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8", errors="ignore")


def list_uploaded_files() -> List[str]:
    create_uploads_directory()
    return [item.name for item in UPLOAD_DIR.iterdir() if item.is_file()]
