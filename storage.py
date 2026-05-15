import io
from pathlib import Path
from typing import List

import PyPDF2

UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


def create_uploads_directory() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file_type(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def extract_pdf_text(uploaded_file) -> str:
    uploaded_file.seek(0)
    print(f"Extracting text from PDF file: {uploaded_file.name}")  # Debug log for PDF extraction
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getbuffer()))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    print(f"Extracted text from PDF: {pages[0][:500]}")  # Debug log for extracted PDF text
    return "\n".join(pages).strip()


def extract_uploaded_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    print(f"Extracting text from file: {uploaded_file.name} with suffix: {suffix}")  # Debug log for file type
    if suffix == ".pdf":
        print("Detected PDF file. Extracting text using PyPDF2.")  # Debug log for PDF extraction
        return extract_pdf_text(uploaded_file)
    uploaded_file.seek(0)
    return uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()


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


def save_to_permanent_storage(filename: str, content: str) -> None:
    """Save file content to permanent storage (vector DB)."""
    # This will be handled by vector_db.py
    pass


def load_from_permanent_storage(filename: str) -> str:
    """Load file content from permanent storage."""
    # This will be handled by vector_db.py
    return ""
