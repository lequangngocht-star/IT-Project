"""
src/ingestion/doccument_loader.py
---------------------------------
Module đọc tài liệu từ định dạng PDF bằng pdfplumber.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SUPPORTED_EXTENSIONS
from src.logger import get_logger

logger = get_logger(__name__)


def _read_pdf(file_path: Path) -> str:
    """Đọc PDF bằng pdfplumber — xử lý tốt Unicode tiếng Việt."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Vui lòng cài đặt pdfplumber: pip install pdfplumber")

    pages_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text)
    except Exception as e:
        logger.error(f"Lỗi đọc PDF {file_path.name}: {e}")
        return ""

    return "\n".join(pages_text)


def load_document(file_path: str | Path) -> dict:
    """Đóng gói văn bản trích xuất kèm metadata chuẩn hóa."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File không tồn tại: {path}")

    ext = path.suffix.lower()
    if ext != ".pdf":
        return {"source": str(path), "file_name": path.name, "file_type": ext, "raw_text": ""}

    logger.info(f"Đang trích xuất: {path.name}")
    raw_text = _read_pdf(path)
    
    return {
        "source":    str(path),
        "file_name": path.name,
        "file_type": ext,
        "raw_text":  raw_text,
    }


def load_directory(dir_path: str | Path) -> list[dict]:
    """Đọc toàn bộ tài liệu hợp lệ trong thư mục."""
    directory = Path(dir_path)
    docs = []
    for file in sorted(directory.iterdir()):
        if file.suffix.lower() != ".pdf":
            continue
        try:
            doc = load_document(file)
            if doc["raw_text"].strip():
                docs.append(doc)
        except Exception as e:
            logger.error(f"Bỏ qua file {file.name} do lỗi: {e}")

    logger.info(f"Tổng kết: Đã trích xuất xong {len(docs)} tài liệu PDF.")
    return docs