"""
src/ingestion/text_cleaner.py
------------------------------
Làm sạch văn bản, chuẩn hóa Unicode và loại bỏ nhiễu PDF artifacts.
"""

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.logger import get_logger

logger = get_logger(__name__)


def clean_document(doc: dict) -> dict:
    """Thực thi pipeline làm sạch văn bản."""
    text = doc.get("raw_text", "")
    if not text:
        return {**doc, "cleaned_text": ""}
        
    # 1. Chuẩn hóa cấu trúc Unicode về NFC
    text = unicodedata.normalize("NFC", text)
    # 2. Gộp từ bị ngắt dòng bởi dấu gạch ngang
    text = re.sub(r"-\n", "", text)
    # 3. Thu gọn khoảng trắng và tab lẫn lộn
    text = re.sub(r"[ \t\u00a0\u200b\ufeff]+", " ", text)
    
    # 4. Xóa bỏ số trang cô lập đứng riêng dòng
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"[-–—\s]*\d{1,4}[-–—\s]*", stripped) or re.fullmatch(r"[Tt]rang\s+\d{1,4}", stripped):
            continue
        if re.fullmatch(r"[=\-–—_*•·.…\s]{3,}", stripped):
            continue
        cleaned_lines.append(line.strip())
        
    text = "\n".join(cleaned_lines)
    # 5. Thu gọn dòng trống thừa
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return {**doc, "cleaned_text": text.strip()}