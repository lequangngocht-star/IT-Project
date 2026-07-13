"""
src/preprocessing/chunker.py
------------------------------
Phân rã cấu trúc văn bản thành khối ngữ cảnh chunks có overlap.
Quản lý lưu trữ và nạp chỉ mục JSON cho toàn bộ hệ thống.
"""

import sys
import re
import json
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN, DATA_PROCESSED_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def _split_into_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def _iter_paragraph_chunks(text: str, chunk_size: int) -> Iterator[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if buffer:
                yield buffer
                buffer = ""
            sentences = _split_into_sentences(para)
            for sent in sentences:
                if len(buffer) + len(sent) + 1 <= chunk_size:
                    buffer = (buffer + " " + sent).strip() if buffer else sent
                else:
                    if buffer: yield buffer
                    buffer = sent
        else:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer: yield buffer
                buffer = para
    if buffer: yield buffer


def create_chunks_with_metadata(doc: dict) -> list[dict]:
    text = doc.get("cleaned_text", "") or doc.get("raw_text", "")
    if not text.strip(): return []

    raw_chunks = list(_iter_paragraph_chunks(text, CHUNK_SIZE))
    result = []
    tail = ""

    for i, chunk_text in enumerate(raw_chunks):
        if tail:
            chunk_text = (tail + " " + chunk_text).strip()
            if len(chunk_text) > CHUNK_SIZE:
                chunk_text = chunk_text[:CHUNK_SIZE]

        if len(chunk_text) < MIN_CHUNK_LEN:
            continue

        chunk_id = f"{Path(doc['source']).stem}_chunk_{i:04d}"
        result.append({
            "chunk_id":    chunk_id,
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "text":        chunk_text,
            "char_count":  len(chunk_text),
            "source":      doc["source"],
            "file_name":   doc.get("file_name", ""),
            "file_type":   doc.get("file_type", ""),
            "num_pages":   doc.get("num_pages")
        })
        tail = chunk_text[-CHUNK_OVERLAP:] if CHUNK_OVERLAP > 0 else ""

    return result


# ─────────────────────────────────────────────────────────────
# PHÂN HỆ LƯU / TẢI CHUNKS HỆ THỐNG
# ─────────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict], output_path: str | Path | None = None) -> Path:
    """
    Lưu danh sách chunks kèm metadata vào file JSON để tái sử dụng.
    """
    if output_path is None:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DATA_PROCESSED_DIR / "chunks.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Đã lưu thành công {len(chunks)} chunks vào -> {output_path}")
    return output_path


def load_chunks(input_path: str | Path) -> list[dict]:
    """
    Tải danh sách chunks từ file JSON chỉ mục tri thức.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu chunks chỉ mục: {path}")

    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Đã tải thành công {len(chunks)} chunks từ hệ thống -> {path}")
    return chunks