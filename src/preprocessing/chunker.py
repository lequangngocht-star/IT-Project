import os
import re
import json
from pathlib import Path
from typing import List, Dict, Iterator

CHUNK_SIZE = 600       # Độ dài ký tự lý tưởng cho 1 chunk
CHUNK_OVERLAP = 100    # Độ gối đầu giữa 2 chunk để tránh đứt đoạn ngữ cảnh
MIN_CHUNK_LEN = 100    # Loại bỏ các đoạn quá ngắn rác

def split_into_sentences(text: str) -> List[str]:
    """Tách đoạn thành các câu hoàn chỉnh dựa trên dấu câu."""
    return re.split(r"(?<=[.!?])\s+", text)

def iter_paragraph_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> Iterator[str]:
    """Cắt nhỏ văn bản ưu tiên giữ nguyên khối đoạn văn và câu."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if buffer:
                yield buffer
                buffer = ""
            sentences = split_into_sentences(para)
            for sent in sentences:
                if len(buffer) + len(sent) + 1 <= chunk_size:
                    buffer = (buffer + " " + sent).strip() if buffer else sent
                else:
                    if buffer:
                        yield buffer
                    buffer = sent
        else:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    yield buffer
                buffer = para

    if buffer:
        yield buffer

def process_file_to_chunks(file_path: Path) -> List[Dict]:
    """Đọc 1 file txt, bóc metadata ở đầu file và chia thành các chunk chuẩn."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Bóc metadata ở phần header đã ghi lúc crawl
    subject = "IT"
    title = file_path.stem
    source = "Wikipedia Academic"

    header_match = re.match(r"^Subject:\s*(.*?)\n(?:Topic|Title):\s*(.*?)\nSource:\s*(.*?)\n\n", content)
    if header_match:
        subject = header_match.group(1).strip()
        title = header_match.group(2).strip()
        source = header_match.group(3).strip()
        body_text = content[header_match.end():].strip()
    else:
        body_text = content.strip()

    raw_chunks = list(iter_paragraph_chunks(body_text, CHUNK_SIZE))
    chunks = []
    tail = ""

    for i, chunk_text in enumerate(raw_chunks):
        if tail:
            chunk_text = (tail + " " + chunk_text).strip()
            if len(chunk_text) > CHUNK_SIZE:
                chunk_text = chunk_text[:CHUNK_SIZE]

        if len(chunk_text) < MIN_CHUNK_LEN:
            continue

        chunk_id = f"{subject}_{file_path.stem}_c{i:04d}"
        chunks.append({
            "chunk_id": chunk_id,
            "chunk_index": i,
            "subject": subject,
            "title": title,
            "source": source,
            "file_name": file_path.name,
            "char_count": len(chunk_text),
            "text": chunk_text
        })
        tail = chunk_text[-CHUNK_OVERLAP:] if CHUNK_OVERLAP > 0 else ""

    return chunks

def save_chunks(chunks: List[Dict], output_file: str = "data/processed/chunks.json"):
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[✓] Đã lưu thành công {len(chunks)} chunks vào: {out_path.resolve()}")

def load_chunks(input_file: str = "data/processed/chunks.json") -> List[Dict]:
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)