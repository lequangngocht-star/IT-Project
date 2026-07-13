"""
src/pipeline_m1.py
------------------
Luồng điều phối trung tâm xử lý dữ liệu hàng loạt của Milestone 1.
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.logger import get_logger
from src.ingestion.document_loader import load_directory
from src.ingestion.text_cleaner import clean_document
from src.preprocessing.chunker import create_chunks_with_metadata

logger = get_logger(__name__)


def run_milestone1_pipeline():
    start_all = time.time()
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU CHẠY PIPELINE TRÍCH XUẤT HÀNG LOẠT MILESTONE 1")
    logger.info("=" * 60)

    # 1. Đọc hàng loạt dữ liệu PDF từ kho raw
    raw_docs = load_directory(DATA_RAW_DIR)
    if not raw_docs:
        logger.error("Không tìm thấy dữ liệu thô trong data/raw/!")
        return

    # 2. Làm sạch văn bản chuyên sâu
    logger.info("Đang xử lý chuẩn hóa và làm sạch văn bản...")
    cleaned_docs = [clean_document(doc) for doc in raw_docs]

    # 3. Chunking Paragraph-Aware
    logger.info("Đang thực hiện phân rã văn bản thành các chunks...")
    all_chunks = []
    for doc in cleaned_docs:
        chunks = create_chunks_with_metadata(doc)
        all_chunks.extend(chunks)

    # 4. Xuất kết quả
    output_json_path = DATA_PROCESSED_DIR / "chunks.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"HOÀN THÀNH MILESTONE 1 TRONG {time.time() - start_all:.2f}s")
    logger.info(f"Tổng số chunks sinh ra: {len(all_chunks)}")
    logger.info(f"Tệp tin kết quả xuất thành công: {output_json_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_milestone1_pipeline()