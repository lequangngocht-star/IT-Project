"""
src/pipeline_m5.py
-------------------
Entry-point chạy Milestone 5: System Evaluation Pipeline.
Tự động tính toán các chỉ số định lượng và xuất báo cáo kết quả.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Thêm root vào sys.path để import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BASE_DIR,
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    LOG_DIR,
    RETRIEVAL_TOP_K,
    RERANKER_TOP_N,
)
from src.logger import get_logger
from src.pipeline_m3 import build_retrieval_pipeline, run_query
from src.evaluation.evaluator import RAGEvaluator

logger = get_logger(__name__)

EVAL_DATASET_PATH = BASE_DIR / "tests" / "eval_dataset.json"
EVAL_OUTPUT_PATH = LOG_DIR / "evaluation_results.json"


def run_milestone5_pipeline(
    use_reranker: bool = True,
    dataset_path: Path = EVAL_DATASET_PATH,
    output_path: Path = EVAL_OUTPUT_PATH,
) -> dict:
    """
    Thực thi pipeline kiểm thử tự động toàn hệ thống.
    """
    logger.info("═" * 60)
    logger.info("🚀 KHỞI ĐỘNG MILESTONE 5: SYSTEM EVALUATION PIPELINE")
    logger.info("═" * 60)

    # 1. Kiểm tra điều kiện tiên quyết
    if not dataset_path.exists():
        logger.error(f"Không tìm thấy file dataset kiểm thử tại: {dataset_path}")
        sys.exit(1)

    if not FAISS_INDEX_FILE.exists() or not FAISS_META_FILE.exists():
        logger.error("Chưa build FAISS index! Hãy chạy 'python src/pipeline_m2.py' trước.")
        sys.exit(1)

    # 2. Nạp dataset kiểm thử
    with open(dataset_path, encoding="utf-8") as f:
        testset = json.load(f)
    logger.info(f"Đã tải {len(testset)} câu hỏi mẫu từ {dataset_path.name}")

    # 3. Khởi tạo Pipeline M3 (Retrieval + Rerank)
    retriever, reranker = build_retrieval_pipeline(use_reranker=use_reranker)

    # 4. Định nghĩa runner callback function
    # Định nghĩa runner callback function
    def query_runner(query_text: str):
        return run_query(
            query=query_text,
            retriever=retriever,
            reranker=reranker if use_reranker else None,
            top_k=RETRIEVAL_TOP_K,
            top_n=RERANKER_TOP_N
        )

    # 5. Khởi tạo và thực thi Evaluator
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_dataset(testset, query_runner)
    results["configuration"] = {
        "use_reranker": use_reranker,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "reranker_top_n": RERANKER_TOP_N,
    }

    # 6. Xuất kết quả báo cáo
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 7. In bảng tổng hợp trực quan ra Terminal
    print("\n" + "═" * 60)
    print("📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG (MILESTONE 5)")
    print("═" * 60)
    print(f" Số lượng câu hỏi test  : {results['total_queries']}")
    print(f" Cấu hình Reranker      : {'BẬT (Cross-Encoder)' if use_reranker else 'TẮT (Dense Only)'}")
    print(f" Thời gian phản hồi trung bình (Latency) : {results['average_latency_ms']} ms")
    print(f" Chỉ số MRR (Mean Reciprocal Rank)      : {results['mrr']}")
    print(f" Chỉ số Keyword Recall                  : {results['keyword_recall']}")
    print(" Tỉ lệ Hit Rate:")
    for k_name, val in results["hit_rates"].items():
        print(f"   • {k_name.upper()}: {val * 100:.1f}%")
    print("═" * 60)
    print(f"✅ Báo cáo chi tiết đã lưu tại: {output_path}\n")

    return results


if __name__ == "__main__":
    run_milestone5_pipeline()