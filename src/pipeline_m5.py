"""
src/pipeline_m5.py
-------------------
Entry-point Milestone 5: Evaluation Pipeline.

Mục tiêu của M5:
- Đánh giá định lượng chất lượng toàn bộ hệ thống RAG.
- So sánh các cấu hình: Dense only vs Dense+Rerank, các LLM khác nhau.
- Tạo báo cáo lưu vào logs/evaluation/ để dùng trong báo cáo học thuật.

Cách dùng:
    # Đánh giá full pipeline (cần model thật)
    python src/pipeline_m5.py

    # Chỉ đánh giá retrieval (không cần LLM)
    python src/pipeline_m5.py --retrieval-only

    # Chạy ablation study: so sánh có/không reranker
    python src/pipeline_m5.py --ablation

    # Chỉ chạy N samples đầu (debug nhanh)
    python src/pipeline_m5.py --n-samples 5

    # Chỉ chạy 1 category
    python src/pipeline_m5.py --category de_cuong

Prerequisite:
    - pipeline_m1.py → data/processed/chunks.json       ✓
    - pipeline_m2.py → vector_store/faiss.index          ✓
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVAL_TOP_K_VALUES,
    FAISS_INDEX_FILE,
    LLM_MODEL_NAME,
    RERANKER_ENABLED,
    RETRIEVAL_TOP_K,
    RERANKER_TOP_N,
    BASE_DIR,
    LOG_DIR,
)

# Thử import từ config, nếu chưa có sẽ tự động tạo đường dẫn chuẩn
try:
    from config import EVAL_DATASET_PATH, EVAL_RESULTS_DIR
except ImportError:
    EVAL_DATASET_PATH = BASE_DIR / "tests" / "eval_dataset.json"
    EVAL_RESULTS_DIR  = LOG_DIR / "evaluation"
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
from src.logger import get_logger
from src.evaluation.evaluator import RAGEvaluator, EvalReport

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# RAG query function builder
# ─────────────────────────────────────────────────────────────

def _build_rag_fn(
    use_reranker: bool = RERANKER_ENABLED,
    top_k:        int  = RETRIEVAL_TOP_K,
    top_n:        int  = RERANKER_TOP_N,
    load_llm:     bool = True,
):
    """
    Tạo hàm rag_query_fn(query: str) → RAGResponse để truyền vào RAGEvaluator.

    Tách riêng hàm này để ablation study dễ dàng:
    - Thay use_reranker=False để so sánh retrieval chỉ dùng dense.
    - Thay load_llm=False để chỉ đánh giá retrieval metrics.

    Returns:
        callable: hàm (query: str) → RAGResponse
    """
    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index không tồn tại: {FAISS_INDEX_FILE}\n"
            "  → Chạy M1+M2 trước: python src/pipeline_m1.py && python src/pipeline_m2.py"
        )

    from src.pipeline_m3 import build_retrieval_pipeline, run_query as retrieve
    retriever, reranker = build_retrieval_pipeline(
        use_reranker=use_reranker,
        top_k=top_k,
        top_n=top_n,
    )

    if load_llm:
        from src.pipeline_m4 import rag_query
        from src.LLM.model_manager import LLMManager
        from src.LLM.prompter import build_messages, parse_response

        llm = LLMManager()
        llm.load()

        def rag_fn(query: str):
            return rag_query(query, retriever, reranker, llm)
    else:
        # Retrieval-only mode: trả về mock RAGResponse chỉ có sources
        from src.LLM.prompter import RAGResponse

        def rag_fn(query: str):
            chunks = retrieve(query, retriever, reranker)
            return RAGResponse(
                query=query,
                answer="[RETRIEVAL-ONLY MODE]",
                sources=chunks,
                has_answer=False,
                context_used=len(chunks),
                latency_s=0.0,
            )

    return rag_fn


# ─────────────────────────────────────────────────────────────
# Ablation study
# ─────────────────────────────────────────────────────────────

def run_ablation_study(n_samples: int | None = None) -> dict:
    """
    So sánh 2 cấu hình: Dense Only vs Dense + Rerank.

    Đây là thực nghiệm quan trọng để chứng minh giá trị của reranker
    trong báo cáo học thuật.

    Args:
        n_samples: Giới hạn số sample (None = tất cả).

    Returns:
        Dict chứa 2 EvalReport để so sánh.
    """
    logger.info("═" * 60)
    logger.info("  ABLATION STUDY: Dense Only vs Dense + Rerank")
    logger.info("═" * 60)

    results = {}
    configs = [
        ("dense_only",         False),
        ("dense_plus_rerank",  True),
    ]

    sample_ids = None
    if n_samples is not None:
        # Lấy N samples đầu từ dataset
        with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
            all_samples = json.load(f)
        sample_ids = [s["id"] for s in all_samples[:n_samples]]

    for config_name, use_reranker in configs:
        logger.info(f"\n[Config: {config_name}]")
        rag_fn   = _build_rag_fn(use_reranker=use_reranker, load_llm=False)
        evaluator = RAGEvaluator(rag_fn)
        report    = evaluator.run(sample_ids=sample_ids, save=True)
        results[config_name] = report
        RAGEvaluator.print_report(report)

    # In bảng so sánh
    _print_ablation_comparison(results)
    return results


def _print_ablation_comparison(results: dict) -> None:
    """In bảng so sánh 2 cấu hình side-by-side."""
    configs = list(results.keys())
    if len(configs) < 2:
        return

    w = 70
    print("\n" + "═" * w)
    print(f"{'  ABLATION COMPARISON':^{w}}")
    print("═" * w)
    print(f"  {'Metric':<30} {'dense_only':>15} {'dense+rerank':>15}")
    print("─" * w)

    metrics = [
        ("Context Recall",    "mean_context_recall"),
        ("Context Precision", "mean_context_precision"),
        ("Recall@1",          "recall@1"),
        ("Recall@3",          "recall@3"),
        ("Recall@5",          "recall@5"),
        ("Mean Latency (s)",  "mean_latency_s"),
    ]

    for label, key in metrics:
        vals = []
        for config in configs:
            report = results[config]
            d = report.to_dict()
            if key.startswith("recall@"):
                v = d.get("mean_recall_at_k", {}).get(key, 0.0)
            else:
                v = d.get(key, 0.0)
            vals.append(v)

        # Highlight cột cao hơn (trừ latency)
        better_idx = 0 if vals[0] >= vals[1] else 1
        if key == "mean_latency_s":
            better_idx = 0 if vals[0] <= vals[1] else 1

        row = f"  {label:<30}"
        for i, v in enumerate(vals):
            marker = " ✓" if i == better_idx else "  "
            row += f" {v:>13.4f}{marker}"
        print(row)

    print("═" * w + "\n")


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def run_evaluation(
    retrieval_only: bool = False,
    use_reranker:   bool = RERANKER_ENABLED,
    n_samples:      int | None = None,
    category:       str | None = None,
) -> EvalReport:
    """
    Chạy evaluation đầy đủ và in báo cáo.

    Args:
        retrieval_only: Chỉ đánh giá retrieval, không gọi LLM.
        use_reranker:   Có dùng Cross-Encoder reranker không.
        n_samples:      Giới hạn số sample.
        category:       Lọc theo category (de_cuong, giao_trinh, quy_dinh).

    Returns:
        EvalReport đã tổng hợp.
    """
    rag_fn = _build_rag_fn(use_reranker=use_reranker, load_llm=not retrieval_only)
    evaluator = RAGEvaluator(rag_fn)

    # Filter sample_ids
    sample_ids = None
    if n_samples is not None or category is not None:
        with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
            all_raw = json.load(f)

        filtered = all_raw
        if category:
            filtered = [s for s in filtered if s.get("category") == category]
        if n_samples:
            filtered = filtered[:n_samples]

        sample_ids = [s["id"] for s in filtered]
        logger.info(f"Filtered: {len(sample_ids)} samples (category={category})")

    report = evaluator.run(sample_ids=sample_ids, save=True)
    RAGEvaluator.print_report(report)
    return report


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline Milestone 5: Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python src/pipeline_m5.py                        # Full evaluation
  python src/pipeline_m5.py --retrieval-only        # Chỉ đánh giá retrieval
  python src/pipeline_m5.py --ablation              # So sánh dense vs rerank
  python src/pipeline_m5.py --n-samples 10          # Chạy 10 samples đầu
  python src/pipeline_m5.py --category de_cuong     # Chỉ đánh giá 1 category
        """,
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Chỉ đánh giá retrieval (không cần LLM)"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Chạy ablation study: dense only vs dense+rerank"
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Tắt Cross-Encoder reranker"
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Giới hạn số sample (debug)"
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=["de_cuong", "giao_trinh", "quy_dinh"],
        help="Lọc theo category"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.ablation:
        run_ablation_study(n_samples=args.n_samples)
    else:
        run_evaluation(
            retrieval_only = args.retrieval_only,
            use_reranker   = not args.no_rerank,
            n_samples      = args.n_samples,
            category       = args.category,
        )
