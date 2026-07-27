"""
src/pipeline_m3.py
-------------------
Entry-point chạy Milestone 3: Online Retrieval Pipeline.

Milestone 3 là bước "Online" — chạy mỗi khi user đặt câu hỏi,
khác với M1+M2 là "Offline" (chỉ chạy 1 lần khi indexing).

Pipeline M3 (mỗi query):
    User query (str)
        → DenseRetriever.retrieve()     # embed + FAISS search → top-10
        → CrossEncoderReranker.rerank() # cross-encoder → top-5 chính xác
        → format_retrieval_result()     # chuẩn hóa schema
        → Hiển thị kết quả + trích dẫn nguồn

Cách dùng:
    # Interactive mode (hỏi nhiều câu)
    python src/pipeline_m3.py

    # Single query mode
    python src/pipeline_m3.py --query "Điều kiện tiên quyết môn ML là gì?"

    # Không dùng reranker (ablation study)
    python src/pipeline_m3.py --no-rerank

    # Tuỳ chỉnh số kết quả
    python src/pipeline_m3.py --top-k 10 --top-n 3

Prerequisite:
    - Đã chạy pipeline_m1.py → data/processed/chunks.json
    - Đã chạy pipeline_m2.py → vector_store/faiss.index + faiss_meta.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    RETRIEVAL_TOP_K,
    RERANKER_TOP_N,
    RERANKER_ENABLED,
)
from src.logger import get_logger
from src.embedding.embedder import EmbeddingModel
from vector_store.faiss_store import FaissVectorStore
from src.retrieval.searcher import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker, format_retrieval_result

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────

def _print_results(query: str, results: list[dict], elapsed: float) -> None:
    """In kết quả retrieval ra console theo format dễ đọc."""
    print(f"\n{'═' * 65}")
    print(f"  🔍 QUERY: {query}")
    print(f"  ⏱  Thời gian: {elapsed:.3f}s | Kết quả: {len(results)} chunks")
    print(f"{'═' * 65}")

    if not results:
        print("  ⚠️  Không tìm thấy kết quả phù hợp.")
        return

    for r in results:
        print(f"\n  ┌─ #{r['rank']} | {r['file_name']}")
        print(f"  │  Dense score  : {r['score']:.4f}")
        print(f"  │  Rerank score : {r['rerank_score']:.4f}")
        print(f"  │  Chunk ID     : {r['chunk_id']}")
        # Hiển thị 200 ký tự đầu, thay newline bằng space cho gọn
        preview = r["text"][:220].replace("\n", " ").strip()
        print(f"  └─ 📄 {preview}...")

    print(f"\n{'─' * 65}")
    print("  📚 NGUỒN TÀI LIỆU:")
    seen = set()
    for r in results:
        src = r["file_name"]
        if src not in seen:
            print(f"     • {src} (chunk #{r['chunk_index']})")
            seen.add(src)
    print(f"{'─' * 65}\n")


# ─────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────

def build_retrieval_pipeline(
    use_reranker: bool = RERANKER_ENABLED,
    top_k:        int  = RETRIEVAL_TOP_K,
    top_n:        int  = RERANKER_TOP_N,
) -> tuple[DenseRetriever, CrossEncoderReranker]:
    """
    Khởi tạo toàn bộ Retrieval Pipeline (gọi 1 lần, dùng nhiều query).

    Tách riêng hàm này để:
    - Milestone 4 (LLM) import và tái dùng pipeline mà không cần khởi tạo lại.
    - Test dễ hơn (mock từng thành phần riêng lẻ).

    Args:
        use_reranker: Có dùng Cross-Encoder reranker không.
        top_k:        Số ứng viên lấy từ FAISS.
        top_n:        Số chunk giữ lại sau rerank.

    Returns:
        Tuple (DenseRetriever, CrossEncoderReranker).

    Raises:
        FileNotFoundError: Nếu FAISS index chưa được build (chạy M2 trước).
    """
    # Kiểm tra prerequisite
    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index không tồn tại: {FAISS_INDEX_FILE}\n"
            "  → Chạy M2 trước: python src/pipeline_m2.py"
        )

    logger.info("═" * 55)
    logger.info("  KHỞI TẠO RETRIEVAL PIPELINE — Milestone 3")
    logger.info("═" * 55)

    # Load embedding model + FAISS store
    logger.info("[1/3] Load Embedding Model...")
    embedder = EmbeddingModel()

    logger.info("[2/3] Load FAISS Index...")
    vector_store = FaissVectorStore.load(FAISS_INDEX_FILE, FAISS_META_FILE)

    # Dense Retriever
    retriever = DenseRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=top_k,
    )

    # Cross-Encoder Reranker
    logger.info(f"[3/3] Load Reranker (enabled={use_reranker})...")
    reranker = CrossEncoderReranker(
        enabled=use_reranker,
        top_n=top_n,
    )

    logger.info("Retrieval Pipeline sẵn sàng ✓")
    logger.info(f"  Dense top_k  : {top_k}")
    logger.info(f"  Rerank top_n : {top_n}")
    logger.info(f"  Reranker     : {'ON' if use_reranker else 'OFF'}")
    logger.info("═" * 55)

    return retriever, reranker


def run_query(
    query:     str,
    retriever: DenseRetriever,
    reranker:  CrossEncoderReranker,
    top_n:     int | None = None,
) -> list[dict]:
    """
    Chạy 1 query qua full retrieval pipeline.

    Hàm này được export để Milestone 4 (LLM) gọi trực tiếp.

    Args:
        query:     Câu hỏi người dùng.
        retriever: DenseRetriever đã khởi tạo.
        reranker:  CrossEncoderReranker đã khởi tạo.
        top_n:     Override số chunk trả về cuối cùng.

    Returns:
        List[dict] chunk đã rerank và chuẩn hóa schema.
    """
    t0 = time.time()

    # Bước 1: Dense retrieval
    candidates = retriever.retrieve(query)

    # Bước 2: Rerank
    reranked = reranker.rerank(query, candidates, top_n=top_n)

    # Bước 3: Chuẩn hóa output schema
    results = format_retrieval_result(reranked)

    elapsed = time.time() - t0
    logger.info(f"Query hoàn tất | {elapsed:.3f}s | {len(results)} chunks")

    return results


# ─────────────────────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────────────────────

def _interactive_mode(
    retriever: DenseRetriever,
    reranker:  CrossEncoderReranker,
) -> None:
    """Vòng lặp hỏi-đáp tương tác trên terminal."""
    print("\n" + "═" * 65)
    print("  🏛  RAG LIBRARY — Milestone 3: Retrieval")
    print("  Gõ câu hỏi và nhấn Enter. Gõ 'quit' hoặc 'q' để thoát.")
    print("═" * 65)

    while True:
        try:
            query = input("\n❓ Câu hỏi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Thoát.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "q", "exit"):
            print("👋 Thoát.")
            break

        t0      = time.time()
        results = run_query(query, retriever, reranker)
        elapsed = time.time() - t0
        _print_results(query, results, elapsed)


def _single_query_mode(
    query:     str,
    retriever: DenseRetriever,
    reranker:  CrossEncoderReranker,
) -> None:
    """Chạy 1 query rồi thoát."""
    t0      = time.time()
    results = run_query(query, retriever, reranker)
    elapsed = time.time() - t0
    _print_results(query, results, elapsed)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline Milestone 3: Retrieval (Dense + Rerank)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python src/pipeline_m3.py
  python src/pipeline_m3.py --query "Điều kiện tiên quyết môn ML?"
  python src/pipeline_m3.py --no-rerank --top-k 5
        """,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Câu hỏi cụ thể (nếu không truyền → chế độ interactive)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL_TOP_K,
        help=f"Số ứng viên từ FAISS (default: {RETRIEVAL_TOP_K})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=RERANKER_TOP_N,
        help=f"Số chunk giữ lại sau rerank (default: {RERANKER_TOP_N})",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Tắt Cross-Encoder reranker (dùng dense score trực tiếp)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    retriever, reranker = build_retrieval_pipeline(
        use_reranker = not args.no_rerank,
        top_k        = args.top_k,
        top_n        = args.top_n,
    )

    if args.query:
        _single_query_mode(args.query, retriever, reranker)
    else:
        _interactive_mode(retriever, reranker)
