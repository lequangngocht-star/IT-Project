"""
src/pipeline_m4.py
-------------------
Entry-point Milestone 4: Full RAG Pipeline — Retrieval + LLM Generation.

Đây là milestone hoàn chỉnh đầu tiên: user hỏi → hệ thống trả lời
bằng câu văn hoàn chỉnh + trích dẫn nguồn.

Pipeline đầy đủ (mỗi query):
    User query
        ↓ [M3] DenseRetriever.retrieve()       → top-10 ứng viên
        ↓ [M3] CrossEncoderReranker.rerank()   → top-5 chính xác
        ↓ [M4] Prompter.build_messages()       → chat messages + context
        ↓ [M4] LLMManager.generate_chat()      → raw text answer
        ↓ [M4] parse_response()                → RAGResponse chuẩn hóa
        ↓       Hiển thị câu trả lời + nguồn trích dẫn

Cách dùng:
    # Interactive mode
    python src/pipeline_m4.py

    # Single query
    python src/pipeline_m4.py --query "sentence?"

    # Không dùng reranker (ablation)
    python src/pipeline_m4.py --no-rerank

    # Chọn model khác
    python src/pipeline_m4.py --model "Qwen/Qwen2.5-3B-Instruct"

    # Dùng quantization 4-bit (cần bitsandbytes, GPU khuyến nghị)
    python src/pipeline_m4.py --4bit

Prerequisite:
    - pipeline_m1.py → data/processed/chunks.json       ✓
    - pipeline_m2.py → vector_store/faiss.index          ✓
    - (pipeline_m3.py không cần chạy riêng — M4 tích hợp luôn)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FAISS_INDEX_FILE,
    LLM_MODEL_NAME,
    LLM_DEVICE,
    LLM_LOAD_IN_4BIT,
    LLM_LOAD_IN_8BIT,
    RETRIEVAL_TOP_K,
    RERANKER_TOP_N,
    RERANKER_ENABLED,
)
from src.logger import get_logger
from src.pipeline_m3 import build_retrieval_pipeline, run_query as retrieve
from src.LLM.model_manager import LLMManager
from src.LLM.prompter import (
    build_messages,
    parse_response,
    format_response_for_display,
    RAGResponse,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Full RAG query
# ─────────────────────────────────────────────────────────────

def rag_query(
    query:     str,
    retriever,
    reranker,
    llm:       LLMManager,
    top_n:     int | None = None,
) -> RAGResponse:
    """
    Chạy full RAG pipeline cho một câu hỏi.

    Đây là hàm cốt lõi của Milestone 4, kết nối M3 và LLM.
    Được export để Milestone 5 (evaluation) gọi hàng loạt trên test set.

    Luồng xử lý:
        1. Retrieve: lấy top-N chunks liên quan nhất.
        2. Prompt:   đóng gói query + chunks thành messages.
        3. Generate: LLM sinh câu trả lời dựa trên context.
        4. Parse:    chuẩn hóa output thành RAGResponse.

    Args:
        query:     Câu hỏi người dùng (tiếng Việt).
        retriever: DenseRetriever đã khởi tạo (từ M3).
        reranker:  CrossEncoderReranker đã khởi tạo (từ M3).
        llm:       LLMManager đã load model.
        top_n:     Override số chunk đưa vào context.

    Returns:
        RAGResponse với answer, sources, latency.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query không được rỗng")

    logger.info(f"RAG query: '{query[:70]}...'")
    t_total = time.time()

    # ── Bước 1: Retrieval (M3) ────────────────────────────────
    t_ret = time.time()
    chunks = retrieve(query, retriever, reranker, top_n=top_n)
    ret_elapsed = time.time() - t_ret
    logger.info(f"Retrieval: {len(chunks)} chunks | {ret_elapsed:.3f}s")

    if not chunks:
        logger.warning("Không retrieve được chunk nào — trả lời 'không có thông tin'")
        return RAGResponse(
            query=query,
            answer="Xin lỗi, tôi không tìm thấy tài liệu liên quan đến câu hỏi của bạn trong thư viện số.",
            sources=[],
            has_answer=False,
            context_used=0,
            latency_s=time.time() - t_total,
        )

    # ── Bước 2: Build prompt ──────────────────────────────────
    messages, used_chunks = build_messages(query, chunks)
    logger.info(f"Prompt: {len(used_chunks)} chunks trong context")

    # ── Bước 3: LLM Generation ───────────────────────────────
    t_gen = time.time()
    raw_answer = llm.generate_chat(messages)
    gen_elapsed = time.time() - t_gen
    logger.info(f"LLM generation: {gen_elapsed:.2f}s")

    # ── Bước 4: Parse & chuẩn hóa ───────────────────────────
    total_elapsed = time.time() - t_total
    response = parse_response(
        raw_answer=raw_answer,
        query=query,
        used_chunks=used_chunks,
        latency_s=total_elapsed,
    )

    return response


# ─────────────────────────────────────────────────────────────
# Pipeline initialization
# ─────────────────────────────────────────────────────────────

def build_rag_pipeline(
    model_name:   str  = LLM_MODEL_NAME,
    device:       str  = LLM_DEVICE,
    load_in_4bit: bool = LLM_LOAD_IN_4BIT,
    load_in_8bit: bool = LLM_LOAD_IN_8BIT,
    use_reranker: bool = RERANKER_ENABLED,
    top_k:        int  = RETRIEVAL_TOP_K,
    top_n:        int  = RERANKER_TOP_N,
):
    """
    Khởi tạo toàn bộ RAG pipeline: Retrieval + LLM.

    Hàm này load tất cả model 1 lần — sau đó dùng nhiều query
    mà không phải load lại. Gọi 1 lần khi start ứng dụng.

    Args:
        model_name:   LLM model ID (HuggingFace).
        device:       "auto" | "cpu" | "cuda".
        load_in_4bit: Bật 4-bit quantization.
        load_in_8bit: Bật 8-bit quantization.
        use_reranker: Có dùng Cross-Encoder reranker không.
        top_k:        Số ứng viên từ FAISS.
        top_n:        Số chunk giữ lại sau rerank.

    Returns:
        Tuple (retriever, reranker, llm_manager).

    Raises:
        FileNotFoundError: Nếu FAISS index chưa build.
    """
    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index không tồn tại: {FAISS_INDEX_FILE}\n"
            "  → Chạy trước: python src/pipeline_m1.py && python src/pipeline_m2.py"
        )

    logger.info("═" * 60)
    logger.info("  KHỞI TẠO RAG PIPELINE — Milestone 4")
    logger.info("═" * 60)

    # ── M3: Retrieval pipeline ────────────────────────────────
    logger.info("[1/2] Khởi tạo Retrieval Pipeline (M3)...")
    retriever, reranker = build_retrieval_pipeline(
        use_reranker=use_reranker,
        top_k=top_k,
        top_n=top_n,
    )

    # ── M4: LLM ──────────────────────────────────────────────
    logger.info(f"[2/2] Load LLM: {model_name}...")
    llm = LLMManager(
        model_name=model_name,
        device=device,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    llm.load()

    logger.info("═" * 60)
    logger.info("  ✅ RAG Pipeline sẵn sàng")
    logger.info(f"  LLM    : {model_name}")
    logger.info(f"  Quant  : {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'full'}")
    logger.info(f"  Rerank : {'ON' if use_reranker else 'OFF'}")
    logger.info("═" * 60)

    return retriever, reranker, llm


# ─────────────────────────────────────────────────────────────
# CLI modes
# ─────────────────────────────────────────────────────────────

def _interactive_mode(retriever, reranker, llm: LLMManager) -> None:
    """Vòng lặp hỏi đáp tương tác với full RAG."""
    print("\n" + "═" * 65)
    print("  🏛  RAG LIBRARY — Milestone 4: Hỏi đáp thông minh")
    print("  Model:", llm.model_name)
    print("  Gõ câu hỏi và nhấn Enter. Gõ 'quit' để thoát.")
    print("═" * 65)

    while True:
        try:
            query = input("\n❓ Câu hỏi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Thoát.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "q", "exit", "thoát"):
            print("👋 Thoát.")
            break

        try:
            response = rag_query(query, retriever, reranker, llm)
            print(format_response_for_display(response))
        except Exception as e:
            logger.error(f"Lỗi xử lý query: {e}")
            print(f"\n⚠️  Lỗi: {e}\n")


def _single_query_mode(
    query:     str,
    retriever,
    reranker,
    llm:       LLMManager,
) -> None:
    """Chạy 1 query rồi in kết quả và thoát."""
    try:
        response = rag_query(query, retriever, reranker, llm)
        print(format_response_for_display(response))
    except Exception as e:
        logger.error(f"Lỗi: {e}")
        print(f"\n⚠️  Lỗi: {e}\n")


# ─────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline Milestone 4: Full RAG (Retrieval + LLM Generation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python src/pipeline_m4.py
  python src/pipeline_m4.py --query "Machine Learning là gì?"
  python src/pipeline_m4.py --model "Qwen/Qwen2.5-3B-Instruct" --no-rerank
  python src/pipeline_m4.py --4bit --query "Tóm tắt chương 1 giáo trình AI"
        """,
    )
    parser.add_argument(
        "--query", "-q",
        type=str, default=None,
        help="Câu hỏi cụ thể (không truyền → chế độ interactive)",
    )
    parser.add_argument(
        "--model",
        type=str, default=LLM_MODEL_NAME,
        help=f"HuggingFace model ID (default: {LLM_MODEL_NAME})",
    )
    parser.add_argument(
        "--device",
        type=str, default=LLM_DEVICE,
        help=f"Device: auto|cpu|cuda (default: {LLM_DEVICE})",
    )
    parser.add_argument(
        "--4bit",
        dest="load_4bit",
        action="store_true",
        help="Bật 4-bit quantization (cần bitsandbytes + GPU khuyến nghị)",
    )
    parser.add_argument(
        "--8bit",
        dest="load_8bit",
        action="store_true",
        help="Bật 8-bit quantization (cần bitsandbytes)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Tắt Cross-Encoder reranker",
    )
    parser.add_argument(
        "--top-k",
        type=int, default=RETRIEVAL_TOP_K,
        help=f"Số ứng viên từ FAISS (default: {RETRIEVAL_TOP_K})",
    )
    parser.add_argument(
        "--top-n",
        type=int, default=RERANKER_TOP_N,
        help=f"Số chunk vào context LLM (default: {RERANKER_TOP_N})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    retriever, reranker, llm = build_rag_pipeline(
        model_name   = args.model,
        device       = args.device,
        load_in_4bit = args.load_4bit,
        load_in_8bit = args.load_8bit,
        use_reranker = not args.no_rerank,
        top_k        = args.top_k,
        top_n        = args.top_n,
    )

    if args.query:
        _single_query_mode(args.query, retriever, reranker, llm)
    else:
        _interactive_mode(retriever, reranker, llm)
