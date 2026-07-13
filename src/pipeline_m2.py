"""
src/pipeline_m2.py
-------------------
Entry-point chạy Milestone 2: Embedding + Build FAISS Index.

Script này có thể chạy theo 2 chế độ:
  - Chế độ đầy đủ (mặc định): chạy M1 trước, sau đó M2.
  - Chế độ M2 only: dùng chunks.json đã có từ lần chạy M1 trước.

Cách dùng:
    # Chạy M2 (dùng chunks đã có từ M1)
    python src/pipeline_m2.py

    # Chạy M1 + M2 liên tiếp
    python src/pipeline_m2.py --full

    # Force rebuild dù index đã tồn tại
    python src/pipeline_m2.py --rebuild

Kết quả:
    vector_store/faiss.index       ← FAISS binary index
    vector_store/faiss_meta.json   ← Metadata tương ứng
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED_DIR, FAISS_INDEX_FILE, FAISS_META_FILE
from src.logger import get_logger
from src.embedding.index_builder import build_index, load_index, index_exists
from vector_store.faiss_store import FaissVectorStore
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Demo search — kiểm tra index sau khi build
# ─────────────────────────────────────────────────────────────

def _smoke_test(store: FaissVectorStore) -> None:
    """
    Chạy vài câu hỏi mẫu để xác nhận pipeline hoạt động đúng.
    Không phải evaluation chính thức — chỉ là sanity check.
    """
    from src.embedding.embedder import EmbeddingModel

    logger.info("─" * 55)
    logger.info("SMOKE TEST — Kiểm tra retrieval")
    logger.info("─" * 55)

    # Tái sử dụng model đã load (tránh load lại)
    embedder = EmbeddingModel()

    test_queries = [
        "Machine Learning là gì?",
        "Điều kiện tiên quyết để đăng ký môn học",
        "Phương pháp đánh giá sinh viên",
    ]

    for query in test_queries:
        q_emb   = embedder.embed_query(query)
        results = store.search(q_emb, top_k=2)

        logger.info(f"\n🔍 Query: '{query}'")
        for i, r in enumerate(results):
            logger.info(
                f"   #{i+1} score={r['score']:.4f} | "
                f"file={r.get('file_name', 'N/A')} | "
                f"chunk={r.get('chunk_id', 'N/A')}"
            )
            # Hiển thị 120 ký tự đầu của chunk
            preview = r["text"][:120].replace("\n", " ")
            logger.info(f"       → {preview}...")


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def run_pipeline_m2(
    run_m1_first: bool = False,
    rebuild: bool = False,
) -> FaissVectorStore:
    """
    Chạy pipeline Milestone 2.

    Args:
        run_m1_first: Nếu True, chạy M1 (ingestion + chunking) trước.
        rebuild:      Nếu True, build lại index dù đã tồn tại.

    Returns:
        FaissVectorStore đã build hoặc load.
    """
    t_total = time.time()

    logger.info("═" * 55)
    logger.info("  PIPELINE MILESTONE 2: EMBEDDING + FAISS INDEX")
    logger.info("═" * 55)

    # ── Chạy M1 nếu được yêu cầu ─────────────────────────────
    if run_m1_first:
        logger.info("[PRE] Chạy Milestone 1 trước...")
        from src.pipeline_m1 import run_pipeline as run_m1
        run_m1()

    # ── Kiểm tra chunks.json đầu vào ─────────────────────────
    chunks_path = DATA_PROCESSED_DIR / "chunks.json"
    if not chunks_path.exists():
        logger.error(
            f"Không tìm thấy chunks.json tại {chunks_path}.\n"
            "  → Chạy Milestone 1 trước: python src/pipeline_m1.py\n"
            "  → Hoặc dùng: python src/pipeline_m2.py --full"
        )
        sys.exit(1)

    # ── Load index nếu đã tồn tại và không cần rebuild ───────
    if index_exists() and not rebuild:
        logger.info("Index đã tồn tại. Dùng --rebuild để build lại.")
        store = load_index()
    else:
        if rebuild and index_exists():
            logger.info("--rebuild flag: build lại index từ đầu")
        store = build_index(chunks_path=chunks_path)

    # ── Smoke test ────────────────────────────────────────────
    _smoke_test(store)

    logger.info("═" * 55)
    logger.info(f"  ✅ MILESTONE 2 HOÀN THÀNH ({time.time()-t_total:.2f}s)")
    logger.info(f"  Index: {store.size} vectors | dim={store.dim}")
    logger.info("═" * 55)

    return store


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline Milestone 2: Embedding + FAISS Index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python src/pipeline_m2.py              # M2 only (cần chunks.json có sẵn)
  python src/pipeline_m2.py --full       # Chạy M1 rồi M2
  python src/pipeline_m2.py --rebuild    # Build lại index từ đầu
        """,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Chạy Milestone 1 (ingestion + chunking) trước khi embedding",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Build lại FAISS index dù đã tồn tại",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline_m2(run_m1_first=args.full, rebuild=args.rebuild)