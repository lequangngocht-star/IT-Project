"""
src/embedding/index_builder.py
-------------------------------
Chịu trách nhiệm DUY NHẤT: điều phối quá trình build FAISS index từ chunks.

Đây là cầu nối giữa Milestone 1 (chunks) và Milestone 2 (vector store).

Tại sao tách riêng file này?
- embedder.py không biết gì về chunks hay FAISS.
- faiss_store.py không biết gì về embedding model hay chunks.
- index_builder.py là "coordinator" — gọi embedder rồi đẩy kết quả vào store.
- Pattern này gọi là Facade: ẩn sự phức tạp của pipeline sau một API đơn giản.

Luồng dữ liệu:
    chunks.json (M1 output)
        → load_chunks()
        → EmbeddingModel.embed_corpus()
        → FaissVectorStore.add()
        → FaissVectorStore.save()
        → faiss.index + faiss_meta.json (M2 output)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DATA_PROCESSED_DIR,
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    EMBEDDING_BATCH_SIZE,
)
from src.logger import get_logger
from src.preprocessing.chunker import load_chunks
from src.embedding.embedder import EmbeddingModel
from vector_store.faiss_store import FaissVectorStore
logger = get_logger(__name__)


def build_index(
    chunks_path: Path = DATA_PROCESSED_DIR / "chunks.json",
    index_path:  Path = FAISS_INDEX_FILE,
    meta_path:   Path = FAISS_META_FILE,
    batch_size:  int  = EMBEDDING_BATCH_SIZE,
    embedder:    EmbeddingModel | None = None,
) -> FaissVectorStore:
    """
    Build FAISS index từ file chunks đã tạo ở Milestone 1.

    Quy trình:
        1. Tải chunks từ JSON.
        2. Trích xuất list[str] text để embed.
        3. Embed toàn bộ text bằng EmbeddingModel.
        4. Tạo FaissVectorStore và add embeddings + metadata.
        5. Lưu index và metadata ra file.

    Args:
        chunks_path: Đường dẫn file chunks.json từ M1.
        index_path:  Nơi lưu FAISS binary index.
        meta_path:   Nơi lưu metadata JSON.
        batch_size:  Batch size cho embedding (tăng nếu có GPU).
        embedder:    EmbeddingModel đã khởi tạo (None → tạo mới với config mặc định).
                     Truyền vào nếu đã load model từ trước để tránh load lại.

    Returns:
        FaissVectorStore đã được build và lưu.

    Raises:
        FileNotFoundError: Nếu chunks_path không tồn tại.
    """
    t_start = time.time()
    logger.info("─" * 55)
    logger.info("BUILD INDEX — Milestone 2")
    logger.info("─" * 55)

    # ── Bước 1: Tải chunks ───────────────────────────────────
    logger.info(f"[1/4] Tải chunks từ: {chunks_path}")
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"File chunks rỗng hoặc không có dữ liệu: {chunks_path}")

    texts = [c["text"] for c in chunks]
    logger.info(f"      {len(texts)} chunks | avg {sum(len(t) for t in texts)//len(texts)} ký tự/chunk")

    # ── Bước 2: Load embedding model ─────────────────────────
    logger.info("[2/4] Khởi tạo Embedding Model...")
    if embedder is None:
        embedder = EmbeddingModel()

    # ── Bước 3: Embed corpus ──────────────────────────────────
    logger.info(f"[3/4] Embedding {len(texts)} chunks (batch_size={batch_size})...")
    t_embed = time.time()
    embeddings: np.ndarray = embedder.embed_corpus(
        texts,
        batch_size=batch_size,
        show_progress=True,
    )
    embed_elapsed = time.time() - t_embed
    logger.info(
        f"      Xong embedding | shape={embeddings.shape} | "
        f"time={embed_elapsed:.1f}s | "
        f"speed={len(texts)/embed_elapsed:.0f} chunks/s"
    )

    # ── Bước 4: Build và lưu FAISS index ─────────────────────
    logger.info("[4/4] Build FAISS index và lưu...")
    store = FaissVectorStore(dim=embedder.dim)
    store.add(embeddings, chunks)
    store.save(index_path=index_path, meta_path=meta_path)

    total_elapsed = time.time() - t_start
    logger.info("─" * 55)
    logger.info(f"BUILD INDEX HOÀN THÀNH | {total_elapsed:.2f}s")
    logger.info(f"  Vectors trong index : {store.size}")
    logger.info(f"  FAISS index file    : {index_path}")
    logger.info(f"  Metadata file       : {meta_path}")
    logger.info("─" * 55)

    return store


def load_index(
    index_path: Path = FAISS_INDEX_FILE,
    meta_path:  Path = FAISS_META_FILE,
) -> FaissVectorStore:
    """
    Tải FAISS index đã build sẵn từ file.

    Dùng khi hệ thống khởi động lại — không cần build lại từ đầu.

    Args:
        index_path: Đường dẫn file FAISS binary index.
        meta_path:  Đường dẫn file metadata JSON.

    Returns:
        FaissVectorStore đã restore đầy đủ.
    """
    logger.info(f"Tải FAISS index từ: {index_path}")
    store = FaissVectorStore.load(index_path=index_path, meta_path=meta_path)
    logger.info(f"Đã tải index: {store.size} vectors | dim={store.dim}")
    return store


def index_exists(
    index_path: Path = FAISS_INDEX_FILE,
    meta_path:  Path = FAISS_META_FILE,
) -> bool:
    """
    Kiểm tra FAISS index đã được build và lưu chưa.

    Dùng để quyết định build mới hay load lại:
        if index_exists():
            store = load_index()
        else:
            store = build_index()

    Args:
        index_path: Đường dẫn file FAISS binary index.
        meta_path:  Đường dẫn file metadata JSON.

    Returns:
        True nếu cả hai file đều tồn tại.
    """
    return Path(index_path).exists() and Path(meta_path).exists()