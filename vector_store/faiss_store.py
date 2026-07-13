"""
src/vectordb/faiss_store.py
----------------------------
Chịu trách nhiệm DUY NHẤT: lưu trữ và tìm kiếm vector bằng FAISS.

Tại sao tách riêng file này?
- Hoàn toàn độc lập với embedding model và chunker.
- Dễ thay bằng ChromaDB / Qdrant sau này chỉ bằng cách tạo class mới
  có cùng interface (add / search / save / load).
- Metadata store (list[dict]) tách biệt với FAISS index —
  vì FAISS chỉ lưu được float array, không lưu được dict.

Lựa chọn FAISS index type:
- IndexFlatIP (Inner Product): brute-force chính xác 100%.
  Phù hợp dataset nhỏ-vừa (< 100K vectors).
  Sau normalize embedding, IP = cosine similarity.
- Nếu cần scale lên triệu vector → đổi sang IndexIVFFlat (approximate).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    TOP_K,
)
from src.logger import get_logger

logger = get_logger(__name__)


class FaissVectorStore:
    """
    FAISS-backed vector store với metadata sidecar.

    Cấu trúc nội bộ:
        _index    : faiss.IndexFlatIP  — lưu float32 vectors
        _metadata : list[dict]         — lưu chunk dicts tương ứng theo index

    Bất biến: len(_metadata) == _index.ntotal luôn đúng sau mỗi thao tác.
    """

    def __init__(self, dim: int) -> None:
        """
        Khởi tạo index FAISS mới (rỗng).

        Args:
            dim: Số chiều của embedding vector (phải khớp với embedding model).
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("Cài đặt faiss-cpu: pip install faiss-cpu")

        self._dim = dim
        # IndexFlatIP: exact search bằng inner product
        # Với embedding đã L2-normalize → IP == cosine similarity
        self._index = faiss.IndexFlatIP(dim)
        self._metadata: list[dict] = []

        logger.info(f"Khởi tạo FaissVectorStore | dim={dim} | IndexFlatIP")

    # ─────────────────────────────────────────────────────────
    # Core Operations
    # ─────────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """
        Thêm batch vectors và metadata tương ứng vào index.

        Args:
            embeddings: shape (N, dim), dtype=float32, đã L2-normalize.
            chunks:     list[dict] tương ứng, len == N.

        Raises:
            ValueError: Nếu shape không khớp.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self._dim:
            raise ValueError(
                f"Embedding shape sai: nhận {embeddings.shape}, "
                f"cần (N, {self._dim})"
            )
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Số embedding ({len(embeddings)}) "
                f"≠ số chunks ({len(chunks)})"
            )

        # FAISS yêu cầu C-contiguous float32
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(vectors)
        self._metadata.extend(chunks)

        logger.info(
            f"Thêm {len(chunks)} vectors | "
            f"Tổng trong index: {self.size}"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Tìm top_k chunks gần nhất với query vector.

        Args:
            query_embedding: shape (1, dim) hoặc (dim,), dtype=float32.
            top_k:           Số kết quả trả về (tối đa size của index).

        Returns:
            List[dict] — mỗi dict là chunk gốc + thêm key "score" (float).
            Sắp xếp giảm dần theo score (chunk liên quan nhất ở đầu).
        """
        if self.size == 0:
            logger.warning("Index rỗng — không có kết quả")
            return []

        # Đảm bảo shape (1, dim)
        vec = query_embedding.reshape(1, -1).astype(np.float32)
        k   = min(top_k, self.size)

        # scores shape (1, k), indices shape (1, k)
        scores, indices = self._index.search(vec, k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS trả -1 khi không đủ k kết quả (hiếm gặp với FlatIP)
                continue
            # Copy chunk dict + gắn score — không mutate dict gốc
            entry = {**self._metadata[idx], "score": float(score)}
            results.append(entry)

        score_strs = [f"{r['score']:.3f}" for r in results]
        logger.debug(f"Search top-{k} | scores={score_strs}")
        return results

    # ─────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────

    def save(
        self,
        index_path: Path = FAISS_INDEX_FILE,
        meta_path:  Path = FAISS_META_FILE,
    ) -> None:
        """
        Lưu FAISS index (binary) và metadata (JSON) ra file.

        Tại sao lưu 2 file tách biệt?
        - FAISS index là binary — không chứa text được.
        - Metadata là JSON — human-readable, dễ inspect khi debug.

        Args:
            index_path: Đường dẫn file .index (binary FAISS).
            meta_path:  Đường dẫn file .json (metadata).
        """
        import faiss

        index_path = Path(index_path)
        meta_path  = Path(meta_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(index_path))
        logger.info(f"Đã lưu FAISS index ({self.size} vectors) → {index_path}")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Đã lưu metadata ({len(self._metadata)} entries) → {meta_path}")

    @classmethod
    def load(
        cls,
        index_path: Path = FAISS_INDEX_FILE,
        meta_path:  Path = FAISS_META_FILE,
    ) -> "FaissVectorStore":
        """
        Tải FAISS index và metadata từ file đã lưu.

        Args:
            index_path: Đường dẫn file .index (binary FAISS).
            meta_path:  Đường dẫn file .json (metadata).

        Returns:
            FaissVectorStore đã được khôi phục đầy đủ.

        Raises:
            FileNotFoundError: Nếu một trong hai file không tồn tại.
        """
        import faiss

        index_path = Path(index_path)
        meta_path  = Path(meta_path)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index không tồn tại: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file không tồn tại: {meta_path}")

        # Dùng __new__ để bypass __init__ (tránh tạo index rỗng rồi replace)
        store = cls.__new__(cls)
        store._index    = faiss.read_index(str(index_path))
        store._dim      = store._index.d

        with open(meta_path, encoding="utf-8") as f:
            store._metadata = json.load(f)

        # Kiểm tra bất biến sau khi load
        if store._index.ntotal != len(store._metadata):
            logger.warning(
                f"Bất đồng bộ: index có {store._index.ntotal} vectors "
                f"nhưng metadata có {len(store._metadata)} entries"
            )

        logger.info(
            f"Đã tải FaissVectorStore | "
            f"{store.size} vectors | dim={store._dim}"
        )
        return store

    # ─────────────────────────────────────────────────────────
    # Properties & Utilities
    # ─────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Số vectors hiện có trong index."""
        return self._index.ntotal

    @property
    def dim(self) -> int:
        """Số chiều của embedding."""
        return self._dim

    def get_all_metadata(self) -> list[dict]:
        """Trả về toàn bộ metadata (read-only copy)."""
        return list(self._metadata)

    def reset(self) -> None:
        """Xóa toàn bộ index và metadata, giữ nguyên dim."""
        import faiss
        self._index    = faiss.IndexFlatIP(self._dim)
        self._metadata = []
        logger.info("Reset FaissVectorStore — index và metadata đã xóa")

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"FaissVectorStore("
            f"size={self.size}, "
            f"dim={self._dim}, "
            f"index_type='IndexFlatIP')"
        )