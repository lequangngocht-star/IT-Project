"""
src/embedding/embedder.py
--------------------------
Chịu trách nhiệm DUY NHẤT: load model embedding và sinh vector.

Tại sao tách riêng file này?
- Tách biệt model loading khỏi vector DB logic.
- Dễ swap model: chỉ đổi class này, không đụng vào FAISS hay chunker.
- Có thể mock class này trong test mà không cần download model thật.

Thiết kế:
- Class EmbeddingModel là singleton-like: load 1 lần, dùng nhiều lần.
- Tách embed_corpus() và embed_query() vì lý do kỹ thuật:
    BGE-M3 khuyến nghị thêm instruction prefix cho query để tăng recall.
    Corpus KHÔNG thêm prefix (embedding lúc index).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
)
from src.logger import get_logger

logger = get_logger(__name__)

# Instruction prefix giúp BGE-M3 hiểu context retrieval khi encode query
# Tham khảo: https://huggingface.co/BAAI/bge-m3
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    """
    Wrapper xung quanh SentenceTransformer cho bài toán RAG.

    Attributes:
        model_name: Tên model trên HuggingFace Hub.
        device:     "cpu" hoặc "cuda".
        dim:        Số chiều của embedding vector.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = EMBEDDING_DEVICE,
    ) -> None:
        """
        Load model từ HuggingFace Hub (hoặc cache local).

        Args:
            model_name: HuggingFace model ID, ví dụ "BAAI/bge-m3".
            device:     "cpu" hoặc "cuda".

        Raises:
            ImportError: Nếu sentence-transformers chưa được cài.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Cài đặt sentence-transformers: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device

        logger.info(f"Đang load embedding model: {model_name} (device={device})")
        logger.info("Lần đầu chạy sẽ download model (~570MB cho BGE-M3)...")

        self._model = SentenceTransformer(model_name, device=device)
        self.dim: int = self._model.get_sentence_embedding_dimension()

        logger.info(f"Model sẵn sàng | dim={self.dim} | device={device}")

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def embed_corpus(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode danh sách văn bản từ corpus (documents/chunks).

        KHÔNG thêm instruction prefix — corpus được index nguyên văn.
        Dùng normalize_embeddings=True để dot product = cosine similarity.

        Args:
            texts:         List chuỗi văn bản cần encode.
            batch_size:    Số text xử lý một lúc. Tăng nếu có GPU.
            show_progress: Hiện progress bar (tqdm).

        Returns:
            np.ndarray shape (N, dim), dtype=float32, đã L2-normalize.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        logger.info(
            f"Encoding corpus: {len(texts)} texts | "
            f"batch_size={batch_size} | device={self.device}"
        )

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # dot(a,b) == cosine_sim(a,b)
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

        result = embeddings.astype(np.float32)
        logger.info(f"Corpus embedding xong | shape={result.shape}")
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode một câu hỏi đơn lẻ với instruction prefix.

        Tại sao có prefix cho query nhưng không có cho corpus?
        BGE-M3 được fine-tune theo cách này: corpus giữ nguyên,
        query được thêm prefix để mô hình hiểu đây là "tìm kiếm".
        Kết quả: recall tăng ~2-5% trên nhiều benchmark.

        Args:
            query: Câu hỏi cần tìm kiếm.

        Returns:
            np.ndarray shape (1, dim), dtype=float32, đã L2-normalize.
        """
        prefixed = f"{_BGE_QUERY_PREFIX}{query}"

        embedding = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embedding.astype(np.float32)   # shape (1, dim)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """
        Encode nhiều câu hỏi cùng lúc (dùng trong evaluation).

        Args:
            queries: List câu hỏi.

        Returns:
            np.ndarray shape (N, dim), dtype=float32.
        """
        prefixed = [f"{_BGE_QUERY_PREFIX}{q}" for q in queries]
        embeddings = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    # ─────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Tính cosine similarity giữa 2 vectors (đã normalize → chỉ là dot product).

        Args:
            a, b: 1D arrays có cùng shape.

        Returns:
            Float trong khoảng [-1, 1].
        """
        return float(np.dot(a.flatten(), b.flatten()))

    def __repr__(self) -> str:
        return f"EmbeddingModel(model='{self.model_name}', dim={self.dim}, device='{self.device}')"