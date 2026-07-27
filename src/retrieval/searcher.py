"""
src/retrieval/searcher.py
--------------------------
Chịu trách nhiệm DUY NHẤT: Dense Retrieval — nhận câu hỏi, trả về
top-K chunks liên quan nhất từ FAISS index.

Vị trí trong pipeline RAG:
    User query
        → DenseRetriever.retrieve()
            → embed_query()          # EmbeddingModel
            → FaissVectorStore.search()
        → list[dict] ứng viên
        → Reranker (searcher không biết gì về bước này)

Tại sao tách searcher.py và reranker.py?
- Searcher: tốc độ cao, approximate, dùng ANN (Approximate Nearest Neighbor).
  Trả về RETRIEVAL_TOP_K ứng viên (nhiều hơn cần thiết).
- Reranker: chính xác hơn nhưng chậm hơn (cross-encoder chạy full attention).
  Chỉ chạy trên tập ứng viên nhỏ từ searcher.
- Tách 2 bước = "coarse-to-fine": nhanh trước, chính xác sau.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    RETRIEVAL_TOP_K,
)
from src.logger import get_logger
from src.embedding.embedder import EmbeddingModel
from vector_store.faiss_store import FaissVectorStore

logger = get_logger(__name__)


class DenseRetriever:
    """
    Dense Retrieval dựa trên FAISS + BGE-M3 embedding.

    "Dense" nghĩa là dùng vector embedding dày đặc (dense vectors) để
    đo độ tương đồng ngữ nghĩa — khác với "sparse" (BM25/TF-IDF dùng
    từ khóa rời rạc).

    Dense tốt hơn sparse khi:
    - Câu hỏi dùng từ đồng nghĩa với tài liệu.
    - Tài liệu có cấu trúc văn xuôi, không phải danh sách từ khóa.
    - Ngôn ngữ có biến thể cao như tiếng Việt.

    Attributes:
        _embedder:     EmbeddingModel để encode query.
        _vector_store: FaissVectorStore để tìm kiếm vector.
        _top_k:        Số chunk trả về từ FAISS (trước rerank).
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: FaissVectorStore,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> None:
        """
        Khởi tạo DenseRetriever với các dependency đã được inject.

        Tại sao dùng dependency injection thay vì tạo bên trong?
        - Test dễ hơn: mock embedder và store mà không cần file thật.
        - Tái sử dụng: cùng embedder dùng cho cả indexing và retrieval.
        - Linh hoạt: swap model mà không đụng vào class này.

        Args:
            embedder:     EmbeddingModel đã load (dùng chung với index_builder).
            vector_store: FaissVectorStore đã load từ file.
            top_k:        Số kết quả trả về từ FAISS (ứng viên cho reranker).
        """
        self._embedder     = embedder
        self._vector_store = vector_store
        self._top_k        = top_k

        logger.info(
            f"DenseRetriever sẵn sàng | "
            f"index_size={vector_store.size} | top_k={top_k}"
        )

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Truy xuất các chunk liên quan nhất cho câu hỏi.

        Pipeline nội bộ:
            1. Embed query bằng BGE-M3 (có instruction prefix).
            2. Tìm top_k vectors gần nhất trong FAISS (cosine similarity).
            3. Trả về list[dict] chunk kèm trường "score".

        Args:
            query:  Câu hỏi từ người dùng (tiếng Việt).
            top_k:  Override top_k mặc định (None = dùng giá trị khởi tạo).

        Returns:
            List[dict] sắp xếp giảm dần theo score cosine similarity.
            Mỗi dict gồm tất cả metadata của chunk + key "score" (float).

        Raises:
            ValueError: Nếu query rỗng.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query không được rỗng")

        k = top_k if top_k is not None else self._top_k

        logger.info(f"Dense retrieve | query='{query[:60]}...' | top_k={k}")

        # Bước 1: Embed query
        query_embedding: np.ndarray = self._embedder.embed_query(query)

        # Bước 2: FAISS search
        results = self._vector_store.search(query_embedding, top_k=k)

        logger.info(
            f"Dense retrieve xong | "
            f"trả về {len(results)} chunks | "
            f"top score={results[0]['score']:.4f}" if results else "Dense retrieve xong | không có kết quả"
        )
        return results

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int | None = None,
    ) -> list[list[dict]]:
        """
        Truy xuất cho nhiều câu hỏi cùng lúc.
        Dùng trong Milestone 5 (evaluation) để tính Recall@k trên toàn bộ test set.

        Args:
            queries: List câu hỏi.
            top_k:   Số kết quả mỗi query.

        Returns:
            List of lists — results[i] là kết quả cho queries[i].
        """
        k = top_k if top_k is not None else self._top_k
        logger.info(f"Batch retrieve | {len(queries)} queries | top_k={k}")

        all_results: list[list[dict]] = []
        for query in queries:
            results = self.retrieve(query, top_k=k)
            all_results.append(results)

        return all_results

    # ─────────────────────────────────────────────────────────
    # Factory method (convenience)
    # ─────────────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls,
        index_path: Path = FAISS_INDEX_FILE,
        meta_path:  Path = FAISS_META_FILE,
        top_k:      int  = RETRIEVAL_TOP_K,
    ) -> "DenseRetriever":
        """
        Tạo DenseRetriever từ file index đã lưu.
        Shortcut để không phải khởi tạo embedder và store riêng lẻ.

        Args:
            index_path: Đường dẫn FAISS binary index.
            meta_path:  Đường dẫn metadata JSON.
            top_k:      Số kết quả trả về.

        Returns:
            DenseRetriever đã sẵn sàng dùng.
        """
        logger.info("Khởi tạo DenseRetriever từ file index...")
        embedder     = EmbeddingModel()
        vector_store = FaissVectorStore.load(index_path, meta_path)
        return cls(embedder=embedder, vector_store=vector_store, top_k=top_k)

    def __repr__(self) -> str:
        return (
            f"DenseRetriever("
            f"index_size={self._vector_store.size}, "
            f"top_k={self._top_k})"
        )
