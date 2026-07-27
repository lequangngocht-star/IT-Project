"""
src/retrieval/reranker.py
--------------------------
Chịu trách nhiệm DUY NHẤT: Reranking — chấm điểm lại danh sách ứng
viên từ DenseRetriever bằng Cross-Encoder, trả về top-N chính xác hơn.

Tại sao cần Reranker?
─────────────────────
Dense retrieval (FAISS + embedding) rất nhanh nhưng dùng "bi-encoder":
query và document được encode RIÊNG BIỆT, không thấy nhau trong attention.
→ Tốt cho recall (tìm đúng ứng viên) nhưng kém precision (xếp hạng chưa tốt).

Cross-Encoder khắc phục:
- Nhận (query, document) làm INPUT CHUNG, chạy full attention giữa chúng.
- Điểm số phản ánh quan hệ thực sự giữa câu hỏi và đoạn văn.
- Chậm hơn → chỉ chạy trên tập ứng viên nhỏ (10-20 chunks từ FAISS).

Ví dụ cụ thể:
  Query: "Điều kiện tiên quyết môn Machine Learning"
  Dense trả về 10 chunks → reranker chấm lại → giữ top 5 chính xác nhất.

Mô hình dùng: cross-encoder/ms-marco-MiniLM-L-6-v2
- ~22MB, chạy nhanh trên CPU.
- Được train trên MS MARCO passage ranking.
- Hiệu quả với tiếng Việt qua multilingual subword tokenizer.
- Thay bằng BAAI/bge-reranker-v2-m3 để tốt hơn với tiếng Việt (nặng hơn).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    RERANKER_MODEL_NAME,
    RERANKER_DEVICE,
    RERANKER_TOP_N,
    RERANKER_ENABLED,
    RERANKER_SCORE_THRESHOLD,
)
from src.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Reranker dùng Cross-Encoder để xếp hạng lại ứng viên từ dense retrieval.

    Luồng xử lý:
        list[dict] ứng viên (từ DenseRetriever)
            → tạo pairs [(query, chunk_text), ...]
            → CrossEncoder.predict() → list[float] logit scores
            → sort giảm dần + lọc threshold
            → trả về top_n chunks tốt nhất

    Attributes:
        _model:     CrossEncoder đã load.
        _top_n:     Số chunk giữ lại sau rerank.
        _threshold: Bỏ qua chunk có score < threshold.
        _enabled:   Nếu False, trả về input nguyên vẹn (bypass reranker).
    """

    def __init__(
        self,
        model_name:  str   = RERANKER_MODEL_NAME,
        device:      str   = RERANKER_DEVICE,
        top_n:       int   = RERANKER_TOP_N,
        threshold:   float = RERANKER_SCORE_THRESHOLD,
        enabled:     bool  = RERANKER_ENABLED,
    ) -> None:
        """
        Load Cross-Encoder model.

        Args:
            model_name: HuggingFace model ID của cross-encoder.
            device:     "cpu" hoặc "cuda".
            top_n:      Số chunk giữ lại sau rerank.
            threshold:  Score tối thiểu để giữ chunk (logit scale).
            enabled:    False = skip reranking (dùng khi ablation study).

        Raises:
            ImportError: Nếu sentence-transformers chưa cài.
        """
        self._top_n     = top_n
        self._threshold = threshold
        self._enabled   = enabled

        if not enabled:
            logger.info("Reranker DISABLED — sẽ bypass, dùng dense scores trực tiếp")
            self._model = None
            return

        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError:
            raise ImportError(
                "Cài đặt sentence-transformers: pip install sentence-transformers"
            )

        logger.info(f"Đang load Cross-Encoder: {model_name} (device={device})")
        self._model = CrossEncoder(model_name, device=device)
        logger.info(f"Cross-Encoder sẵn sàng | top_n={top_n} | threshold={threshold}")

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def rerank(
        self,
        query:      str,
        candidates: list[dict],
        top_n:      int | None = None,
    ) -> list[dict]:
        """
        Chấm điểm lại và xếp hạng ứng viên theo Cross-Encoder score.

        Args:
            query:      Câu hỏi gốc của người dùng.
            candidates: List chunk từ DenseRetriever (đã có key "score").
            top_n:      Override top_n mặc định.

        Returns:
            List[dict] — tập con của candidates, sắp xếp theo rerank_score,
            mỗi dict thêm key "rerank_score" (float logit).
            Nếu reranker bị disabled → trả về candidates[:top_n] nguyên vẹn.
        """
        if not candidates:
            return []

        n = top_n if top_n is not None else self._top_n

        # ── Bypass mode ──────────────────────────────────────
        if not self._enabled or self._model is None:
            logger.debug("Reranker bypass — dùng dense score")
            # Thêm rerank_score = dense score để interface nhất quán
            for c in candidates:
                c["rerank_score"] = c.get("score", 0.0)
            return candidates[:n]

        # ── Tạo input pairs cho Cross-Encoder ────────────────
        # Cross-Encoder nhận list of [query, text] pairs
        pairs = [[query, c["text"]] for c in candidates]

        logger.info(
            f"Reranking {len(candidates)} ứng viên cho query: "
            f"'{query[:50]}...'"
        )

        # ── Chạy Cross-Encoder ────────────────────────────────
        # predict() trả về numpy array of logit scores (không phải probability)
        scores: np.ndarray = self._model.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # ── Gắn rerank_score vào mỗi chunk ───────────────────
        scored_candidates = []
        for chunk, rerank_score in zip(candidates, scores):
            entry = {**chunk, "rerank_score": float(rerank_score)}
            scored_candidates.append(entry)

        # ── Sort giảm dần theo rerank_score ──────────────────
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # ── Lọc theo threshold ────────────────────────────────
        filtered = [
            c for c in scored_candidates
            if c["rerank_score"] >= self._threshold
        ]

        if not filtered:
            # Fallback: nếu tất cả dưới threshold → trả về top-1 để tránh trống
            logger.warning(
                f"Tất cả {len(scored_candidates)} candidates đều dưới threshold "
                f"({self._threshold}). Fallback về top-1."
            )
            filtered = scored_candidates[:1]

        result = filtered[:n]

        logger.info(
            f"Rerank xong | {len(candidates)} → {len(result)} chunks | "
            f"top rerank_score={result[0]['rerank_score']:.3f}"
        )
        return result

    # ─────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """Reranker có đang hoạt động không."""
        return self._enabled

    @property
    def top_n(self) -> int:
        """Số chunk giữ lại sau rerank."""
        return self._top_n

    def __repr__(self) -> str:
        if not self._enabled:
            return "CrossEncoderReranker(enabled=False)"
        return (
            f"CrossEncoderReranker("
            f"model='{RERANKER_MODEL_NAME}', "
            f"top_n={self._top_n}, "
            f"threshold={self._threshold})"
        )


# ─────────────────────────────────────────────────────────────
# Retrieval Result — dataclass-like dict schema (tài liệu hóa)
# ─────────────────────────────────────────────────────────────

def format_retrieval_result(chunks: list[dict]) -> list[dict]:
    """
    Chuẩn hóa output của retrieval pipeline thành schema nhất quán
    để Milestone 4 (LLM) và Milestone 5 (Evaluation) consume.

    Schema đầu ra mỗi dict:
        chunk_id      (str)   : ID duy nhất của chunk
        text          (str)   : Nội dung văn bản
        source        (str)   : Đường dẫn file gốc
        file_name     (str)   : Tên file
        chunk_index   (int)   : Thứ tự chunk trong tài liệu
        score         (float) : Dense retrieval score (cosine sim)
        rerank_score  (float) : Cross-encoder score (hoặc = score nếu bypass)
        rank          (int)   : Thứ hạng cuối (1 = tốt nhất)

    Args:
        chunks: List[dict] từ reranker.rerank() hoặc DenseRetriever.retrieve().

    Returns:
        List[dict] đã chuẩn hóa, thêm key "rank".
    """
    standardized = []
    for rank, chunk in enumerate(chunks, start=1):
        entry = {
            "rank":         rank,
            "chunk_id":     chunk.get("chunk_id", ""),
            "text":         chunk.get("text", ""),
            "source":       chunk.get("source", ""),
            "file_name":    chunk.get("file_name", ""),
            "chunk_index":  chunk.get("chunk_index", -1),
            "score":        chunk.get("score", 0.0),
            "rerank_score": chunk.get("rerank_score", chunk.get("score", 0.0)),
        }
        standardized.append(entry)
    return standardized
