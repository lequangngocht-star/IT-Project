"""
tests/test_milestone3.py
-------------------------
Kiểm thử Milestone 3: DenseRetriever, CrossEncoderReranker, format_retrieval_result.

Chiến lược:
- Mock EmbeddingModel và FaissVectorStore → test logic hoàn toàn không cần
  download model hay đọc file.
- Mock CrossEncoder → test reranker logic không cần model thật.
- Integration test có flag skip (cần model thật + FAISS index đã build).

Chạy:
    python -m pytest tests/test_milestone3.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────
# Fixtures & Mocks dùng chung
# ─────────────────────────────────────────────────────────────

DIM = 64

def _make_chunks(n: int = 10) -> list[dict]:
    """Tạo n chunk dict giả phục vụ test."""
    return [
        {
            "chunk_id":    f"doc_chunk_{i:04d}",
            "chunk_index": i,
            "text":        f"Đây là nội dung chunk thứ {i}. " * 8,
            "char_count":  200,
            "source":      f"/data/raw/doc_{i // 3}.pdf",
            "file_name":   f"doc_{i // 3}.pdf",
            "file_type":   ".pdf",
            "num_pages":   20,
            "score":       float(1.0 - i * 0.05),   # score giảm dần
        }
        for i in range(n)
    ]

def _normalized_vec(n: int = 1) -> np.ndarray:
    """Tạo vector đã normalize, shape (n, DIM)."""
    vecs = np.random.randn(n, DIM).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-8)

def _mock_embedder() -> MagicMock:
    """Mock EmbeddingModel — embed_query trả về random normalized vector."""
    m = MagicMock()
    m.dim = DIM
    m.embed_query.return_value = _normalized_vec(1)
    m.embed_queries.side_effect = lambda qs: _normalized_vec(len(qs))
    return m

def _mock_vector_store(chunks: list[dict]) -> MagicMock:
    """Mock FaissVectorStore — search trả về chunks đã cho."""
    m = MagicMock()
    type(m).size = PropertyMock(return_value=len(chunks))
    m.search.return_value = chunks
    return m


# ─────────────────────────────────────────────────────────────
# Test DenseRetriever
# ─────────────────────────────────────────────────────────────

class TestDenseRetriever(unittest.TestCase):

    def setUp(self):
        self.chunks   = _make_chunks(10)
        self.embedder = _mock_embedder()
        self.store    = _mock_vector_store(self.chunks)

        from src.retrieval.searcher import DenseRetriever
        self.retriever = DenseRetriever(
            embedder=self.embedder,
            vector_store=self.store,
            top_k=10,
        )

    # ── retrieve() ───────────────────────────────────────────
    def test_retrieve_calls_embed_query(self):
        self.retriever.retrieve("Câu hỏi test")
        self.embedder.embed_query.assert_called_once_with("Câu hỏi test")

    def test_retrieve_calls_faiss_search(self):
        self.retriever.retrieve("test query")
        self.store.search.assert_called_once()

    def test_retrieve_returns_list_of_dicts(self):
        results = self.retriever.retrieve("test")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], dict)

    def test_retrieve_empty_query_raises(self):
        from src.retrieval.searcher import DenseRetriever
        with self.assertRaises(ValueError):
            self.retriever.retrieve("")

    def test_retrieve_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve("   ")

    def test_retrieve_trims_query(self):
        """Câu hỏi có khoảng trắng đầu/cuối vẫn chạy bình thường."""
        results = self.retriever.retrieve("  test query  ")
        self.assertIsNotNone(results)
        # embed_query được gọi với query đã trim
        called_query = self.embedder.embed_query.call_args[0][0]
        self.assertEqual(called_query, "test query")

    def test_retrieve_top_k_override(self):
        self.retriever.retrieve("test", top_k=3)
        # search phải được gọi với top_k=3
        _, kwargs = self.store.search.call_args
        # Kiểm tra top_k được truyền vào search
        call_args = self.store.search.call_args
        passed_k = call_args[1].get("top_k") or call_args[0][1]
        self.assertEqual(passed_k, 3)

    # ── retrieve_batch() ──────────────────────────────────────
    def test_retrieve_batch_returns_list_of_lists(self):
        queries = ["Câu hỏi 1", "Câu hỏi 2", "Câu hỏi 3"]
        all_results = self.retriever.retrieve_batch(queries)
        self.assertEqual(len(all_results), 3)
        for result in all_results:
            self.assertIsInstance(result, list)

    def test_retrieve_batch_calls_retrieve_n_times(self):
        queries = ["q1", "q2", "q3"]
        with patch.object(self.retriever, "retrieve", wraps=self.retriever.retrieve) as mock_r:
            self.retriever.retrieve_batch(queries)
            self.assertEqual(mock_r.call_count, 3)

    # ── repr ──────────────────────────────────────────────────
    def test_repr(self):
        r = repr(self.retriever)
        self.assertIn("DenseRetriever", r)


# ─────────────────────────────────────────────────────────────
# Test CrossEncoderReranker (Bypass mode — không cần model)
# ─────────────────────────────────────────────────────────────

class TestCrossEncoderRerankerDisabled(unittest.TestCase):
    """Test reranker ở chế độ disabled (bypass) — không cần model thật."""

    def setUp(self):
        from src.retrieval.reranker import CrossEncoderReranker
        self.reranker = CrossEncoderReranker(enabled=False, top_n=3)
        self.chunks   = _make_chunks(8)

    def test_rerank_disabled_returns_top_n(self):
        results = self.reranker.rerank("query", self.chunks, top_n=3)
        self.assertEqual(len(results), 3)

    def test_rerank_disabled_adds_rerank_score(self):
        results = self.reranker.rerank("query", self.chunks)
        for r in results:
            self.assertIn("rerank_score", r)

    def test_rerank_disabled_rerank_score_equals_dense_score(self):
        results = self.reranker.rerank("query", self.chunks)
        for r in results:
            self.assertAlmostEqual(r["rerank_score"], r["score"], places=5)

    def test_rerank_empty_candidates_returns_empty(self):
        results = self.reranker.rerank("query", [])
        self.assertEqual(results, [])

    def test_rerank_top_n_override(self):
        results = self.reranker.rerank("query", self.chunks, top_n=2)
        self.assertEqual(len(results), 2)

    def test_rerank_top_n_larger_than_candidates(self):
        """top_n > len(candidates) phải trả về tất cả candidates."""
        results = self.reranker.rerank("query", self.chunks[:3], top_n=100)
        self.assertEqual(len(results), 3)

    def test_enabled_property_false(self):
        self.assertFalse(self.reranker.enabled)

    def test_repr_disabled(self):
        r = repr(self.reranker)
        self.assertIn("enabled=False", r)


# ─────────────────────────────────────────────────────────────
# Test CrossEncoderReranker (Mock model — test logic rerank)
# ─────────────────────────────────────────────────────────────

class TestCrossEncoderRerankerMocked(unittest.TestCase):
    """Test reranker với CrossEncoder được mock — test sorting và threshold logic."""

    def _make_reranker_with_mock(self, scores: list[float], top_n: int = 3, threshold: float = -10.0):
        """Tạo reranker với CrossEncoder mock trả về scores cố định."""
        from src.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._top_n     = top_n
        reranker._threshold = threshold
        reranker._enabled   = True

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(scores, dtype=np.float32)
        reranker._model = mock_model
        return reranker

    def test_rerank_sorts_by_score_descending(self):
        """Chunk với rerank_score cao nhất phải ở vị trí đầu tiên."""
        chunks = _make_chunks(4)
        # Scores không theo thứ tự: chunk[2] sẽ có score cao nhất (0.9)
        mock_scores = [0.1, 0.3, 0.9, 0.5]
        reranker = self._make_reranker_with_mock(mock_scores, top_n=4)

        results = reranker.rerank("query", chunks)
        # Dùng assertAlmostEqual vì numpy float32 → Python float có sai số nhỏ
        self.assertAlmostEqual(results[0]["rerank_score"], 0.9, places=5)
        self.assertAlmostEqual(results[1]["rerank_score"], 0.5, places=5)
        self.assertAlmostEqual(results[2]["rerank_score"], 0.3, places=5)

    def test_rerank_returns_top_n(self):
        chunks = _make_chunks(6)
        scores = [0.1, 0.5, 0.3, 0.8, 0.2, 0.7]
        reranker = self._make_reranker_with_mock(scores, top_n=3)
        results = reranker.rerank("query", chunks)
        self.assertEqual(len(results), 3)

    def test_rerank_filters_below_threshold(self):
        chunks = _make_chunks(4)
        # Threshold = 0.4 → chỉ giữ scores >= 0.4
        scores  = [-1.0, 0.5, 0.2, 0.8]
        reranker = self._make_reranker_with_mock(scores, top_n=4, threshold=0.4)
        results  = reranker.rerank("query", chunks)
        # Chỉ 2 chunks vượt ngưỡng (0.5 và 0.8)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertGreaterEqual(r["rerank_score"], 0.4)

    def test_rerank_fallback_when_all_below_threshold(self):
        """Nếu tất cả dưới threshold → trả về ít nhất 1 chunk (fallback)."""
        chunks = _make_chunks(3)
        scores  = [-5.0, -3.0, -4.0]
        reranker = self._make_reranker_with_mock(scores, top_n=3, threshold=100.0)
        results  = reranker.rerank("query", chunks)
        self.assertEqual(len(results), 1)  # fallback về top-1

    def test_rerank_empty_candidates(self):
        reranker = self._make_reranker_with_mock([], top_n=3)
        results  = reranker.rerank("query", [])
        self.assertEqual(results, [])

    def test_rerank_adds_rerank_score_key(self):
        chunks   = _make_chunks(3)
        scores   = [0.1, 0.5, 0.3]
        reranker = self._make_reranker_with_mock(scores, top_n=3)
        results  = reranker.rerank("query", chunks)
        for r in results:
            self.assertIn("rerank_score", r)

    def test_rerank_does_not_mutate_original_chunks(self):
        """rerank() không được thay đổi dict gốc trong candidates list."""
        chunks = _make_chunks(3)
        original_keys = set(chunks[0].keys())
        scores   = [0.3, 0.1, 0.5]
        reranker = self._make_reranker_with_mock(scores, top_n=3)
        reranker.rerank("query", chunks)
        # Dict gốc không có thêm key
        self.assertEqual(set(chunks[0].keys()), original_keys)

    def test_rerank_preserves_chunk_metadata(self):
        chunks = _make_chunks(3)
        scores = [0.3, 0.7, 0.5]
        reranker = self._make_reranker_with_mock(scores, top_n=3)
        results  = reranker.rerank("query", chunks)
        # Chunk tốt nhất (score=0.7) là chunk[1] → file_name = doc_0.pdf
        best = results[0]
        self.assertIn("chunk_id", best)
        self.assertIn("text", best)
        self.assertIn("file_name", best)


# ─────────────────────────────────────────────────────────────
# Test format_retrieval_result
# ─────────────────────────────────────────────────────────────

class TestFormatRetrievalResult(unittest.TestCase):

    def setUp(self):
        from src.retrieval.reranker import format_retrieval_result
        self.format = format_retrieval_result

    def test_adds_rank_field(self):
        chunks  = _make_chunks(3)
        results = self.format(chunks)
        for i, r in enumerate(results):
            self.assertEqual(r["rank"], i + 1)

    def test_rank_starts_at_1(self):
        chunks  = _make_chunks(2)
        results = self.format(chunks)
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[1]["rank"], 2)

    def test_output_has_required_keys(self):
        required = {"rank", "chunk_id", "text", "source", "file_name",
                    "chunk_index", "score", "rerank_score"}
        chunks  = _make_chunks(2)
        results = self.format(chunks)
        for r in results:
            self.assertTrue(required.issubset(set(r.keys())), f"Thiếu keys: {required - set(r.keys())}")

    def test_empty_input(self):
        results = self.format([])
        self.assertEqual(results, [])

    def test_rerank_score_defaults_to_score(self):
        """Nếu chunk không có rerank_score → dùng score làm fallback."""
        chunk   = _make_chunks(1)[0]
        chunk.pop("rerank_score", None)   # xóa nếu có
        chunk["score"] = 0.75
        results = self.format([chunk])
        self.assertAlmostEqual(results[0]["rerank_score"], 0.75)

    def test_preserves_text(self):
        chunks  = _make_chunks(1)
        results = self.format(chunks)
        self.assertEqual(results[0]["text"], chunks[0]["text"])

    def test_preserves_file_name(self):
        chunks  = _make_chunks(1)
        results = self.format(chunks)
        self.assertEqual(results[0]["file_name"], chunks[0]["file_name"])


# ─────────────────────────────────────────────────────────────
# Test run_query (pipeline M3 integration với mocks)
# ─────────────────────────────────────────────────────────────

class TestRunQuery(unittest.TestCase):

    def setUp(self):
        self.chunks   = _make_chunks(5)
        self.embedder = _mock_embedder()
        self.store    = _mock_vector_store(self.chunks)

        from src.retrieval.searcher import DenseRetriever
        from src.retrieval.reranker import CrossEncoderReranker

        self.retriever = DenseRetriever(
            embedder=self.embedder,
            vector_store=self.store,
            top_k=5,
        )
        self.reranker = CrossEncoderReranker(enabled=False, top_n=3)

    def test_run_query_returns_formatted_results(self):
        from src.pipeline_m3 import run_query
        results = run_query("Câu hỏi test", self.retriever, self.reranker)

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        # Mỗi result phải có key "rank"
        for r in results:
            self.assertIn("rank", r)

    def test_run_query_rank_is_sequential(self):
        from src.pipeline_m3 import run_query
        results = run_query("test", self.retriever, self.reranker)
        ranks = [r["rank"] for r in results]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_run_query_top_n_override(self):
        from src.pipeline_m3 import run_query
        results = run_query("test", self.retriever, self.reranker, top_n=2)
        self.assertLessEqual(len(results), 2)


# ─────────────────────────────────────────────────────────────
# Integration Test (skip — cần model thật + FAISS index)
# ─────────────────────────────────────────────────────────────

@unittest.skip(
    "Integration test — cần chạy M1+M2 trước và có model thật. "
    "Bỏ decorator để chạy thủ công."
)
class TestM3Integration(unittest.TestCase):
    """Chạy full pipeline M3 với model thật và FAISS index thật."""

    @classmethod
    def setUpClass(cls):
        from src.pipeline_m3 import build_retrieval_pipeline
        cls.retriever, cls.reranker = build_retrieval_pipeline(use_reranker=False)

    def test_retrieve_vietnamese_query(self):
        results = self.retriever.retrieve("Machine Learning là gì?")
        self.assertGreater(len(results), 0)
        self.assertIn("text", results[0])

    def test_retrieve_returns_relevant_chunks(self):
        results = self.retriever.retrieve("Điều kiện tiên quyết môn học")
        # Top result phải có score > 0.3 (ngưỡng "có liên quan")
        self.assertGreater(results[0]["score"], 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
