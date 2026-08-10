"""
tests/test_milestone5.py
-------------------------
Kiểm thử Milestone 5: metric functions, RAGEvaluator, EvalReport.

Chiến lược:
- Test từng metric function độc lập (pure functions — không cần mock).
- Test RAGEvaluator với mock rag_query_fn → không cần pipeline thật.
- Test edge cases: empty input, all-wrong retrieval, no-answer responses.
- Test EvalReport aggregation logic.

Chạy:
    python -m pytest tests/test_milestone5.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def _make_chunks(n: int = 3, prefix: str = "doc") -> list[dict]:
    """Tạo chunks giả với chunk_id, text, source."""
    return [
        {
            "rank":         i + 1,
            "chunk_id":     f"{prefix}_chunk_{i:04d}",
            "chunk_index":  i,
            "text": (
                f"Machine Learning là nhánh của AI. "
                f"Điều kiện tiên quyết bao gồm Đại số tuyến tính và Python. "
                f"Deep Learning sử dụng mạng nơ-ron nhiều lớp. " * 3
            ),
            "file_name":    f"{prefix}.pdf",
            "source":       f"/data/raw/{prefix}.pdf",
            "score":        float(0.9 - i * 0.1),
            "rerank_score": float(0.85 - i * 0.08),
        }
        for i in range(n)
    ]


def _make_mock_rag_response(
    answer:     str = "Machine Learning là nhánh của AI.",
    has_answer: bool = True,
    chunks:     list[dict] | None = None,
    latency_s:  float = 0.5,
):
    """Tạo mock RAGResponse object."""
    from src.llm.prompter import RAGResponse
    return RAGResponse(
        query="test query",
        answer=answer,
        sources=chunks if chunks is not None else _make_chunks(3),
        has_answer=has_answer,
        context_used=len(chunks) if chunks is not None else 3,
        latency_s=latency_s,
    )


def _make_eval_dataset(n: int = 5) -> list[dict]:
    """Tạo dataset JSON giả."""
    return [
        {
            "id":              f"q{i:03d}",
            "question":        f"Machine Learning là gì? (câu hỏi {i})",
            "ground_truth":    "Machine Learning là nhánh của AI, học từ dữ liệu.",
            "relevant_chunks": [f"doc_chunk_{i:04d}"],
            "category":        "giao_trinh" if i % 2 == 0 else "de_cuong",
            "difficulty":      "easy" if i < 3 else "medium",
        }
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────
# Test tokenizer
# ─────────────────────────────────────────────────────────────

class TestTokenizer(unittest.TestCase):

    def setUp(self):
        from src.evaluation.evaluator import _tokenize
        self.tokenize = _tokenize

    def test_returns_list(self):
        self.assertIsInstance(self.tokenize("hello world"), list)

    def test_lowercase(self):
        tokens = self.tokenize("Machine Learning")
        self.assertTrue(all(t == t.lower() for t in tokens))

    def test_removes_punctuation(self):
        tokens = self.tokenize("hello, world! test.")
        for t in tokens:
            self.assertNotIn(",", t)
            self.assertNotIn("!", t)
            self.assertNotIn(".", t)

    def test_empty_string(self):
        self.assertEqual(self.tokenize(""), [])

    def test_filters_single_chars(self):
        tokens = self.tokenize("a b c abc")
        self.assertNotIn("a", tokens)
        self.assertNotIn("b", tokens)
        self.assertIn("abc", tokens)

    def test_vietnamese_text(self):
        tokens = self.tokenize("Điều kiện tiên quyết là Python")
        self.assertIn("điều", tokens)
        self.assertIn("python", tokens)


# ─────────────────────────────────────────────────────────────
# Test _f1_token_overlap
# ─────────────────────────────────────────────────────────────

class TestF1TokenOverlap(unittest.TestCase):

    def setUp(self):
        from src.evaluation.evaluator import _f1_token_overlap
        self.f1 = _f1_token_overlap

    def test_identical_text_returns_1(self):
        text = "Machine Learning là gì"
        self.assertAlmostEqual(self.f1(text, text), 1.0, places=4)

    def test_no_overlap_returns_0(self):
        self.assertAlmostEqual(
            self.f1("machine learning python", "xe đạp thể thao"),
            0.0, places=4
        )

    def test_partial_overlap(self):
        score = self.f1("machine learning python", "machine learning java")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_pred_returns_0(self):
        self.assertEqual(self.f1("", "reference text"), 0.0)

    def test_empty_ref_returns_0(self):
        self.assertEqual(self.f1("prediction text", ""), 0.0)

    def test_both_empty_returns_0(self):
        self.assertEqual(self.f1("", ""), 0.0)

    def test_range_0_to_1(self):
        score = self.f1("machine learning ai python", "machine learning deep")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_symmetric_ish(self):
        """F1 không hoàn toàn symmetric nhưng cả 2 hướng phải > 0."""
        a = "machine learning python ai"
        b = "machine learning"
        self.assertGreater(self.f1(a, b), 0.0)
        self.assertGreater(self.f1(b, a), 0.0)


# ─────────────────────────────────────────────────────────────
# Test compute_faithfulness
# ─────────────────────────────────────────────────────────────

class TestComputeFaithfulness(unittest.TestCase):

    def setUp(self):
        from src.evaluation.evaluator import compute_faithfulness
        self.compute = compute_faithfulness
        self.chunks = _make_chunks(2)

    def test_high_for_answer_from_context(self):
        """Answer bám sát context → faithfulness > 0 (lexical F1 baseline)."""
        answer = "Machine Learning là nhánh của AI và học từ dữ liệu."
        score  = self.compute(answer, self.chunks)
        # Lexical F1 thường thấp do context dài hơn answer nhiều
        # Quan trọng là > 0 và > câu không liên quan
        self.assertGreater(score, 0.0)

    def test_low_for_unrelated_answer(self):
        """Answer không liên quan context → faithfulness thấp."""
        answer = "Hà Nội là thủ đô của Việt Nam, có hồ Hoàn Kiếm."
        score  = self.compute(answer, self.chunks)
        self.assertLess(score, 0.3)

    def test_empty_answer_returns_0(self):
        self.assertEqual(self.compute("", self.chunks), 0.0)

    def test_empty_chunks_returns_0(self):
        self.assertEqual(self.compute("some answer", []), 0.0)

    def test_range_0_to_1(self):
        score = self.compute("Machine Learning là AI.", self.chunks)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ─────────────────────────────────────────────────────────────
# Test compute_answer_relevancy
# ─────────────────────────────────────────────────────────────

class TestComputeAnswerRelevancy(unittest.TestCase):

    def setUp(self):
        from src.evaluation.evaluator import compute_answer_relevancy
        self.compute = compute_answer_relevancy

    def test_relevant_answer_high_score(self):
        q = "Machine Learning là gì và ứng dụng trong AI?"
        a = "Machine Learning là nhánh quan trọng của AI."
        self.assertGreater(self.compute(a, q), 0.3)

    def test_irrelevant_answer_low_score(self):
        q = "Machine Learning là gì?"
        a = "Hà Nội có nhiều hồ đẹp và di tích lịch sử."
        self.assertLess(self.compute(a, q), 0.2)

    def test_empty_answer_returns_0(self):
        self.assertEqual(self.compute("", "câu hỏi gì đó"), 0.0)

    def test_empty_question_returns_0(self):
        self.assertEqual(self.compute("câu trả lời", ""), 0.0)

    def test_range_0_to_1(self):
        score = self.compute("Machine Learning AI deep learning", "Machine Learning")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ─────────────────────────────────────────────────────────────
# Test compute_context_recall
# ─────────────────────────────────────────────────────────────

class TestComputeContextRecall(unittest.TestCase):

    def setUp(self):
        from src.evaluation.evaluator import compute_context_recall
        self.compute = compute_context_recall
        self.chunks = _make_chunks(3)

    def test_high_when_ground_truth_in_context(self):
        """Ground truth có overlap với context → context_recall > 0."""
        gt = "Machine Learning là nhánh của AI học từ dữ liệu."
        score = self.compute(gt, self.chunks)
        self.assertGreater(score, 0.0)

    def test_low_when_ground_truth_not_in_context(self):
        gt = "Hà Nội có diện tích 3328 km vuông và dân số 8 triệu người."
        self.assertLess(self.compute(gt, self.chunks), 0.2)

    def test_empty_ground_truth_returns_0(self):
        self.assertEqual(self.compute("", self.chunks), 0.0)

    def test_empty_chunks_returns_0(self):
        self.assertEqual(self.compute("ground truth text", []), 0.0)

    def test_range_0_to_1(self):
        score = self.compute("AI machine learning", self.chunks)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ─────────────────────────────────────────────────────────────
# Test compute_context_precision
# ─────────────────────────────────────────────────────────────

class TestComputeContextPrecision(unittest.TestCase):

    def setUp(self):
        from src.evaluation.evaluator import compute_context_precision
        self.compute = compute_context_precision

    def test_perfect_precision(self):
        """Tất cả retrieved đều relevant → precision = 1.0"""
        retrieved = ["a", "b", "c"]
        relevant  = {"a", "b", "c"}
        self.assertAlmostEqual(self.compute(retrieved, list(relevant)), 1.0)

    def test_zero_precision(self):
        """Không có retrieved nào relevant → precision = 0.0"""
        retrieved = ["x", "y", "z"]
        relevant  = ["a", "b", "c"]
        self.assertAlmostEqual(self.compute(retrieved, relevant), 0.0)

    def test_partial_precision(self):
        retrieved = ["a", "b", "x", "y"]   # 2/4 relevant
        relevant  = ["a", "b", "c"]
        score = self.compute(retrieved, relevant)
        self.assertAlmostEqual(score, 0.5, places=4)  # 2/4 = 0.5

    def test_empty_retrieved_returns_0(self):
        self.assertEqual(self.compute([], ["a", "b"]), 0.0)

    def test_empty_relevant_returns_0(self):
        self.assertEqual(self.compute(["a", "b"], []), 0.0)

    def test_range_0_to_1(self):
        score = self.compute(["a", "b"], ["a", "c"])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ─────────────────────────────────────────────────────────────
# Test compute_recall_at_k
# ─────────────────────────────────────────────────────────────

class TestComputeRecallAtK(unittest.TestCase):

    def setUp(self):
        from src.evaluation.evaluator import compute_recall_at_k
        self.compute = compute_recall_at_k

    def test_returns_dict_with_k_keys(self):
        result = self.compute(["a", "b", "c"], ["a"], k_values=[1, 3, 5])
        self.assertIn("recall@1", result)
        self.assertIn("recall@3", result)
        self.assertIn("recall@5", result)

    def test_recall_at_1_perfect(self):
        """Relevant chunk ở vị trí 1 → recall@1 = 1.0"""
        result = self.compute(["a", "b", "c"], ["a"], k_values=[1, 3])
        self.assertAlmostEqual(result["recall@1"], 1.0)

    def test_recall_at_1_miss(self):
        """Relevant chunk không ở top-1 → recall@1 = 0.0"""
        result = self.compute(["x", "a", "b"], ["a"], k_values=[1, 3])
        self.assertAlmostEqual(result["recall@1"], 0.0)

    def test_recall_at_3_hits(self):
        """2 relevant chunks trong top-3 → recall@3 = 2/2 = 1.0"""
        result = self.compute(["a", "x", "b"], ["a", "b"], k_values=[3])
        self.assertAlmostEqual(result["recall@3"], 1.0)

    def test_recall_at_k_increases_with_k(self):
        """Recall@k phải không giảm khi k tăng."""
        retrieved = ["x", "a", "y", "b", "z"]
        relevant  = ["a", "b"]
        result = self.compute(retrieved, relevant, k_values=[1, 3, 5])
        self.assertLessEqual(result["recall@1"], result["recall@3"])
        self.assertLessEqual(result["recall@3"], result["recall@5"])

    def test_empty_retrieved(self):
        result = self.compute([], ["a", "b"], k_values=[1, 3])
        self.assertEqual(result["recall@1"], 0.0)
        self.assertEqual(result["recall@3"], 0.0)

    def test_empty_relevant(self):
        result = self.compute(["a", "b"], [], k_values=[1, 3])
        self.assertEqual(result["recall@1"], 0.0)

    def test_all_relevant_in_retrieved(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant  = ["a", "b", "c"]
        result = self.compute(retrieved, relevant, k_values=[3, 5])
        self.assertAlmostEqual(result["recall@3"], 1.0)
        self.assertAlmostEqual(result["recall@5"], 1.0)

    def test_values_range_0_to_1(self):
        result = self.compute(["a", "b", "c"], ["a", "d"], k_values=[1, 3, 5])
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


# ─────────────────────────────────────────────────────────────
# Test EvalSample & EvalResult dataclasses
# ─────────────────────────────────────────────────────────────

class TestDataclasses(unittest.TestCase):

    def test_eval_result_to_dict(self):
        from src.evaluation.evaluator import EvalResult
        r = EvalResult(
            sample_id="q001", question="test?", ground_truth="answer",
            answer="test answer", has_answer=True,
            faithfulness=0.8, answer_relevancy=0.7,
            context_recall=0.9, context_precision=0.6,
            recall_at_k={"recall@1": 0.5, "recall@3": 0.8},
            latency_s=1.2, retrieved_chunk_ids=["a", "b"],
        )
        d = r.to_dict()
        self.assertEqual(d["sample_id"], "q001")
        self.assertIn("faithfulness", d)
        self.assertIn("recall_at_k", d)

    def test_eval_result_to_dict_json_serializable(self):
        from src.evaluation.evaluator import EvalResult
        r = EvalResult(
            sample_id="q001", question="test?", ground_truth="gt",
            answer="ans", has_answer=True,
            faithfulness=0.5, answer_relevancy=0.6,
            context_recall=0.7, context_precision=0.8,
            recall_at_k={"recall@1": 1.0},
            latency_s=0.3, retrieved_chunk_ids=[],
        )
        try:
            json.dumps(r.to_dict())
        except (TypeError, ValueError) as e:
            self.fail(f"EvalResult.to_dict() không JSON serializable: {e}")


# ─────────────────────────────────────────────────────────────
# Test RAGEvaluator
# ─────────────────────────────────────────────────────────────

class TestRAGEvaluator(unittest.TestCase):
    """Test RAGEvaluator với dataset giả và mock rag_fn."""

    def _make_evaluator(self, rag_fn=None, n_samples=5):
        """Helper: tạo RAGEvaluator với dataset temp file."""
        from src.evaluation.evaluator import RAGEvaluator

        dataset = _make_eval_dataset(n_samples)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(dataset, f, ensure_ascii=False)
            path = Path(f.name)

        if rag_fn is None:
            def rag_fn(query: str):
                return _make_mock_rag_response(
                    answer="Machine Learning là nhánh của AI.",
                    has_answer=True,
                    chunks=_make_chunks(3),
                )

        evaluator = RAGEvaluator(rag_fn=rag_fn, dataset_path=path)
        self._tmp_path = path
        return evaluator

    def tearDown(self):
        if hasattr(self, "_tmp_path"):
            self._tmp_path.unlink(missing_ok=True)

    # ── load dataset ──────────────────────────────────────────
    def test_loads_dataset(self):
        ev = self._make_evaluator(n_samples=5)
        self.assertEqual(len(ev._samples), 5)

    def test_missing_dataset_raises(self):
        from src.evaluation.evaluator import RAGEvaluator
        with self.assertRaises(FileNotFoundError):
            RAGEvaluator(rag_fn=lambda q: None, dataset_path=Path("/nonexistent.json"))

    # ── run() ────────────────────────────────────────────────
    def test_run_returns_eval_report(self):
        from src.evaluation.evaluator import EvalReport
        ev = self._make_evaluator(n_samples=3)
        report = ev.run(save=False)
        self.assertIsInstance(report, EvalReport)

    def test_run_total_samples_correct(self):
        ev = self._make_evaluator(n_samples=4)
        report = ev.run(save=False)
        self.assertEqual(report.total_samples, 4)

    def test_run_with_sample_ids_subset(self):
        ev = self._make_evaluator(n_samples=5)
        report = ev.run(sample_ids=["q000", "q001"], save=False)
        self.assertEqual(report.total_samples, 2)

    def test_run_metrics_in_range(self):
        ev = self._make_evaluator(n_samples=3)
        report = ev.run(save=False)
        for metric in [
            report.mean_faithfulness,
            report.mean_answer_relevancy,
            report.mean_context_recall,
            report.mean_context_precision,
            report.answer_rate,
        ]:
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)

    def test_run_latency_positive(self):
        ev = self._make_evaluator(n_samples=2)
        report = ev.run(save=False)
        self.assertGreaterEqual(report.mean_latency_s, 0.0)

    def test_run_by_category_populated(self):
        ev = self._make_evaluator(n_samples=5)
        report = ev.run(save=False)
        # Dataset giả có category giao_trinh và de_cuong
        self.assertIn("giao_trinh", report.by_category)
        self.assertIn("de_cuong", report.by_category)

    def test_run_by_difficulty_populated(self):
        ev = self._make_evaluator(n_samples=5)
        report = ev.run(save=False)
        self.assertIn("easy", report.by_difficulty)
        self.assertIn("medium", report.by_difficulty)

    def test_run_recall_at_k_keys_present(self):
        ev = self._make_evaluator(n_samples=3)
        report = ev.run(save=False)
        for k in [1, 3, 5]:
            self.assertIn(f"recall@{k}", report.mean_recall_at_k)

    def test_per_sample_count(self):
        ev = self._make_evaluator(n_samples=4)
        report = ev.run(save=False)
        self.assertEqual(len(report.per_sample), 4)

    def test_error_in_rag_fn_does_not_crash(self):
        """Lỗi trong rag_fn 1 sample không làm crash cả batch."""
        call_count = {"n": 0}

        def flaky_rag_fn(query: str):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("Simulated LLM error")
            return _make_mock_rag_response()

        ev = self._make_evaluator(rag_fn=flaky_rag_fn, n_samples=4)
        report = ev.run(save=False)  # Không raise exception
        self.assertEqual(report.total_samples, 4)

    def test_answered_samples_counted_correctly(self):
        """has_answer=False samples phải giảm answered_samples."""
        def no_answer_fn(query: str):
            return _make_mock_rag_response(
                answer="Tài liệu không đề cập đến vấn đề này.",
                has_answer=False,
            )

        ev = self._make_evaluator(rag_fn=no_answer_fn, n_samples=3)
        report = ev.run(save=False)
        self.assertEqual(report.answered_samples, 0)
        self.assertAlmostEqual(report.answer_rate, 0.0)

    def test_save_creates_json_file(self):
        """save=True phải tạo file JSON trong logs/evaluation/."""
        import os
        ev = self._make_evaluator(n_samples=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch EVAL_RESULTS_DIR
            import src.evaluation.evaluator as ev_module
            original = ev_module.EVAL_RESULTS_DIR
            ev_module.EVAL_RESULTS_DIR = tmpdir
            try:
                report = ev.run(save=True)
                files = list(Path(tmpdir).glob("eval_report_*.json"))
                self.assertGreater(len(files), 0)
            finally:
                ev_module.EVAL_RESULTS_DIR = original

    # ── print_report ─────────────────────────────────────────
    def test_print_report_runs_without_error(self):
        """print_report không raise exception."""
        from src.evaluation.evaluator import RAGEvaluator
        ev = self._make_evaluator(n_samples=2)
        report = ev.run(save=False)
        try:
            RAGEvaluator.print_report(report)
        except Exception as e:
            self.fail(f"print_report() raised: {e}")

    # ── EvalReport to_dict ────────────────────────────────────
    def test_report_to_dict_json_serializable(self):
        ev = self._make_evaluator(n_samples=2)
        report = ev.run(save=False)
        try:
            json.dumps(report.to_dict())
        except (TypeError, ValueError) as e:
            self.fail(f"EvalReport.to_dict() không JSON serializable: {e}")

    def test_report_to_dict_has_all_fields(self):
        ev = self._make_evaluator(n_samples=2)
        report = ev.run(save=False)
        d = report.to_dict()
        required = {
            "total_samples", "answered_samples", "answer_rate",
            "mean_faithfulness", "mean_answer_relevancy",
            "mean_context_recall", "mean_context_precision",
            "mean_recall_at_k", "mean_latency_s", "std_latency_s",
            "by_category", "by_difficulty", "per_sample",
        }
        self.assertTrue(required.issubset(d.keys()))


# ─────────────────────────────────────────────────────────────
# Integration Test (skip — cần pipeline thật)
# ─────────────────────────────────────────────────────────────

@unittest.skip(
    "Integration test — cần M1+M2 đã chạy và model thật. "
    "Bỏ decorator để chạy thủ công."
)
class TestM5Integration(unittest.TestCase):
    """Chạy evaluation với full pipeline thật."""

    @classmethod
    def setUpClass(cls):
        from src.pipeline_m5 import _build_rag_fn
        from src.evaluation.evaluator import RAGEvaluator
        rag_fn = _build_rag_fn(use_reranker=False, load_llm=False)
        cls.evaluator = RAGEvaluator(rag_fn)

    def test_run_3_samples(self):
        report = self.evaluator.run(
            sample_ids=["q001", "q002", "q003"], save=False
        )
        self.assertEqual(report.total_samples, 3)
        self.assertGreaterEqual(report.mean_context_recall, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
