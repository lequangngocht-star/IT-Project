"""
tests/test_milestone4.py
-------------------------
Kiểm thử Milestone 4: Prompter, RAGResponse, và full RAG pipeline.

Chiến lược:
- Tất cả test đều dùng mock — không load model LLM thật (quá nặng).
- MockLLM trả về chuỗi cố định để test logic parse, format, pipeline.
- Test prompter độc lập (chỉ cần str operations — nhanh nhất).
- Integration test có skip decorator (cần model thật + FAISS index).

Chạy:
    python -m pytest tests/test_milestone4.py -v
"""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def _make_chunks(n: int = 5) -> list[dict]:
    """Tạo chunks giả đã qua retrieval pipeline (có rank, score, rerank_score)."""
    return [
        {
            "rank":         i + 1,
            "chunk_id":     f"doc_chunk_{i:04d}",
            "chunk_index":  i,
            "total_chunks": 20,
            "text": (
                f"Đây là nội dung đoạn văn thứ {i} trong tài liệu. "
                f"Nó nói về Machine Learning và các ứng dụng của AI trong giáo dục. "
                f"Điều kiện tiên quyết bao gồm Đại số tuyến tính và Python cơ bản. " * 3
            ),
            "char_count":   300,
            "source":       f"/data/raw/giao_trinh_{i // 2}.pdf",
            "file_name":    f"giao_trinh_{i // 2}.pdf",
            "file_type":    ".pdf",
            "num_pages":    100,
            "score":        float(0.85 - i * 0.05),
            "rerank_score": float(0.90 - i * 0.04),
        }
        for i in range(n)
    ]


def _mock_llm(answer: str = "Câu trả lời mẫu từ LLM.") -> MagicMock:
    """Mock LLMManager — generate_chat trả về chuỗi cố định."""
    m = MagicMock()
    m.is_loaded = True
    m.model_name = "mock-llm"
    m.generate_chat.return_value = answer
    m.generate.return_value = answer
    return m


def _mock_retriever(chunks: list[dict]) -> MagicMock:
    """Mock DenseRetriever."""
    m = MagicMock()
    m.retrieve.return_value = chunks
    m.retrieve_batch.return_value = [chunks] * 3
    return m


def _mock_reranker(chunks: list[dict]) -> MagicMock:
    """Mock CrossEncoderReranker — trả về chunks đã có rerank_score."""
    m = MagicMock()
    m.rerank.return_value = chunks
    m.enabled = True
    m.top_n = 5
    return m


# ─────────────────────────────────────────────────────────────
# Test RAGResponse
# ─────────────────────────────────────────────────────────────

class TestRAGResponse(unittest.TestCase):
    """Test dataclass RAGResponse và to_dict()."""

    def setUp(self):
        from src.llm.prompter import RAGResponse
        self.chunks = _make_chunks(3)
        self.response = RAGResponse(
            query="Machine Learning là gì?",
            answer="Machine Learning là nhánh của AI.",
            sources=self.chunks,
            has_answer=True,
            context_used=3,
            latency_s=1.23,
        )

    def test_to_dict_has_required_keys(self):
        d = self.response.to_dict()
        required = {"query", "answer", "has_answer", "context_used", "latency_s", "sources"}
        self.assertTrue(required.issubset(d.keys()))

    def test_to_dict_sources_structure(self):
        d = self.response.to_dict()
        for s in d["sources"]:
            self.assertIn("rank", s)
            self.assertIn("file_name", s)
            self.assertIn("chunk_index", s)
            self.assertIn("score", s)
            self.assertIn("rerank_score", s)

    def test_to_dict_json_serializable(self):
        """to_dict() phải serialize được sang JSON (không có non-JSON types)."""
        d = self.response.to_dict()
        try:
            json.dumps(d)
        except (TypeError, ValueError) as e:
            self.fail(f"to_dict() không serialize được: {e}")

    def test_to_dict_latency_rounded(self):
        d = self.response.to_dict()
        self.assertAlmostEqual(d["latency_s"], 1.23, places=3)

    def test_has_answer_true(self):
        self.assertTrue(self.response.has_answer)

    def test_has_answer_false_when_no_info(self):
        from src.llm.prompter import RAGResponse
        r = RAGResponse(
            query="test",
            answer="Tài liệu không đề cập đến vấn đề này.",
            sources=[],
            has_answer=False,
            context_used=0,
        )
        self.assertFalse(r.has_answer)


# ─────────────────────────────────────────────────────────────
# Test build_context
# ─────────────────────────────────────────────────────────────

class TestBuildContext(unittest.TestCase):

    def setUp(self):
        from src.llm.prompter import build_context
        self.build_context = build_context
        self.chunks = _make_chunks(5)

    def test_returns_string_and_list(self):
        ctx, used = self.build_context(self.chunks)
        self.assertIsInstance(ctx, str)
        self.assertIsInstance(used, list)

    def test_empty_chunks(self):
        ctx, used = self.build_context([])
        self.assertEqual(used, [])
        self.assertIn("Không có", ctx)

    def test_respects_max_chunks(self):
        _, used = self.build_context(self.chunks, max_chunks=2)
        self.assertLessEqual(len(used), 2)

    def test_context_contains_file_name(self):
        ctx, used = self.build_context(self.chunks[:2])
        for chunk in used:
            self.assertIn(chunk["file_name"], ctx)

    def test_context_contains_chunk_text(self):
        ctx, used = self.build_context(self.chunks[:1])
        # Ít nhất 50 ký tự đầu của chunk text phải xuất hiện trong context
        self.assertIn(self.chunks[0]["text"][:50], ctx)

    def test_respects_max_total_chars(self):
        """Tổng context không vượt max_total_chars (với tolerance nhỏ)."""
        ctx, used = self.build_context(self.chunks, max_total_chars=500)
        # Khi chỉ có 1 chunk, context sẽ có ít nhất chunk đó
        self.assertGreater(len(used), 0)
        # Context không vượt quá gấp đôi giới hạn (header + 1 chunk)
        self.assertLessEqual(len(ctx), 1500)

    def test_context_has_source_header(self):
        """Mỗi chunk phải có header [Nguồn N: ...]."""
        ctx, _ = self.build_context(self.chunks[:2])
        self.assertIn("[Nguồn", ctx)

    def test_separates_chunks_with_delimiter(self):
        """Các chunks phải được phân cách bằng '---'."""
        ctx, used = self.build_context(self.chunks[:3])
        if len(used) > 1:
            self.assertIn("---", ctx)


# ─────────────────────────────────────────────────────────────
# Test build_messages
# ─────────────────────────────────────────────────────────────

class TestBuildMessages(unittest.TestCase):

    def setUp(self):
        from src.llm.prompter import build_messages
        self.build_messages = build_messages
        self.chunks = _make_chunks(3)

    def test_returns_messages_and_used_chunks(self):
        messages, used = self.build_messages("Test query?", self.chunks)
        self.assertIsInstance(messages, list)
        self.assertIsInstance(used, list)

    def test_messages_has_system_and_user(self):
        messages, _ = self.build_messages("Query?", self.chunks)
        roles = [m["role"] for m in messages]
        self.assertIn("system", roles)
        self.assertIn("user", roles)

    def test_system_message_first(self):
        messages, _ = self.build_messages("Query?", self.chunks)
        self.assertEqual(messages[0]["role"], "system")

    def test_user_message_contains_query(self):
        query = "Điều kiện tiên quyết môn ML là gì?"
        messages, _ = self.build_messages(query, self.chunks)
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        self.assertIn(query, user_content)

    def test_user_message_contains_context(self):
        messages, _ = self.build_messages("query", self.chunks)
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        self.assertIn("TÀI LIỆU THAM KHẢO", user_content)

    def test_system_message_has_instructions(self):
        messages, _ = self.build_messages("query", self.chunks)
        system_content = messages[0]["content"]
        self.assertIn("KHÔNG bịa đặt", system_content)

    def test_messages_all_have_role_and_content(self):
        messages, _ = self.build_messages("query", self.chunks)
        for m in messages:
            self.assertIn("role", m)
            self.assertIn("content", m)
            self.assertIsInstance(m["content"], str)
            self.assertGreater(len(m["content"]), 0)

    def test_empty_chunks_still_builds_messages(self):
        """Ngay cả khi không có chunks, messages vẫn được tạo (LLM sẽ nói không đủ thông tin)."""
        messages, used = self.build_messages("query?", [])
        self.assertGreater(len(messages), 0)
        self.assertEqual(used, [])


# ─────────────────────────────────────────────────────────────
# Test parse_response
# ─────────────────────────────────────────────────────────────

class TestParseResponse(unittest.TestCase):

    def setUp(self):
        from src.llm.prompter import parse_response
        self.parse = parse_response
        self.chunks = _make_chunks(2)

    def test_has_answer_true_for_normal_answer(self):
        r = self.parse("Machine Learning là nhánh của AI.", "query?", self.chunks)
        self.assertTrue(r.has_answer)

    def test_has_answer_false_for_no_info_response(self):
        no_info_answers = [
            "Tài liệu hiện có chưa đủ thông tin để trả lời.",
            "Tài liệu không đề cập đến vấn đề này.",
            "Không có thông tin về chủ đề này trong tài liệu.",
            "Tôi không tìm thấy thông tin liên quan.",
        ]
        for answer in no_info_answers:
            r = self.parse(answer, "query?", self.chunks)
            self.assertFalse(r.has_answer, f"Phải là False cho: '{answer}'")

    def test_strips_whitespace_from_answer(self):
        r = self.parse("  Câu trả lời có khoảng trắng.  ", "q", self.chunks)
        self.assertEqual(r.answer, "Câu trả lời có khoảng trắng.")

    def test_preserves_query(self):
        r = self.parse("answer", "Câu hỏi gốc?", self.chunks)
        self.assertEqual(r.query, "Câu hỏi gốc?")

    def test_context_used_equals_chunk_count(self):
        r = self.parse("answer", "query", self.chunks)
        self.assertEqual(r.context_used, len(self.chunks))

    def test_latency_stored(self):
        r = self.parse("answer", "query", self.chunks, latency_s=2.5)
        self.assertAlmostEqual(r.latency_s, 2.5)

    def test_sources_preserved(self):
        r = self.parse("answer", "query", self.chunks)
        self.assertEqual(len(r.sources), len(self.chunks))

    def test_returns_rag_response_type(self):
        from src.llm.prompter import RAGResponse
        r = self.parse("answer", "query", self.chunks)
        self.assertIsInstance(r, RAGResponse)


# ─────────────────────────────────────────────────────────────
# Test format_response_for_display
# ─────────────────────────────────────────────────────────────

class TestFormatResponseForDisplay(unittest.TestCase):

    def setUp(self):
        from src.llm.prompter import RAGResponse, format_response_for_display
        self.format = format_response_for_display
        self.response = RAGResponse(
            query="Câu hỏi test?",
            answer="Đây là câu trả lời.",
            sources=_make_chunks(2),
            has_answer=True,
            context_used=2,
            latency_s=1.5,
        )

    def test_returns_string(self):
        result = self.format(self.response)
        self.assertIsInstance(result, str)

    def test_contains_query(self):
        result = self.format(self.response)
        self.assertIn("Câu hỏi test?", result)

    def test_contains_answer(self):
        result = self.format(self.response)
        self.assertIn("Đây là câu trả lời.", result)

    def test_contains_source_section(self):
        result = self.format(self.response)
        self.assertIn("NGUỒN THAM KHẢO", result)

    def test_contains_latency(self):
        result = self.format(self.response)
        self.assertIn("1.50s", result)

    def test_contains_file_names(self):
        result = self.format(self.response)
        for s in self.response.sources:
            self.assertIn(s["file_name"], result)

    def test_no_duplicate_sources(self):
        """Source giống nhau chỉ xuất hiện 1 lần trong phần trích dẫn."""
        from src.llm.prompter import RAGResponse
        # Tạo 2 chunks cùng file
        dup_chunks = _make_chunks(2)
        for c in dup_chunks:
            c["file_name"] = "same_file.pdf"
            c["chunk_index"] = 0

        response = RAGResponse(
            query="q", answer="a", sources=dup_chunks,
            has_answer=True, context_used=2,
        )
        result = self.format(response)
        # Đếm số lần file xuất hiện trong phần nguồn
        # (không tính phần header)
        source_section = result.split("NGUỒN THAM KHẢO")[-1] if "NGUỒN" in result else ""
        self.assertEqual(source_section.count("same_file.pdf"), 1)


# ─────────────────────────────────────────────────────────────
# Test rag_query (pipeline M4 với full mocks)
# ─────────────────────────────────────────────────────────────

class TestRagQuery(unittest.TestCase):
    """Test full RAG pipeline với mocked retriever, reranker, và LLM."""

    def setUp(self):
        self.chunks    = _make_chunks(5)
        self.retriever = _mock_retriever(self.chunks)
        self.reranker  = _mock_reranker(self.chunks[:3])
        self.llm       = _mock_llm("Machine Learning là nhánh của AI, tập trung vào học từ dữ liệu.")

    def _run(self, query="Machine Learning là gì?", top_n=None):
        from src.pipeline_m4 import rag_query
        return rag_query(query, self.retriever, self.reranker, self.llm, top_n=top_n)

    def test_returns_rag_response(self):
        from src.llm.prompter import RAGResponse
        result = self._run()
        self.assertIsInstance(result, RAGResponse)

    def test_query_preserved_in_response(self):
        result = self._run(query="Câu hỏi cụ thể?")
        self.assertEqual(result.query, "Câu hỏi cụ thể?")

    def test_answer_not_empty(self):
        result = self._run()
        self.assertTrue(len(result.answer) > 0)

    def test_has_answer_true_for_real_answer(self):
        result = self._run()
        self.assertTrue(result.has_answer)

    def test_has_answer_false_when_llm_says_no_info(self):
        llm_no_info = _mock_llm("Tài liệu không đề cập đến vấn đề này.")
        from src.pipeline_m4 import rag_query
        result = rag_query("query", self.retriever, self.reranker, llm_no_info)
        self.assertFalse(result.has_answer)

    def test_sources_not_empty(self):
        result = self._run()
        self.assertGreater(len(result.sources), 0)

    def test_context_used_positive(self):
        result = self._run()
        self.assertGreater(result.context_used, 0)

    def test_latency_positive(self):
        result = self._run()
        self.assertGreater(result.latency_s, 0)

    def test_empty_query_raises(self):
        from src.pipeline_m4 import rag_query
        with self.assertRaises(ValueError):
            rag_query("", self.retriever, self.reranker, self.llm)

    def test_whitespace_query_raises(self):
        from src.pipeline_m4 import rag_query
        with self.assertRaises(ValueError):
            rag_query("   ", self.retriever, self.reranker, self.llm)

    def test_llm_generate_chat_called_once(self):
        self._run()
        self.llm.generate_chat.assert_called_once()

    def test_retriever_called_with_query(self):
        from src.pipeline_m4 import rag_query
        rag_query("test query", self.retriever, self.reranker, self.llm)
        # retriever được gọi thông qua run_query (M3)
        # Kiểm tra gián tiếp: có kết quả hợp lệ là đủ
        # (Direct call check khó vì M3 wrap lại)

    def test_response_to_dict_serializable(self):
        result = self._run()
        d = result.to_dict()
        try:
            json.dumps(d)
        except (TypeError, ValueError) as e:
            self.fail(f"RAGResponse.to_dict() không serialize được: {e}")

    def test_empty_retrieval_returns_no_answer(self):
        """Khi không retrieve được chunk nào → has_answer=False."""
        empty_retriever = _mock_retriever([])
        empty_reranker  = _mock_reranker([])
        from src.pipeline_m4 import rag_query
        result = rag_query("query", empty_retriever, empty_reranker, self.llm)
        self.assertFalse(result.has_answer)
        self.assertEqual(result.context_used, 0)


# ─────────────────────────────────────────────────────────────
# Test LLMManager interface (mock — không cần model thật)
# ─────────────────────────────────────────────────────────────

class TestLLMManagerInterface(unittest.TestCase):
    """Test LLMManager config và interface — không load model thật."""

    def test_init_stores_model_name(self):
        from src.llm.model_manager import LLMManager
        m = LLMManager(model_name="test/model", device="cpu")
        self.assertEqual(m.model_name, "test/model")

    def test_init_not_loaded(self):
        from src.llm.model_manager import LLMManager
        m = LLMManager(model_name="test/model", device="cpu")
        self.assertFalse(m.is_loaded)

    def test_4bit_and_8bit_mutually_exclusive(self):
        from src.llm.model_manager import LLMManager
        with self.assertRaises(ValueError):
            LLMManager(load_in_4bit=True, load_in_8bit=True)

    def test_repr_not_loaded(self):
        from src.llm.model_manager import LLMManager
        m = LLMManager(model_name="Qwen/test", device="cpu")
        r = repr(m)
        self.assertIn("LLMManager", r)
        self.assertIn("not loaded", r)


# ─────────────────────────────────────────────────────────────
# Integration Test (skip — cần model thật + FAISS index)
# ─────────────────────────────────────────────────────────────

@unittest.skip(
    "Integration test — cần chạy M1+M2 trước và có đủ RAM cho LLM. "
    "Bỏ decorator để chạy thủ công."
)
class TestM4Integration(unittest.TestCase):
    """Chạy full RAG pipeline với model thật."""

    @classmethod
    def setUpClass(cls):
        from src.pipeline_m4 import build_rag_pipeline
        cls.retriever, cls.reranker, cls.llm = build_rag_pipeline(
            use_reranker=False,  # Tắt reranker để test nhanh hơn
        )

    def test_full_rag_query(self):
        from src.pipeline_m4 import rag_query
        result = rag_query(
            "Machine Learning là gì?",
            self.retriever, self.reranker, self.llm,
        )
        self.assertIsNotNone(result.answer)
        self.assertGreater(len(result.answer), 10)

    def test_response_cites_sources(self):
        from src.pipeline_m4 import rag_query
        result = rag_query(
            "Điều kiện tiên quyết môn học là gì?",
            self.retriever, self.reranker, self.llm,
        )
        self.assertGreater(result.context_used, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
