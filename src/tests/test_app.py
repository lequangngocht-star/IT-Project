"""
tests/test_app.py
------------------
Kiểm thử FastAPI endpoints bằng TestClient — không cần server thật.

Chiến lược:
- Patch toàn bộ ML dependencies (model, FAISS) bằng mock.
- TestClient gọi endpoints trực tiếp trong process.
- Kiểm thử: status codes, response schema, error handling, validation.

Chạy:
    python -m pytest tests/test_app.py -v
"""

from __future__ import annotations

import json
import sys
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_mock_rag_response(
    answer: str = "Machine Learning là nhánh của AI.",
    has_answer: bool = True,
):
    """Tạo mock RAGResponse cho test."""
    m = MagicMock()
    m.query        = "test query"
    m.answer       = answer
    m.has_answer   = has_answer
    m.context_used = 3
    m.latency_s    = 0.5
    m.sources      = [
        {
            "rank": 1, "file_name": "giao_trinh.pdf",
            "chunk_index": 2, "score": 0.85, "rerank_score": 0.90,
        },
        {
            "rank": 2, "file_name": "de_cuong.pdf",
            "chunk_index": 5, "score": 0.78, "rerank_score": 0.82,
        },
    ]
    return m


def _get_test_client():
    """
    Tạo TestClient với app.state đã mock hoàn toàn.
    Dùng lifespan=False để bypass startup (không load model thật).
    """
    from fastapi.testclient import TestClient
    from app import app

    # Bypass lifespan — set state thủ công
    app.state.index_loaded = True
    app.state.llm_loaded   = True

    mock_retriever = MagicMock()
    mock_retriever._vector_store = MagicMock()
    mock_retriever._vector_store.size = 150

    app.state.retriever = mock_retriever
    app.state.reranker  = MagicMock()
    app.state.llm       = MagicMock()

    # TestClient với lifespan disabled (tránh gọi startup thật)
    client = TestClient(app, raise_server_exceptions=False)
    return client, app


class TestHealthEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client, cls.app = _get_test_client()

    def test_health_returns_200(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)

    def test_health_schema(self):
        r = self.client.get("/health")
        d = r.json()
        required = {"status", "index_loaded", "llm_loaded",
                    "index_size", "model_name", "uptime_s"}
        self.assertTrue(required.issubset(d.keys()))

    def test_health_status_ok_when_both_loaded(self):
        self.app.state.index_loaded = True
        self.app.state.llm_loaded   = True
        r = self.client.get("/health")
        self.assertEqual(r.json()["status"], "ok")

    def test_health_status_degraded_when_llm_missing(self):
        self.app.state.llm_loaded = False
        r = self.client.get("/health")
        self.assertEqual(r.json()["status"], "degraded")
        self.app.state.llm_loaded = True  # restore


class TestStatsEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client, cls.app = _get_test_client()

    def test_stats_returns_200(self):
        r = self.client.get("/stats")
        self.assertEqual(r.status_code, 200)

    def test_stats_has_required_fields(self):
        r = self.client.get("/stats")
        d = r.json()
        required = {
            "index_size", "embedding_model", "llm_model",
            "reranker_enabled", "top_k", "top_n", "total_queries",
        }
        self.assertTrue(required.issubset(d.keys()))

    def test_stats_embedding_model_correct(self):
        r = self.client.get("/stats")
        self.assertEqual(r.json()["embedding_model"], "BAAI/bge-m3")


class TestAskEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client, cls.app = _get_test_client()

    def _patch_rag(self, response=None):
        if response is None:
            response = _make_mock_rag_response()
        return patch("app._run_rag_sync", return_value=response)

    def test_ask_returns_200(self):
        with self._patch_rag():
            r = self.client.post("/ask", json={"query": "Machine Learning là gì?"})
        self.assertEqual(r.status_code, 200)

    def test_ask_response_schema(self):
        with self._patch_rag():
            r = self.client.post("/ask", json={"query": "test query này?"})
        d = r.json()
        required = {"query", "answer", "has_answer", "context_used",
                    "latency_s", "sources"}
        self.assertTrue(required.issubset(d.keys()))

    def test_ask_sources_schema(self):
        with self._patch_rag():
            r = self.client.post("/ask", json={"query": "câu hỏi test?"})
        sources = r.json()["sources"]
        self.assertIsInstance(sources, list)
        for s in sources:
            self.assertIn("rank", s)
            self.assertIn("file_name", s)
            self.assertIn("score", s)

    def test_ask_empty_query_returns_422(self):
        r = self.client.post("/ask", json={"query": ""})
        self.assertEqual(r.status_code, 422)

    def test_ask_short_query_returns_422(self):
        """Query dưới 3 ký tự → 422 Unprocessable Entity."""
        r = self.client.post("/ask", json={"query": "ab"})
        self.assertEqual(r.status_code, 422)

    def test_ask_missing_query_returns_422(self):
        r = self.client.post("/ask", json={})
        self.assertEqual(r.status_code, 422)

    def test_ask_has_answer_true(self):
        resp = _make_mock_rag_response(has_answer=True)
        with self._patch_rag(resp):
            r = self.client.post("/ask", json={"query": "câu hỏi dài hơn?"})
        self.assertTrue(r.json()["has_answer"])

    def test_ask_has_answer_false(self):
        resp = _make_mock_rag_response(
            answer="Tài liệu không đề cập đến vấn đề này.",
            has_answer=False,
        )
        with self._patch_rag(resp):
            r = self.client.post("/ask", json={"query": "câu hỏi không có đáp án?"})
        self.assertFalse(r.json()["has_answer"])

    def test_ask_with_top_n_override(self):
        with self._patch_rag():
            r = self.client.post("/ask", json={
                "query": "câu hỏi test với top_n?",
                "top_n": 3,
            })
        self.assertEqual(r.status_code, 200)

    def test_ask_index_not_loaded_returns_503(self):
        self.app.state.index_loaded = False
        r = self.client.post("/ask", json={"query": "câu hỏi test service?"})
        self.assertEqual(r.status_code, 503)
        self.app.state.index_loaded = True  # restore

    def test_ask_llm_not_loaded_returns_503(self):
        self.app.state.llm_loaded = False
        r = self.client.post("/ask", json={"query": "câu hỏi test llm?"})
        self.assertEqual(r.status_code, 503)
        self.app.state.llm_loaded = True  # restore

    def test_ask_adds_to_history(self):
        import app as app_module
        initial_count = len(app_module._query_history)
        with self._patch_rag():
            self.client.post("/ask", json={"query": "câu hỏi cho history test?"})
        self.assertEqual(len(app_module._query_history), initial_count + 1)


class TestUploadEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client, cls.app = _get_test_client()

    def test_upload_txt_returns_200(self):
        content = b"N\xe1\xbb\x99i dung t\xe3\x80\x80i li\xe1\xbb\x87u test."
        r = self.client.post(
            "/upload",
            files={"file": ("test.txt", BytesIO(content), "text/plain")},
        )
        self.assertEqual(r.status_code, 200)

    def test_upload_response_schema(self):
        content = b"Test document content for upload."
        r = self.client.post(
            "/upload",
            files={"file": ("doc.txt", BytesIO(content), "text/plain")},
        )
        d = r.json()
        self.assertIn("filename", d)
        self.assertIn("size_bytes", d)
        self.assertIn("message", d)

    def test_upload_unsupported_format_returns_415(self):
        content = b"fake xlsx content"
        r = self.client.post(
            "/upload",
            files={"file": ("data.xlsx", BytesIO(content),
                             "application/vnd.ms-excel")},
        )
        self.assertEqual(r.status_code, 415)

    def test_upload_pdf_accepted(self):
        # PDF header magic bytes
        content = b"%PDF-1.4 fake pdf content for testing purposes only"
        r = self.client.post(
            "/upload",
            files={"file": ("test.pdf", BytesIO(content), "application/pdf")},
        )
        self.assertEqual(r.status_code, 200)

    def test_upload_saves_correct_size(self):
        content = b"A" * 1000
        r = self.client.post(
            "/upload",
            files={"file": ("size_test.txt", BytesIO(content), "text/plain")},
        )
        self.assertEqual(r.json()["size_bytes"], 1000)


class TestHistoryEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client, cls.app = _get_test_client()

    def test_history_returns_200(self):
        r = self.client.get("/history")
        self.assertEqual(r.status_code, 200)

    def test_history_schema(self):
        r = self.client.get("/history")
        d = r.json()
        self.assertIn("total", d)
        self.assertIn("showing", d)
        self.assertIn("history", d)
        self.assertIsInstance(d["history"], list)

    def test_history_limit_param(self):
        r = self.client.get("/history?limit=5")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertLessEqual(d["showing"], 5)


class TestRootEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client, cls.app = _get_test_client()

    def test_root_returns_200(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)

    def test_root_has_endpoints_map(self):
        r = self.client.get("/")
        d = r.json()
        self.assertIn("endpoints", d)
        self.assertIn("ask", d["endpoints"])


class TestAskRequestValidation(unittest.TestCase):
    """Test Pydantic schema validation trực tiếp."""

    def test_valid_request(self):
        from app import AskRequest
        req = AskRequest(query="Câu hỏi hợp lệ này?")
        self.assertEqual(req.query, "Câu hỏi hợp lệ này?")

    def test_top_k_must_be_positive(self):
        from app import AskRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            AskRequest(query="test query đây?", top_k=0)

    def test_top_k_max_20(self):
        from app import AskRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            AskRequest(query="test query đây?", top_k=21)

    def test_top_n_max_10(self):
        from app import AskRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            AskRequest(query="test query đây?", top_n=11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
