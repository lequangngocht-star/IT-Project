"""
src/evaluation/evaluator.py
---------------------------
Chịu trách nhiệm: Tính toán định lượng các chỉ số hiệu năng (Evaluation Metrics)
cho cả hai tầng Truy xuất (Retrieval) và Hệ thống tổng thể (End-to-End RAG).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Thêm root vào sys.path để import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import EVAL_TOP_K_VALUES
from src.logger import get_logger

logger = get_logger(__name__)


class RAGEvaluator:
    """
    Đánh giá định lượng hiệu năng của hệ thống RAG dựa trên tập Golden Testset.
    """

    def __init__(self, top_k_list: List[int] | None = None) -> None:
        self.top_k_list = top_k_list or EVAL_TOP_K_VALUES
        logger.info(f"Khởi tạo RAGEvaluator với k_list={self.top_k_list}")

    def evaluate_retrieval_query(
        self,
        retrieved_chunks: List[dict],
        expected_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Đánh giá hiệu năng truy xuất cho 1 câu hỏi đơn lẻ dựa trên Keyword Hit.
        """
        retrieved_texts = [c.get("text", "").lower() for c in retrieved_chunks]
        kw_lower = [k.lower() for k in expected_keywords]

        # Kiểm tra sự xuất hiện từ khóa trong từng rank
        hits_at_k = {}
        for k in self.top_k_list:
            sub_texts = " ".join(retrieved_texts[:k])
            has_hit = any(kw in sub_texts for kw in kw_lower) if kw_lower else False
            hits_at_k[f"hit_at_{k}"] = 1.0 if has_hit else 0.0

        # Tính Reciprocal Rank (MRR)
        first_hit_rank = 0
        for idx, text in enumerate(retrieved_texts, start=1):
            if any(kw in text for kw in kw_lower):
                first_hit_rank = idx
                break

        mrr = (1.0 / first_hit_rank) if first_hit_rank > 0 else 0.0

        # Tính tỉ lệ từ khóa thu hồi được trong Top-K lớn nhất
        all_retrieved_concat = " ".join(retrieved_texts)
        matched_kw_count = sum(1 for kw in kw_lower if kw in all_retrieved_concat)
        keyword_recall = (matched_kw_count / len(kw_lower)) if kw_lower else 0.0

        return {
            "hits_at_k": hits_at_k,
            "mrr": mrr,
            "keyword_recall": keyword_recall,
            "first_hit_rank": first_hit_rank
        }

    def evaluate_dataset(
        self,
        testset: List[dict],
        query_runner_func: Any
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ bộ kiểm thử qua hàm query_runner_func và tổng hợp kết quả.
        
        Args:
            testset: Danh sách dict từ eval_dataset.json.
            query_runner_func: Hàm callback nhận `query` và trả về `(results, latency_ms)`.
        """
        logger.info(f"Bắt đầu đánh giá định lượng trên {len(testset)} mẫu thử...")

        total_samples = len(testset)
        if total_samples == 0:
            return {"error": "Testset rỗng"}

        accumulated_hits = {f"hit_at_{k}": 0.0 for k in self.top_k_list}
        total_mrr = 0.0
        total_kw_recall = 0.0
        total_latency = 0.0
        per_query_details = []

        for sample in testset:
            q_id = sample.get("query_id", "")
            query = sample.get("query", "")
            expected_kw = sample.get("expected_keywords", [])

            t0 = time.time()
            retrieved_results = query_runner_func(query)
            latency_ms = (time.time() - t0) * 1000.0

            single_eval = self.evaluate_retrieval_query(retrieved_results, expected_kw)

            # Cộng dồn chỉ số
            for k_key, hit_val in single_eval["hits_at_k"].items():
                accumulated_hits[k_key] += hit_val

            total_mrr += single_eval["mrr"]
            total_kw_recall += single_eval["keyword_recall"]
            total_latency += latency_ms

            per_query_details.append({
                "query_id": q_id,
                "query": query,
                "latency_ms": round(latency_ms, 2),
                "mrr": round(single_eval["mrr"], 4),
                "keyword_recall": round(single_eval["keyword_recall"], 4),
                "hits": single_eval["hits_at_k"],
                "num_retrieved": len(retrieved_results)
            })

        # Tính trung bình các độ đo
        avg_hits = {k: round(v / total_samples, 4) for k, v in accumulated_hits.items()}
        avg_mrr = round(total_mrr / total_samples, 4)
        avg_kw_recall = round(total_kw_recall / total_samples, 4)
        avg_latency = round(total_latency / total_samples, 2)

        summary = {
            "total_queries": total_samples,
            "average_latency_ms": avg_latency,
            "mrr": avg_mrr,
            "keyword_recall": avg_kw_recall,
            "hit_rates": avg_hits,
            "query_details": per_query_details
        }

        logger.info(f"Hoàn thành đánh giá! MRR: {avg_mrr} | Latency: {avg_latency}ms")
        return summary