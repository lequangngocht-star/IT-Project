"""
src/evaluation/evaluator.py
----------------------------
Chịu trách nhiệm DUY NHẤT: tính toán định lượng các chỉ số đánh giá
chất lượng hệ thống RAG trên bộ dữ liệu chuẩn (eval_dataset.json).

Các chỉ số được tính:
─────────────────────
┌─────────────────────┬──────────────────────────────────────────────────┐
│ Chỉ số              │ Đánh giá điều gì                                 │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Faithfulness        │ Câu trả lời có bám sát context không?            │
│                     │ (tỷ lệ câu trong answer có nguồn từ context)     │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Answer Relevancy    │ Câu trả lời có liên quan đến câu hỏi không?      │
│                     │ (overlap từ giữa answer và question)             │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Context Recall      │ Context có chứa đủ thông tin để trả lời không?   │
│                     │ (overlap giữa ground_truth và context)           │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Context Precision   │ Context retrieved có chính xác không?            │
│                     │ (tỷ lệ chunk retrieved thực sự liên quan)        │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Recall@k            │ Trong top-k chunk, có bao nhiêu là relevant?     │
│                     │ (k ∈ {1, 3, 5} — đánh giá retrieval)            │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Response Time       │ Latency trung bình mỗi query (giây)              │
│                     │ (end-to-end: retrieve + rerank + LLM gen)        │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Answer Rate         │ Tỷ lệ query được trả lời (không từ chối)         │
└─────────────────────┴──────────────────────────────────────────────────┘

Triết lý thiết kế:
- Không phụ thuộc vào RAGAS framework (tránh dependency nặng, khó cài).
- Tính toán thuần Python + numpy — chạy được offline, không cần API.
- Faithfulness và Relevancy dùng lexical overlap (F1 token-level) thay vì
  LLM-as-judge để đảm bảo reproducibility và không cần thêm LLM call.
- Kết quả lưu JSON + in ra console cho báo cáo học thuật.

Note học thuật:
- Lexical metrics (F1, ROUGE-style) là baseline đơn giản, reproducible.
- Nếu muốn semantic metrics cao hơn → dùng sentence embedding cosine sim.
- Trong báo cáo: nêu rõ dùng lexical overlap, không phải RAGAS gốc.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    EVAL_DATASET_PATH,
    EVAL_RESULTS_DIR,
    EVAL_TOP_K_VALUES,
)
from src.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """Một mẫu trong bộ dataset đánh giá."""
    id:               str
    question:         str
    ground_truth:     str
    relevant_chunks:  list[str]     # chunk_id ground-truth
    category:         str = ""
    difficulty:       str = ""


@dataclass
class EvalResult:
    """
    Kết quả đánh giá cho một query đơn lẻ.
    Tất cả metrics trong [0, 1] trừ latency_s.
    """
    sample_id:          str
    question:           str
    ground_truth:       str
    answer:             str
    has_answer:         bool
    faithfulness:       float       # [0,1] — answer bám sát context
    answer_relevancy:   float       # [0,1] — answer liên quan câu hỏi
    context_recall:     float       # [0,1] — context chứa đủ ground truth
    context_precision:  float       # [0,1] — chunk retrieved có relevant
    recall_at_k:        dict        # {"recall@1": float, "recall@3": float, ...}
    latency_s:          float       # Thời gian end-to-end (giây)
    retrieved_chunk_ids: list[str]  # chunk_id thực sự retrieved
    category:           str = ""
    difficulty:         str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalReport:
    """Báo cáo tổng hợp trên toàn bộ dataset."""
    total_samples:          int
    answered_samples:       int
    answer_rate:            float       # tỷ lệ có câu trả lời
    mean_faithfulness:      float
    mean_answer_relevancy:  float
    mean_context_recall:    float
    mean_context_precision: float
    mean_recall_at_k:       dict        # {"recall@1": float, ...}
    mean_latency_s:         float
    std_latency_s:          float
    by_category:            dict        # metrics phân theo category
    by_difficulty:          dict        # metrics phân theo difficulty
    per_sample:             list[dict]  # list EvalResult.to_dict()

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────
# Tokenizer tiếng Việt đơn giản
# ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Tokenize văn bản tiếng Việt ở mức từ đơn giản.

    Tiếng Việt không có dấu cách giữa âm tiết trong từ ghép,
    nên dùng whitespace tokenization là đủ cho lexical F1.
    Lowercase + bỏ dấu câu để tăng recall.

    Args:
        text: Chuỗi văn bản.

    Returns:
        List token lowercase, đã bỏ dấu câu và stopword ngắn.
    """
    text = text.lower().strip()
    # Bỏ dấu câu, giữ chữ và số và dấu thanh tiếng Việt
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = text.split()
    # Bỏ token quá ngắn (1 ký tự) — thường là nhiễu
    return [t for t in tokens if len(t) > 1]


def _f1_token_overlap(pred: str, ref: str) -> float:
    """
    Tính F1 score dựa trên token overlap giữa prediction và reference.

    F1 = 2 * precision * recall / (precision + recall)
    Trong đó:
        precision = |pred ∩ ref| / |pred|
        recall    = |pred ∩ ref| / |ref|

    Đây là metric chuẩn trong QA evaluation (SQuAD style).
    Phù hợp cho tiếng Việt hơn BLEU (không cần n-gram).

    Args:
        pred: Văn bản dự đoán (câu trả lời từ LLM).
        ref:  Văn bản tham chiếu (ground truth).

    Returns:
        F1 score trong [0, 1].
    """
    pred_tokens = _tokenize(pred)
    ref_tokens  = _tokenize(ref)

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter  = Counter(ref_tokens)

    # Intersection (lấy min của mỗi token)
    common = sum((pred_counter & ref_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall    = common / len(ref_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return f1


# ─────────────────────────────────────────────────────────────
# Individual metric functions
# ─────────────────────────────────────────────────────────────

def compute_faithfulness(answer: str, context_chunks: list[dict]) -> float:
    """
    Đo mức độ câu trả lời bám sát context (không hallucinate).

    Phương pháp lexical:
    - Chia answer thành các câu.
    - Mỗi câu: tính F1 overlap với toàn bộ context.
    - Faithfulness = trung bình F1 của các câu.

    Câu có F1 >= 0.3 coi là "được hỗ trợ bởi context".

    Args:
        answer:         Câu trả lời từ LLM.
        context_chunks: List chunk đã đưa vào prompt.

    Returns:
        Float [0, 1]. 1.0 = hoàn toàn bám sát context.
    """
    if not answer.strip() or not context_chunks:
        return 0.0

    # Gộp tất cả context thành một chuỗi dài
    full_context = " ".join(c.get("text", "") for c in context_chunks)

    # Tách answer thành câu
    sentences = [s.strip() for s in re.split(r"[.!?।]", answer) if len(s.strip()) > 10]
    if not sentences:
        # Không tách được câu → dùng toàn bộ answer
        return _f1_token_overlap(answer, full_context)

    f1_scores = [_f1_token_overlap(sent, full_context) for sent in sentences]
    return float(np.mean(f1_scores))


def compute_answer_relevancy(answer: str, question: str) -> float:
    """
    Đo mức độ câu trả lời liên quan đến câu hỏi.

    Phương pháp: F1 token overlap giữa answer và question.
    Câu trả lời tốt thường dùng lại các từ khóa từ câu hỏi.

    Args:
        answer:   Câu trả lời từ LLM.
        question: Câu hỏi gốc.

    Returns:
        Float [0, 1]. 1.0 = hoàn toàn liên quan.
    """
    if not answer.strip() or not question.strip():
        return 0.0
    return _f1_token_overlap(answer, question)


def compute_context_recall(ground_truth: str, context_chunks: list[dict]) -> float:
    """
    Đo mức độ context đã retrieve đủ thông tin để trả lời câu hỏi.

    Phương pháp: F1 overlap giữa ground_truth và toàn bộ context.
    Nếu context không chứa thông tin trong ground_truth → LLM không thể
    trả lời đúng dù tốt đến đâu.

    Args:
        ground_truth:   Câu trả lời chuẩn từ eval_dataset.json.
        context_chunks: List chunk đã đưa vào prompt.

    Returns:
        Float [0, 1]. 1.0 = context chứa đủ thông tin.
    """
    if not ground_truth.strip() or not context_chunks:
        return 0.0

    full_context = " ".join(c.get("text", "") for c in context_chunks)
    return _f1_token_overlap(ground_truth, full_context)


def compute_context_precision(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids:  list[str],
) -> float:
    """
    Đo tỷ lệ chunk retrieved thực sự liên quan (precision của retrieval).

    Precision = |retrieved ∩ relevant| / |retrieved|

    Dùng chunk_id từ eval_dataset.json làm ground truth "relevant chunks".
    Cần thiết để biết retrieval có đang lấy đúng tài liệu không.

    Args:
        retrieved_chunk_ids: List chunk_id thực sự retrieved.
        relevant_chunk_ids:  List chunk_id "đúng" theo eval_dataset.

    Returns:
        Float [0, 1]. 1.0 = tất cả retrieved đều relevant.
    """
    if not retrieved_chunk_ids:
        return 0.0
    if not relevant_chunk_ids:
        return 0.0

    retrieved_set = set(retrieved_chunk_ids)
    relevant_set  = set(relevant_chunk_ids)
    hits = len(retrieved_set & relevant_set)
    return hits / len(retrieved_set)


def compute_recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids:  list[str],
    k_values:            list[int] = EVAL_TOP_K_VALUES,
) -> dict[str, float]:
    """
    Tính Recall@k cho nhiều giá trị k.

    Recall@k = |top-k retrieved ∩ relevant| / |relevant|

    Metric quan trọng cho retrieval: trong top-k chunk, có bao nhiêu
    chunk thực sự chứa câu trả lời đúng?

    Args:
        retrieved_chunk_ids: List chunk_id theo thứ tự rank (rank 1 đầu tiên).
        relevant_chunk_ids:  List chunk_id "đúng" theo eval_dataset.
        k_values:            List giá trị k cần tính.

    Returns:
        Dict {"recall@1": float, "recall@3": float, "recall@5": float}.
    """
    if not relevant_chunk_ids:
        return {f"recall@{k}": 0.0 for k in k_values}

    relevant_set = set(relevant_chunk_ids)
    results = {}

    for k in k_values:
        top_k_ids = retrieved_chunk_ids[:k]
        hits      = len(set(top_k_ids) & relevant_set)
        recall    = hits / len(relevant_set)
        results[f"recall@{k}"] = round(recall, 4)

    return results


# ─────────────────────────────────────────────────────────────
# RAG Evaluator class
# ─────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Đánh giá toàn bộ RAG pipeline trên bộ dữ liệu chuẩn.

    Workflow:
        1. Load eval_dataset.json.
        2. Với mỗi sample: chạy full RAG pipeline (retrieve + rerank + LLM).
        3. Tính 6 metrics cho từng sample.
        4. Tổng hợp thành EvalReport.
        5. Lưu kết quả JSON + in báo cáo console.

    Dependency injection:
        rag_query_fn: callable nhận (query: str) → RAGResponse.
        Cho phép test evaluator mà không cần khởi tạo full pipeline.
    """

    def __init__(
        self,
        rag_fn: Callable,
        dataset_path: Path = EVAL_DATASET_PATH,
        k_values:     list[int] = EVAL_TOP_K_VALUES,
    ) -> None:
        """
        Args:
            rag_fn:       Hàm nhận query (str) → RAGResponse (hoặc mock).
            dataset_path: Đường dẫn file eval_dataset.json.
            k_values:     Danh sách k cho Recall@k.
        """
        self._rag_fn   = rag_fn
        self._k_values = k_values

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Eval dataset không tồn tại: {dataset_path}")

        self._samples = self._load_dataset(dataset_path)
        logger.info(
            f"RAGEvaluator sẵn sàng | "
            f"{len(self._samples)} samples | k={k_values}"
        )

    # ─────────────────────────────────────────────────────────
    # Dataset loading
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_dataset(path: Path) -> list[EvalSample]:
        """Load và validate eval_dataset.json."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        samples = []
        for item in raw:
            samples.append(EvalSample(
                id              = item["id"],
                question        = item["question"],
                ground_truth    = item["ground_truth"],
                relevant_chunks = item.get("relevant_chunks", []),
                category        = item.get("category", ""),
                difficulty      = item.get("difficulty", ""),
            ))

        logger.info(f"Đã load {len(samples)} eval samples")
        return samples

    # ─────────────────────────────────────────────────────────
    # Per-sample evaluation
    # ─────────────────────────────────────────────────────────

    def _evaluate_sample(self, sample: EvalSample) -> EvalResult:
        """
        Chạy RAG pipeline và tính metrics cho một sample.

        Bắt exception để 1 sample lỗi không làm hỏng cả batch.
        """
        logger.info(f"Eval [{sample.id}] {sample.question[:60]}...")

        # ── Gọi RAG pipeline ─────────────────────────────────
        t0 = time.time()
        try:
            rag_response = self._rag_fn(sample.question)
        except Exception as e:
            logger.error(f"RAG query lỗi [{sample.id}]: {e}")
            # Trả về result với toàn bộ metrics = 0
            return EvalResult(
                sample_id=sample.id, question=sample.question,
                ground_truth=sample.ground_truth, answer="[ERROR]",
                has_answer=False, faithfulness=0.0, answer_relevancy=0.0,
                context_recall=0.0, context_precision=0.0,
                recall_at_k={f"recall@{k}": 0.0 for k in self._k_values},
                latency_s=time.time() - t0,
                retrieved_chunk_ids=[],
                category=sample.category, difficulty=sample.difficulty,
            )

        latency = time.time() - t0

        # ── Trích xuất thông tin từ RAGResponse ──────────────
        answer       = getattr(rag_response, "answer", "")
        has_answer   = getattr(rag_response, "has_answer", False)
        sources      = getattr(rag_response, "sources", [])

        # Lấy chunk_id của các chunk đã retrieved (theo thứ tự rank)
        retrieved_ids = [
            s.get("chunk_id", "")
            for s in sorted(sources, key=lambda x: x.get("rank", 999))
            if s.get("chunk_id")
        ]

        # ── Tính metrics ─────────────────────────────────────
        faithfulness      = compute_faithfulness(answer, sources)
        answer_relevancy  = compute_answer_relevancy(answer, sample.question)
        context_recall    = compute_context_recall(sample.ground_truth, sources)
        context_precision = compute_context_precision(
            retrieved_ids, sample.relevant_chunks
        )
        recall_at_k = compute_recall_at_k(
            retrieved_ids, sample.relevant_chunks, self._k_values
        )

        result = EvalResult(
            sample_id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            answer=answer,
            has_answer=has_answer,
            faithfulness=round(faithfulness, 4),
            answer_relevancy=round(answer_relevancy, 4),
            context_recall=round(context_recall, 4),
            context_precision=round(context_precision, 4),
            recall_at_k=recall_at_k,
            latency_s=round(latency, 3),
            retrieved_chunk_ids=retrieved_ids,
            category=sample.category,
            difficulty=sample.difficulty,
        )

        logger.debug(
            f"  [{sample.id}] faith={faithfulness:.3f} | "
            f"rel={answer_relevancy:.3f} | "
            f"ctx_recall={context_recall:.3f} | "
            f"latency={latency:.2f}s"
        )
        return result

    # ─────────────────────────────────────────────────────────
    # Aggregation
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(results: list[EvalResult]) -> dict:
        """Tính mean của tất cả metrics trên danh sách results."""
        if not results:
            return {}

        def _mean(vals): return round(float(np.mean(vals)), 4)

        k_keys = list(results[0].recall_at_k.keys())
        agg = {
            "mean_faithfulness":      _mean([r.faithfulness      for r in results]),
            "mean_answer_relevancy":  _mean([r.answer_relevancy  for r in results]),
            "mean_context_recall":    _mean([r.context_recall    for r in results]),
            "mean_context_precision": _mean([r.context_precision for r in results]),
            "mean_latency_s":         _mean([r.latency_s         for r in results]),
            "std_latency_s":  round(float(np.std([r.latency_s for r in results])), 4),
            "answer_rate":    round(sum(r.has_answer for r in results) / len(results), 4),
        }
        for k in k_keys:
            agg[f"mean_{k}"] = _mean([r.recall_at_k[k] for r in results])
        return agg

    def _breakdown_by(self, results: list[EvalResult], field: str) -> dict:
        """Tính metrics phân nhóm theo category hoặc difficulty."""
        groups: dict[str, list[EvalResult]] = {}
        for r in results:
            key = getattr(r, field, "unknown") or "unknown"
            groups.setdefault(key, []).append(r)
        return {g: self._aggregate(rs) for g, rs in groups.items()}

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def run(
        self,
        sample_ids: list[str] | None = None,
        save: bool = True,
    ) -> EvalReport:
        """
        Chạy đánh giá trên toàn bộ (hoặc tập con) dataset.

        Args:
            sample_ids: Nếu None → chạy tất cả. Truyền list ID để chạy subset.
            save:       Lưu kết quả JSON ra EVAL_RESULTS_DIR.

        Returns:
            EvalReport chứa tất cả metrics tổng hợp và per-sample.
        """
        samples = self._samples
        if sample_ids is not None:
            id_set  = set(sample_ids)
            samples = [s for s in samples if s.id in id_set]

        logger.info("═" * 60)
        logger.info(f"  BẮT ĐẦU EVALUATION — {len(samples)} samples")
        logger.info("═" * 60)

        t_total = time.time()
        results: list[EvalResult] = []

        for i, sample in enumerate(samples, 1):
            logger.info(f"  [{i}/{len(samples)}] {sample.id}")
            result = self._evaluate_sample(sample)
            results.append(result)

        # ── Tổng hợp ─────────────────────────────────────────
        agg = self._aggregate(results)

        k_keys = list(results[0].recall_at_k.keys()) if results else []

        report = EvalReport(
            total_samples=len(results),
            answered_samples=sum(r.has_answer for r in results),
            answer_rate=agg.get("answer_rate", 0.0),
            mean_faithfulness=agg.get("mean_faithfulness", 0.0),
            mean_answer_relevancy=agg.get("mean_answer_relevancy", 0.0),
            mean_context_recall=agg.get("mean_context_recall", 0.0),
            mean_context_precision=agg.get("mean_context_precision", 0.0),
            mean_recall_at_k={k: agg.get(f"mean_{k}", 0.0) for k in k_keys},
            mean_latency_s=agg.get("mean_latency_s", 0.0),
            std_latency_s=agg.get("std_latency_s", 0.0),
            by_category=self._breakdown_by(results, "category"),
            by_difficulty=self._breakdown_by(results, "difficulty"),
            per_sample=[r.to_dict() for r in results],
        )

        total_elapsed = time.time() - t_total
        logger.info(f"Evaluation hoàn tất | {total_elapsed:.1f}s tổng cộng")

        if save:
            self._save_report(report)

        return report

    # ─────────────────────────────────────────────────────────
    # Save & Display
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _save_report(report: EvalReport) -> Path:
        """Lưu EvalReport ra file JSON với timestamp."""
        import datetime
        out_dir = Path(EVAL_RESULTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"eval_report_{ts}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Đã lưu báo cáo → {path}")
        return path

    @staticmethod
    def print_report(report: EvalReport) -> None:
        """In báo cáo evaluation ra console theo format học thuật."""
        w = 62
        print("\n" + "═" * w)
        print(f"{'  📊 EVALUATION REPORT — RAG LIBRARY':^{w}}")
        print("═" * w)
        print(f"  Tổng samples   : {report.total_samples}")
        print(f"  Có câu trả lời : {report.answered_samples} / {report.total_samples}")
        print(f"  Answer Rate    : {report.answer_rate:.1%}")
        print("─" * w)
        print("  GENERATION METRICS (LLM Quality)")
        print(f"  {'Faithfulness':<28}: {report.mean_faithfulness:.4f}  "
              f"(answer bám context)")
        print(f"  {'Answer Relevancy':<28}: {report.mean_answer_relevancy:.4f}  "
              f"(answer liên quan câu hỏi)")
        print("─" * w)
        print("  RETRIEVAL METRICS")
        print(f"  {'Context Recall':<28}: {report.mean_context_recall:.4f}  "
              f"(context đủ thông tin)")
        print(f"  {'Context Precision':<28}: {report.mean_context_precision:.4f}  "
              f"(chunk retrieved đúng)")
        for k, v in report.mean_recall_at_k.items():
            label = k.replace("recall@", "Recall@")
            print(f"  {label:<28}: {v:.4f}")
        print("─" * w)
        print("  PERFORMANCE")
        print(f"  {'Mean Latency':<28}: {report.mean_latency_s:.3f}s")
        print(f"  {'Std Latency':<28}: {report.std_latency_s:.3f}s")

        if report.by_category:
            print("─" * w)
            print("  BY CATEGORY")
            for cat, metrics in sorted(report.by_category.items()):
                faith = metrics.get("mean_faithfulness", 0)
                rec   = metrics.get("mean_context_recall", 0)
                print(f"  {cat:<18} | Faithfulness={faith:.3f} | CtxRecall={rec:.3f}")

        if report.by_difficulty:
            print("─" * w)
            print("  BY DIFFICULTY")
            for diff, metrics in sorted(report.by_difficulty.items()):
                faith = metrics.get("mean_faithfulness", 0)
                rec   = metrics.get("mean_context_recall", 0)
                print(f"  {diff:<18} | Faithfulness={faith:.3f} | CtxRecall={rec:.3f}")

        print("═" * w + "\n")
