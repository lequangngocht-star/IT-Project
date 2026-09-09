# src/pipeline_m5.py
import time
import json
from pathlib import Path
from typing import List, Dict
from src.retrieval.searcher import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from config import RETRIEVAL_TOP_K, RERANKER_TOP_N

# Bộ câu hỏi kiểm thử chuẩn bám sát kho tri thức CNTT
EVAL_TESTSET = [
    {
        "id": "Q01",
        "subject": "DB",
        "query": "What are the four ACID properties in database management systems and what does each guarantee?",
        "expected_subject": "DB"
    },
    {
        "id": "Q02",
        "subject": "DSA",
        "query": "What is a Binary Search Tree and what is the worst-case time complexity of its search operation?",
        "expected_subject": "DSA"
    },
    {
        "id": "Q03",
        "subject": "AI",
        "query": "How does the A* search algorithm determine the optimal path using heuristic functions?",
        "expected_subject": "AI"
    },
    {
        "id": "Q04",
        "subject": "OS",
        "query": "Explain the difference between a process and a thread in operating systems.",
        "expected_subject": "OS"
    },
    {
        "id": "Q05",
        "subject": "ML",
        "query": "What are the main paradigms of machine learning: supervised, unsupervised, and reinforcement learning?",
        "expected_subject": "ML"
    }
]

def evaluate_pipeline(use_reranker: bool = True) -> Dict:
    mode_name = "Dense + Cross-Encoder Reranker" if use_reranker else "Dense Search Only (No Rerank)"
    print(f"\n[*] Đang chạy đánh giá: {mode_name}...")

    retriever = DenseRetriever(top_k=RETRIEVAL_TOP_K)
    reranker = CrossEncoderReranker() if use_reranker else None

    total_queries = len(EVAL_TESTSET)
    latencies = []
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_ranks = []

    for item in EVAL_TESTSET:
        query = item["query"]
        expected_sub = item["expected_subject"]
        
        t0 = time.time()
        candidates = retriever.retrieve(query, top_k=RETRIEVAL_TOP_K)
        
        if use_reranker and reranker is not None:
            final_chunks = reranker.rerank(query, candidates, top_n=RERANKER_TOP_N)
        else:
            final_chunks = candidates[:RERANKER_TOP_N]
            
        elapsed_ms = (time.time() - t0) * 1000
        latencies.append(elapsed_ms)

        rank = 0
        for idx, chunk in enumerate(final_chunks, 1):
            if chunk.get("subject", "").lower() == expected_sub.lower():
                rank = idx
                break

        if rank == 1:
            hit_at_1 += 1
        if 1 <= rank <= 3:
            hit_at_3 += 1
        if 1 <= rank <= 5:
            hit_at_5 += 1

        reciprocal_ranks.append((1.0 / rank) if rank > 0 else 0.0)
        print(f"   [{item['id']}] Môn kỳ vọng: {expected_sub:<4} | Rank tìm thấy: {rank if rank > 0 else 'Miss':<4} | Latency: {elapsed_ms:.1f}ms")

    return {
        "configuration": mode_name,
        "latency_ms": round(sum(latencies) / total_queries, 1),
        "hit_at_1": round((hit_at_1 / total_queries) * 100, 1),
        "hit_at_3": round((hit_at_3 / total_queries) * 100, 1),
        "hit_at_5": round((hit_at_5 / total_queries) * 100, 1),
        "mrr": round(sum(reciprocal_ranks) / total_queries, 3)
    }

def main():
    print("=" * 70)
    print("📊 BẮT ĐẦU THỰC NGHIỆM ĐÁNH GIÁ ĐỊNH LƯỢNG (MILESTONE 5)")
    print("=" * 70)
    
    # Đo đạc 2 kịch bản phục vụ Ablation Study
    res_no_rerank = evaluate_pipeline(use_reranker=False)
    res_with_rerank = evaluate_pipeline(use_reranker=True)

    print("\n" + "=" * 75)
    print("📈 BẢNG TỔNG HỢP KẾT QUẢ ABLATION STUDY (DÙNG CHO BÁO CÁO VÀ SLIDE)")
    print("=" * 75)
    print(f"{'Cấu hình thử nghiệm':<33} | {'Latency':<9} | {'Hit@1':<8} | {'Hit@3':<8} | {'MRR':<6}")
    print("-" * 75)
    print(f"{res_no_rerank['configuration']:<33} | {res_no_rerank['latency_ms']:>6.1f} ms | {res_no_rerank['hit_at_1']:>6.1f}% | {res_no_rerank['hit_at_3']:>6.1f}% | {res_no_rerank['mrr']:>6.3f}")
    print(f"{res_with_rerank['configuration']:<33} | {res_with_rerank['latency_ms']:>6.1f} ms | {res_with_rerank['hit_at_1']:>6.1f}% | {res_with_rerank['hit_at_3']:>6.1f}% | {res_with_rerank['mrr']:>6.3f}")
    print("=" * 75)

if __name__ == "__main__":
    main()