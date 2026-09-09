# src/pipeline_m3.py
import time
import argparse
from typing import List, Dict
from src.retrieval.searcher import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from config import RETRIEVAL_TOP_K, RERANKER_TOP_N

def run_query(query: str, retriever: DenseRetriever, reranker: CrossEncoderReranker = None, top_k: int = RETRIEVAL_TOP_K, top_n: int = RERANKER_TOP_N) -> List[Dict]:
    """Hàm chạy toàn diện cho một câu hỏi truy vấn."""
    start_time = time.time()
    
    # Giai đoạn 1: Dense Retrieval
    dense_start = time.time()
    candidates = retriever.retrieve(query, top_k=top_k)
    dense_time = (time.time() - dense_start) * 1000
    
    # Giai đoạn 2: Reranking (Nếu có kích hoạt)
    if reranker is not None and candidates:
        rerank_start = time.time()
        final_results = reranker.rerank(query, candidates, top_n=top_n)
        rerank_time = (time.time() - rerank_start) * 1000
    else:
        final_results = candidates[:top_n]
        for c in final_results:
            c["rerank_score"] = c.get("dense_score", 0.0)
        rerank_time = 0.0

    total_time = (time.time() - start_time) * 1000
    return final_results, dense_time, rerank_time, total_time

def interactive_session():
    print("=" * 65)
    print("🏛  RAG LIBRARY — MILESTONE 3: RETRIEVAL & RERANKING TESTING")
    print("=" * 65)
    
    retriever = DenseRetriever()
    reranker = CrossEncoderReranker()
    
    print("\n[✓] Hệ thống đã sẵn sàng! Gõ câu hỏi của bạn (hoặc 'quit'/'exit' để thoát).")
    
    while True:
        try:
            query = input("\n❓ Câu hỏi: ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit", "q"]:
                print("Tạm biệt!")
                break
                
            results, d_time, r_time, t_time = run_query(query, retriever, reranker)
            
            print(f"\n⏱  Hiệu năng: Dense: {d_time:.1f}ms | Rerank: {r_time:.1f}ms | Tổng: {t_time:.1f}ms")
            print(f"📄 Top-{len(results)} Chunks tìm thấy:")
            print("-" * 65)
            
            for i, r in enumerate(results, 1):
                subject = r.get("subject", "N/A")
                title = r.get("title", "N/A")
                d_score = r.get("dense_score", 0.0)
                r_score = r.get("rerank_score", 0.0)
                preview = r.get("text", "")[:150].replace("\n", " ")
                
                print(f" [{i}] Môn: {subject} | Chủ đề: {title}")
                print(f"     Dense Score: {d_score:.4f} | Rerank Score: {r_score:.4f}")
                print(f"     Nội dung: {preview}...\n")
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    interactive_session()