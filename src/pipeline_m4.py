# src/pipeline_m4.py
import time
from typing import Dict
from src.retrieval.searcher import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.llm.model_manager import LocalLLMManager
from src.llm.prompter import AcademicPrompter
from config import RETRIEVAL_TOP_K, RERANKER_TOP_N, LLM_TEMPERATURE, LLM_MAX_NEW_TOKENS

class EndToEndRAGPipeline:
    def __init__(self):
        print("=" * 65)
        print("🏛  KHỞI TẠO HỆ THỐNG RAG TOÀN DIỆN (MILESTONE 4)")
        print("=" * 65)
        self.retriever = DenseRetriever(top_k=RETRIEVAL_TOP_K)
        self.reranker = CrossEncoderReranker()
        self.llm = LocalLLMManager()
        print("[✓] Toàn bộ hệ thống RAG đã sẵn sàng phục vụ!\n")

    def answer_query(self, query: str) -> Dict:
        start_time = time.time()
        
        # 1. Truy xuất thô từ FAISS
        t0 = time.time()
        candidates = self.retriever.retrieve(query, top_k=RETRIEVAL_TOP_K)
        dense_time = (time.time() - t0) * 1000
        
        # 2. Tái sắp xếp (Reranking)
        t1 = time.time()
        reranked_chunks = self.reranker.rerank(query, candidates, top_n=RERANKER_TOP_N)
        rerank_time = (time.time() - t1) * 1000
        
        # 3. Tạo Prompt (Tên biến: prompt - số ít)
        prompt = AcademicPrompter.build_prompt(query, reranked_chunks)
        
        # 4. Sinh câu trả lời qua LLM (Truyền đúng biến: prompt - số ít)
        t2 = time.time()
        response_text = self.llm.generate(
            prompt=prompt, 
            max_new_tokens=LLM_MAX_NEW_TOKENS, 
            temperature=LLM_TEMPERATURE
        )
        gen_time = time.time() - t2
        total_time = time.time() - start_time

        return {
            "query": query,
            "answer": response_text,
            "sources": reranked_chunks,
            "latency": {
                "dense_ms": round(dense_time, 1),
                "rerank_ms": round(rerank_time, 1),
                "generation_s": round(gen_time, 2),
                "total_s": round(total_time, 2)
            }
        }

def interactive_session():
    pipeline = EndToEndRAGPipeline()
    print("=" * 65)
    print("🎓 PHIÊN HỎI ĐÁP THƯ VIỆN SỐ (Gõ 'exit' hoặc 'quit' để thoát)")
    print("=" * 65)
    
    while True:
        try:
            query = input("\n❓ Câu hỏi: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Kết thúc phiên làm việc.")
                break
                
            print("\n[*] Đang tìm kiếm tài liệu và tổng hợp câu trả lời...")
            result = pipeline.answer_query(query)
            
            print("\n" + "=" * 65)
            print("🤖 TRẢ LỜI:")
            print(result["answer"])
            print("\n📚 CÁC NGUỒN TÀI LIỆU ĐƯỢC THAM KHẢO:")
            for i, src in enumerate(result["sources"], 1):
                subject = src.get("subject", "N/A")
                title = src.get("title", "N/A")
                r_score = src.get("rerank_score", 0.0)
                print(f"  [{i}] Môn: {subject} | Chủ đề: {title} (Điểm liên quan: {r_score:.2f})")
            
            lat = result["latency"]
            print(f"\n⏱  Thời gian xử lý: Retrieval={(lat['dense_ms'] + lat['rerank_ms']):.1f}ms | Sinh văn bản (LLM)={lat['generation_s']}s | Tổng={lat['total_s']}s")
            print("=" * 65)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[-] Lỗi trong quá trình xử lý: {e}")

if __name__ == "__main__":
    interactive_session()