# src/retrieval/searcher.py
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from config import (
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    RETRIEVAL_TOP_K
)
from src.embedding.embedder import TextEmbedder
from vector_store.faiss_store import FAISSVectorStore

class DenseRetriever:
    def __init__(self, top_k: int = RETRIEVAL_TOP_K):
        self.top_k = top_k
        print(f"[*] Khởi tạo Dense Retriever (Top-K={self.top_k})...")
        
        # 1. Nạp mô hình nhúng câu hỏi
        self.embedder = TextEmbedder(model_name=EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
        
        # 2. Nạp cơ sở dữ liệu Vector FAISS
        self.vector_store = FAISSVectorStore(FAISS_INDEX_PATH, FAISS_META_PATH)
        self.vector_store.load()
        print(f"[✓] Dense Retriever sẵn sàng kết nối.")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Truy xuất Top-K đoạn văn bản tương đồng ngữ nghĩa nhất."""
        k = top_k or self.top_k
        if not query or not query.strip():
            return []

        # Chuyển đổi câu hỏi thành vector
        query_vec = self.embedder.embed_query(query.strip())
        
        # Quét trên FAISS
        raw_results = self.vector_store.search(query_vec, top_k=k)
        
        candidates = []
        for doc_meta, score in raw_results:
            item = dict(doc_meta)
            item["dense_score"] = float(score)
            candidates.append(item)
            
        return candidates

if __name__ == "__main__":
    retriever = DenseRetriever(top_k=3)
    sample_query = "What is a binary search tree?"
    results = retriever.retrieve(sample_query)
    print(f"\nKết quả tìm kiếm thô cho: '{sample_query}'")
    for r in results:
        print(f"- [{r.get('subject', 'N/A')}] {r.get('title', 'N/A')} (Score: {r['dense_score']:.4f})")