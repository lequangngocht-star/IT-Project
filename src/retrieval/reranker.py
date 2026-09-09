# src/retrieval/reranker.py
import numpy as np
from typing import List, Dict
from config import RERANKER_MODEL_NAME, RERANKER_SCORE_THRESHOLD, RERANKER_TOP_N

class CrossEncoderReranker:
    def __init__(
        self, 
        model_name: str = RERANKER_MODEL_NAME, 
        device: str = "cpu", 
        threshold: float = RERANKER_SCORE_THRESHOLD
    ):
        from sentence_transformers import CrossEncoder
        self.device = device
        self.threshold = threshold
        clean_model_name = model_name.strip()
        print(f"[*] Đang tải Multilingual Reranker Model: {clean_model_name} trên {self.device}...")
        
        # Nạp mô hình bge-reranker-v2-m3
        self.model = CrossEncoder(clean_model_name, device=self.device)
        print("[✓] Reranker Model đã sẵn sàng.")

    def rerank(self, query: str, candidates: List[Dict], top_n: int = RERANKER_TOP_N) -> List[Dict]:
        if not candidates:
            return []

        # Chuẩn hóa query: loại bỏ tiền tố số thứ tự (vd: "9. ") nếu có
        clean_query = query.strip()
        if len(clean_query) > 3 and clean_query[:2].isdigit() and clean_query[2] in [".", ":", " "]:
            clean_query = clean_query[3:].strip()

        pairs = [[clean_query, c.get("text", "")] for c in candidates]
        
        # Dự đoán raw logits
        raw_scores = self.model.predict(pairs)
        
        # Áp dụng Sigmoid đưa điểm về thang đo [0, 1] chuẩn mực
        # Sigmoid: 1 / (1 + exp(-x))
        sigmoid_scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))

        for idx, score in enumerate(sigmoid_scores):
            candidates[idx]["rerank_score"] = float(score)

        # Sắp xếp giảm dần theo điểm tương quan
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # Lọc theo Threshold đã cấu hình (ví dụ: >= 0.20)
        filtered = [c for c in ranked if c["rerank_score"] >= self.threshold]

        # Cơ chế kích hoạt Abstention nếu điểm quá thấp
        if not filtered:
            print(f"[!] Cảnh báo: Tất cả ứng viên đều dưới ngưỡng threshold ({self.threshold}). Kích hoạt cơ chế từ chối (Abstention).")
            return []

        return filtered[:top_n]