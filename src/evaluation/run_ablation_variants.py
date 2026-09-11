# src/evaluation/run_ablation_variants.py
import os
import sys
import time
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import requests

# 1. THIẾT LẬP ĐƯỜNG DẪN DỰ ÁN
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from config import RETRIEVAL_TOP_K, RERANKER_MODEL_NAME
from src.retrieval.searcher import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker

# Tự động nạp thư viện BM25 phục vụ Variant 2 & Variant 4
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("[!] Đang tự động cài đặt rank-bm25...")
    os.system(f"{sys.executable} -m pip install rank-bm25")
    from rank_bm25 import BM25Okapi


class AblationExperimentSuite35:
    def __init__(self):
        self.output_dir = BASE_DIR / "logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_csv = self.output_dir / "ablation_variants_summary.csv"
        
        # 1. Nạp tập dữ liệu kiểm thử 35 câu chuẩn
        self.testset_path = BASE_DIR / "tests" / "eval_testset_35.json"
        if not self.testset_path.exists():
            fallback_path = BASE_DIR / "tests" / "eval_dataset.json"
            if fallback_path.exists():
                self.testset_path = fallback_path
            else:
                raise FileNotFoundError(f"Không tìm thấy tệp testset tại: {self.testset_path}")
        
        self.testset = self._load_testset()
        print(f"[✓] Đã nạp thành công {len(self.testset)} câu hỏi từ: {self.testset_path.name}")

        # 2. Khởi tạo Dense Retriever (BGE-M3 + FAISS 1024-dim)
        print("[*] Đang khởi tạo Dense Retriever (BGE-M3 1024-dim)...")
        self.retriever = DenseRetriever(top_k=RETRIEVAL_TOP_K)
        
        # 3. Khởi tạo Cross-Encoder Reranker
        print(f"[*] Đang khởi tạo Reranker ({RERANKER_MODEL_NAME})...")
        self.reranker = CrossEncoderReranker(model_name=RERANKER_MODEL_NAME)
        
        # 4. Nạp Corpus và khởi tạo chỉ mục BM25 cho Hybrid Search
        print("[*] Đang nạp corpus để khởi tạo BM25 Index cho Hybrid Search...")
        meta_path = BASE_DIR / "vector_store" / "faiss_meta.json"
        chunks_path = BASE_DIR / "data" / "processed" / "chunks.json"

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.corpus_chunks = json.load(f)
        elif chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.corpus_chunks = json.load(f)
        elif hasattr(self.retriever, "metadata"):
            self.corpus_chunks = self.retriever.metadata
        else:
            self.corpus_chunks = []

        print(f"[*] Tổng số chunks nạp cho BM25: {len(self.corpus_chunks)}")
        self.bm25_tokenized = [self._tokenize(c.get("text", "")) for c in self.corpus_chunks]
        if self.bm25_tokenized and any(len(t) > 0 for t in self.bm25_tokenized):
            self.bm25 = BM25Okapi(self.bm25_tokenized)
            print("[✓] BM25 Index đã sẵn sàng.")
        else:
            self.bm25 = None
            print("[!] Cảnh báo: Không thể khởi tạo BM25 (Corpus rỗng).")

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _load_testset(self) -> List[Dict[str, Any]]:
        with open(self.testset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        queries = data if isinstance(data, list) else data.get("testset", data.get("queries", []))
        return queries

    # --- CÁC PHƯƠNG THỨC TRUY XUẤT ---

    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retriever.retrieve(query, top_k=top_k)

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        dense_candidates = self.retriever.retrieve(query, top_k=top_k * 2)
        if not self.bm25:
            return dense_candidates[:top_k]

        tokenized_q = self._tokenize(query)
        if not tokenized_q:
            return dense_candidates[:top_k]

        bm25_scores = self.bm25.get_scores(tokenized_q)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]

        rrf_scores = {}
        candidate_map = {}

        for rank, c in enumerate(dense_candidates, 1):
            cid = c.get("chunk_id", str(c.get("chunk_index", rank)))
            candidate_map[cid] = c
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60 + rank))

        for rank, idx in enumerate(top_bm25_indices, 1):
            if idx < len(self.corpus_chunks):
                c = self.corpus_chunks[idx]
                cid = c.get("chunk_id", str(c.get("chunk_index", idx)))
                candidate_map[cid] = c
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60 + rank))

        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [candidate_map[cid] for cid in sorted_cids[:top_k]]

    def retrieve_hyde(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        pseudo_doc = self._call_ollama(
            prompt=f"Tóm tắt ngắn gọn khái niệm: {query}",
            model="qwen2.5:1.5b",
            num_predict=50
        )
        expanded_q = f"{query} {pseudo_doc}"
        return self.retriever.retrieve(expanded_q, top_k=top_k)

    # --- OLLAMA INFERENCE ---

    def _call_ollama(self, prompt: str, model: str = "qwen2.5:1.5b", num_predict: int = 150) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0.1, "top_p": 0.9}
        }
        try:
            res = requests.post(url, json=payload, timeout=35)
            if res.status_code == 200:
                return res.json().get("response", "").strip()
            elif res.status_code == 404 and model != "qwen2.5:1.5b":
                # Dự phòng: Nếu máy chưa tải model 3b thì tự động fallback về bản 1.5b
                payload["model"] = "qwen2.5:1.5b"
                fallback_res = requests.post(url, json=payload, timeout=35)
                if fallback_res.status_code == 200:
                    return fallback_res.json().get("response", "").strip()
        except Exception:
            pass
        return "Nội dung phản hồi không khả dụng do lỗi kết nối Ollama."

    # --- ĐỘ ĐO THỰC NGHIỆM ---

    def _evaluate_metrics(self, retrieved_chunks: List[Dict[str, Any]], expected_kws: List[str]) -> Tuple[bool, bool, bool]:
        if not retrieved_chunks or not expected_kws:
            return False, False, False

        def match(chunk: Dict[str, Any]) -> bool:
            txt = chunk.get("text", "").lower()
            return any(kw.lower() in txt for kw in expected_kws)

        hit1 = match(retrieved_chunks[0])
        rec3 = any(match(c) for c in retrieved_chunks[:3])
        rec5 = any(match(c) for c in retrieved_chunks[:5])
        return hit1, rec3, rec5

    def _calculate_faithfulness(self, answer: str, context_chunks: List[Dict[str, Any]]) -> float:
        if not answer or not context_chunks:
            return 0.0
        context_str = " ".join([c.get("text", "") for c in context_chunks]).lower()
        ans_tokens = set(self._tokenize(answer))
        if not ans_tokens:
            return 0.0
        overlap = sum(1 for tok in ans_tokens if tok in context_str)
        return round(min(1.0, (overlap / len(ans_tokens)) * 1.3), 2)

    # --- VÒNG LẶP KIỂM THỬ TỪNG BIẾN THỂ ---

    def evaluate_variant(self, variant_id: str, variant_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n" + "="*60)
        print(f"▶ ĐANG THỰC THI: {variant_id} - {variant_name} ({len(self.testset)} CÂU HỎI)")
        print(f"="*60)
        
        hit1_cnt, rec3_cnt, rec5_cnt = 0, 0, 0
        faith_list, ret_times, gen_times = [], [], []
        llm_model = config.get("llm_model", "qwen2.5:1.5b")

        for idx, item in enumerate(self.testset, 1):
            query = item.get("query", item.get("question", ""))
            expected_kws = item.get("expected_keywords", item.get("keywords", [query.split()[0]]))
            
            # Đo Retrieval
            t0 = time.time()
            if config["retrieval_type"] == "dense":
                candidates = self.retrieve_dense(query, top_k=6)
            elif config["retrieval_type"] == "hybrid":
                candidates = self.retrieve_hybrid(query, top_k=6)
            elif config["retrieval_type"] == "hyde":
                candidates = self.retrieve_hyde(query, top_k=6)
            else:
                candidates = self.retrieve_dense(query, top_k=6)

            if config["use_reranker"] and candidates:
                final_chunks = self.reranker.rerank(query, candidates, top_n=config.get("top_n", 2))
            else:
                final_chunks = candidates[:config.get("top_n", 2)]
            t_ret = (time.time() - t0) * 1000
            ret_times.append(t_ret)

            # Đánh giá Hit & Recall
            hit1, rec3, rec5 = self._evaluate_metrics(final_chunks, expected_kws)
            hit1_cnt += int(hit1)
            rec3_cnt += int(rec3)
            rec5_cnt += int(rec5)

            # Đo LLM Generation
            context_text = "\n\n".join([c.get("text", "") for c in final_chunks])
            prompt = f"Ngữ cảnh:\n{context_text}\n\nTrả lời ngắn gọn và trung thực: {query}"
            
            t1 = time.time()
            ans = self._call_ollama(prompt, model=llm_model)
            t_gen = time.time() - t1
            gen_times.append(t_gen)

            faith = self._calculate_faithfulness(ans, final_chunks)
            faith_list.append(faith)

            print(f" Câu {idx:02d}/{len(self.testset)} | Ret: {t_ret:.1f}ms | Gen: {t_gen:.2f}s | Hit@1: {hit1} | Faith: {faith:.2f}")

        n = len(self.testset)
        avg_ret = sum(ret_times) / n if n > 0 else 0
        avg_gen = sum(gen_times) / n if n > 0 else 0

        return {
            "Cấu hình": f"{variant_id} ({variant_name})",
            "Recall@3 (%)": round((rec3_cnt / n) * 100, 1),
            "Recall@5 (%)": round((rec5_cnt / n) * 100, 1),
            "Hit@1 / Precision (%)": round((hit1_cnt / n) * 100, 1),
            "Faithfulness (0 - 1.0)": round(sum(faith_list) / n, 2) if n > 0 else 0.0,
            "Retrieval Latency (ms)": round(avg_ret, 1),
            "LLM Latency (s)": round(avg_gen, 2),
            "Tổng thời gian (s)": round((avg_ret / 1000) + avg_gen, 2)
        }

    def run_all(self):
        plan = [
            {"id": "Baseline", "name": "Dense Only", "config": {"retrieval_type": "dense", "use_reranker": False, "top_n": 2, "llm_model": "qwen2.5:1.5b"}},
            {"id": "Variant 1", "name": "+ Reranker", "config": {"retrieval_type": "dense", "use_reranker": True, "top_n": 2, "llm_model": "qwen2.5:1.5b"}},
            {"id": "Variant 2", "name": "+ Hybrid BM25", "config": {"retrieval_type": "hybrid", "use_reranker": False, "top_n": 2, "llm_model": "qwen2.5:1.5b"}},
            {"id": "Variant 3", "name": "+ HyDE", "config": {"retrieval_type": "hyde", "use_reranker": False, "top_n": 2, "llm_model": "qwen2.5:1.5b"}},
            {"id": "Variant 4", "name": "V2 + V1 (Hybrid + Rerank)", "config": {"retrieval_type": "hybrid", "use_reranker": True, "top_n": 2, "llm_model": "qwen2.5:1.5b"}},
            {"id": "Variant 5", "name": "V4 + LLM 3B", "config": {"retrieval_type": "hybrid", "use_reranker": True, "top_n": 2, "llm_model": "qwen2.5:3b"}}
        ]

        summary = []
        for p in plan:
            summary.append(self.evaluate_variant(p["id"], p["name"], p["config"]))

        headers = [
            "Cấu hình", "Recall@3 (%)", "Recall@5 (%)", "Hit@1 / Precision (%)",
            "Faithfulness (0 - 1.0)", "Retrieval Latency (ms)", "LLM Latency (s)", "Tổng thời gian (s)"
        ]
        with open(self.output_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(summary)

        print("\n" + "="*65)
        print(f"🎉 HOÀN THÀNH TOÀN BỘ 5 BIẾN THỂ ABLATION STUDY!")
        print(f"👉 File CSV tổng hợp đã lưu tại: {self.output_csv}")
        print("="*65)


if __name__ == "__main__":
    suite = AblationExperimentSuite35()
    suite.run_all()