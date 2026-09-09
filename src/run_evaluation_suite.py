# src/run_evaluation_suite.py
import os
import json
import time
import csv
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

from config import RETRIEVAL_TOP_K, RERANKER_TOP_N
from src.retrieval.searcher import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker

TESTSET_PATH = Path("tests/eval_testset_35.json")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_mode(testset: List[Dict], retriever: DenseRetriever, reranker: CrossEncoderReranker = None) -> List[Dict]:
    is_rerank = reranker is not None
    mode_label = "Dense + Reranker" if is_rerank else "Dense Only"
    print(f"\n[*] Đang chạy đánh giá chế độ: {mode_label} ({len(testset)} câu hỏi)...")
    
    results = []
    for idx, item in enumerate(testset, 1):
        qid = item["id"]
        query = item["query"]
        expected_sub = item["subject"]
        
        t0 = time.time()
        candidates = retriever.retrieve(query, top_k=RETRIEVAL_TOP_K)
        
        if is_rerank and candidates:
            final_chunks = reranker.rerank(query, candidates, top_n=RERANKER_TOP_N)
        else:
            final_chunks = candidates[:RERANKER_TOP_N]
            
        latency_ms = (time.time() - t0) * 1000
        
        # Xác định thứ hạng của môn học kỳ vọng
        rank = 0
        for pos, c in enumerate(final_chunks, 1):
            if c.get("subject", "").strip().lower() == expected_sub.strip().lower():
                rank = pos
                break
                
        hit1 = 1 if rank == 1 else 0
        hit3 = 1 if 1 <= rank <= 3 else 0
        hit5 = 1 if 1 <= rank <= 5 else 0
        rr = (1.0 / rank) if rank > 0 else 0.0
        
        results.append({
            "id": qid,
            "subject": expected_sub,
            "query": query,
            "mode": mode_label,
            "rank": rank,
            "hit@1": hit1,
            "hit@3": hit3,
            "hit@5": hit5,
            "mrr": rr,
            "latency_ms": round(latency_ms, 2)
        })
        print(f"  [{idx:02d}/{len(testset)}] {qid} | Subject: {expected_sub:<8} | Rank: {rank if rank > 0 else 'Miss':<4} | Latency: {latency_ms:.1f}ms")
        
    return results

def compute_metrics(eval_rows: List[Dict], mode_name: str) -> Dict:
    n = len(eval_rows)
    avg_h1 = (sum(r["hit@1"] for r in eval_rows) / n) * 100
    avg_h3 = (sum(r["hit@3"] for r in eval_rows) / n) * 100
    avg_h5 = (sum(r["hit@5"] for r in eval_rows) / n) * 100
    avg_mrr = sum(r["mrr"] for r in eval_rows) / n
    avg_lat = sum(r["latency_ms"] for r in eval_rows) / n
    
    return {
        "Configuration": mode_name,
        "Total_Queries": n,
        "Hit@1 (%)": round(avg_h1, 2),
        "Hit@3 (%)": round(avg_h3, 2),
        "Hit@5 (%)": round(avg_h5, 2),
        "MRR": round(avg_mrr, 4),
        "Avg_Latency (ms)": round(avg_lat, 2)
    }

def export_csv(dense_rows: List[Dict], rerank_rows: List[Dict], summaries: List[Dict]):
    # 1. Xuất CSV chi tiết từng câu
    per_query_path = LOG_DIR / "ablation_per_query.csv"
    with open(per_query_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Query_ID", "Subject", "Query", 
                         "Dense_Rank", "Dense_Hit@1", "Dense_Hit@3", "Dense_Hit@5", "Dense_MRR", "Dense_Latency_ms",
                         "Rerank_Rank", "Rerank_Hit@1", "Rerank_Hit@3", "Rerank_Hit@5", "Rerank_MRR", "Rerank_Latency_ms"])
        for d, r in zip(dense_rows, rerank_rows):
            writer.writerow([
                d["id"], d["subject"], d["query"],
                d["rank"], d["hit@1"], d["hit@3"], d["hit@5"], round(d["mrr"], 3), d["latency_ms"],
                r["rank"], r["hit@1"], r["hit@3"], r["hit@5"], round(r["mrr"], 3), r["latency_ms"]
            ])
    print(f"\n[✓] Đã xuất bảng chi tiết từng câu: {per_query_path.resolve()}")

    # 2. Xuất CSV bảng tổng hợp đối chứng
    summary_path = LOG_DIR / "ablation_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[✓] Đã xuất bảng tổng hợp chỉ số: {summary_path.resolve()}")

def plot_charts(summaries: List[Dict], dense_rows: List[Dict], rerank_rows: List[Dict]):
    # Biểu đồ 1: So sánh các chỉ số Hit@K và MRR
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    labels = ["Hit@1 (%)", "Hit@3 (%)", "Hit@5 (%)"]
    dense_vals = [summaries[0]["Hit@1 (%)"], summaries[0]["Hit@3 (%)"], summaries[0]["Hit@5 (%)"]]
    rerank_vals = [summaries[1]["Hit@1 (%)"], summaries[1]["Hit@3 (%)"], summaries[1]["Hit@5 (%)"]]
    
    x = np.arange(len(labels))
    width = 0.32
    
    rects1 = ax1.bar(x - width/2, dense_vals, width, label="Dense Only (FAISS)", color="#3498db")
    rects2 = ax1.bar(x + width/2, rerank_vals, width, label="Dense + Cross-Encoder", color="#2ecc71")
    
    ax1.set_ylabel("Tỷ lệ (%)", fontsize=11, fontweight="bold")
    ax1.set_title("So sánh hiệu năng trích xuất: Dense Only vs. Dense + Rerank\n(Đo trên 35 câu hỏi kiểm thử chuẩn)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f"{height:.1f}%",
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9, fontweight="bold")

    chart1_path = LOG_DIR / "ablation_metrics.png"
    plt.tight_layout()
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[✓] Đã vẽ và lưu biểu đồ chỉ số: {chart1_path.resolve()}")

    # Biểu đồ 2: So sánh độ trễ (Latency) chi tiết qua 35 câu hỏi
    fig, ax2 = plt.subplots(figsize=(10, 4.5))
    q_ids = [d["id"] for d in dense_rows]
    dense_lat = [d["latency_ms"] for d in dense_rows]
    rerank_lat = [r["latency_ms"] for r in rerank_rows]

    ax2.plot(q_ids, dense_lat, marker="o", linewidth=1.8, markersize=4, label=f"Dense Only (TB: {summaries[0]['Avg_Latency (ms)']}ms)", color="#2980b9")
    ax2.plot(q_ids, rerank_lat, marker="s", linewidth=1.8, markersize=4, label=f"Dense + Rerank (TB: {summaries[1]['Avg_Latency (ms)']}ms)", color="#e67e22")

    ax2.set_xlabel("Mã câu hỏi truy vấn (Query ID)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Thời gian phản hồi (ms)", fontsize=10, fontweight="bold")
    ax2.set_title("Phân bố độ trễ phản hồi trên 35 câu hỏi kiểm thử (Latency Profile)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(range(len(q_ids)))
    ax2.set_xticklabels(q_ids, rotation=45, fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    chart2_path = LOG_DIR / "latency_comparison.png"
    plt.tight_layout()
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[✓] Đã vẽ và lưu biểu đồ độ trễ: {chart2_path.resolve()}")

def main():
    if not TESTSET_PATH.exists():
        print(f"[-] Không tìm thấy file {TESTSET_PATH}. Vui lòng tạo file trước!")
        return
        
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        testset = json.load(f)

    # Khởi tạo mô hình truy xuất
    retriever = DenseRetriever(top_k=RETRIEVAL_TOP_K)
    reranker = CrossEncoderReranker()
    # 1. Chạy Dense Only
    dense_results = evaluate_mode(testset, retriever, reranker=None)
    summary_dense = compute_metrics(dense_results, "Dense Search Only (No Rerank)")

    # 2. Chạy Dense + Rerank
    rerank_results = evaluate_mode(testset, retriever, reranker=reranker)
    summary_rerank = compute_metrics(rerank_results, "Dense + Cross-Encoder Reranker")

    summaries = [summary_dense, summary_rerank]

    # In kết quả dạng bảng ra màn hình
    print("\n" + "=" * 80)
    print(f"{'Cấu hình':<32} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8} | {'Latency TB':<10}")
    print("-" * 80)
    for s in summaries:
        print(f"{s['Configuration']:<32} | {s['Hit@1 (%)']:>6.1f}% | {s['Hit@3 (%)']:>6.1f}% | {s['Hit@5 (%)']:>6.1f}% | {s['MRR']:>8.4f} | {s['Avg_Latency (ms)']:>8.1f} ms")
    print("=" * 80)

    # Xuất CSV và vẽ biểu đồ
    export_csv(dense_results, rerank_results, summaries)
    plot_charts(summaries, dense_results, rerank_results)

if __name__ == "__main__":
    main()