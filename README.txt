#  HỆ THỐNG HỎI ĐÁP THÔNG MINH DỰA TRÊN LLM CHO THƯ VIỆN SỐ TRƯỜNG ĐẠI HỌC


##  CẤU TRÚC THƯ MỤC DỰ ÁN

```text
digital_library_rag/
├── config.py                     # Quản lý tham số tập trung (paths, models, thresholds, top-k)
├── requirements.txt              # Danh mục thư viện phụ thuộc (Streamlit 1.63.0, FAISS, PyTorch,...)
├── README.md                     # Tài liệu hướng dẫn cài đặt và vận hành đồ án
├── app.py                        # Giao diện Web Streamlit 
├── data/
│   ├── raw/                      # 1.184 tệp văn bản học liệu thô 
│   └── processed/                # Output M1: chunks.json 
├── vector_store/                 # Output M2: faiss.index (1024-dim) & faiss_meta.json
├── logs/                         # Nhật ký thực thi và kết quả đánh giá thực nghiệm
│   ├── ablation_variants_summary.csv # Bảng số liệu đối chứng 6 cấu hình thử nghiệm
│   ├── ablation_metrics.png      # Biểu đồ trực quan hoá các chỉ số Hit@K và MRR
│   └── latency_comparison.png    # Biểu đồ phân rã độ trễ các tầng xử lý
├── tests/
│   ├── eval_testset_35.json      # Bộ dữ liệu kiểm thử chuẩn 35 câu hỏi bao quát 10 môn học└── src/
    ├── ingestion/                # Thu thập dữ liệu và làm sạch sâu Unicode (text_cleaner.py)
    ├── preprocessing/            # Phân rã văn bản thông minh (chunker.py)
    ├── embedding/                # Module vector hoá BGE-M3 1.024 chiều (embedder.py)
    ├── vectordb/                 # Quản lý chỉ mục FAISS FlatIP (faiss_store.py)
    ├── retrieval/                # Module tìm kiếm Dense (searcher.py) & Reranker (reranker.py)
    ├── llm/                      # Quản lý suy luận Ollama và System Prompt (prompter.py)
    ├── evaluation/               # Script thực nghiệm đối chứng 5 biến thể (run_ablation_variants.py)
    ├── pipeline_m1.py            # Kịch bản chạy Offline: Ingestion & Chunking
    ├── pipeline_m2.py            # Kịch bản chạy Offline: Embedding & FAISS Indexing
    ├── pipeline_m3.py            # Thử nghiệm trực tuyến tầng Retrieval & Reranking
    ├── pipeline_m4.py            # Thử nghiệm toàn mạch End-to-End RAG trên Terminal
    └── pipeline_m5.py            # Kịch bản kiểm thử tự động đo đạc chỉ số cơ sở
    └── run_evaluation_suite.py   # Kịch bản kiểm thử tự động đo đạc chỉ số cơ sở