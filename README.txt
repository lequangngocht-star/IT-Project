# Khởi tạo và kích hoạt môi trường ảo (Virtual Environment)
python -m venv venv
source venv/Scripts/activate  # Trên Windows dùng: venv\Scripts\activate

# Cập nhật bộ quản lý gói và cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt


Cấu trúc thư mục
Project/
│
├── config.py                      # Cấu hình tham số trung tâm (Chunk size, model names, paths)
├── requirements.txt               # Danh sách thư viện phụ thuộc của toàn bộ hệ thống
├── README.md                      # Hướng dẫn cài đặt, cấu hình và vận hành đồ án
│
├── data/                          # Phân hệ quản lý tầng dữ liệu
│   ├── raw/                       # Lưu trữ tài liệu gốc thu thập từ thư viện (.pdf, .docx, .txt)
│   └── processed/                 # Output Milestone 1: Tệp chunks.json chứa toàn bộ siêu dữ liệu
│
├── vector_store/                  # Phân hệ lưu trữ cơ sở dữ liệu Vector (Milestone 2)
│   ├── faiss.index                # Tệp lưu trữ cấu trúc index toán học của FAISS
│   └── faiss_meta.json            # Tệp mapping ID với nội dung văn bản gốc phục vụ trích dẫn
│
├── logs/                          # Thư mục tự động ghi log vận hành hệ thống (rag.log)
│
├── notebooks/                     # [R&D] Nơi chứa các file tài liệu thực nghiệm, báo cáo tiến độ tuần
│   └── milestone1_overview.ipynb  # File Notebook tổng quan chạy tương tác của Milestone 1
│
├── src/                           # [Production] Thư mục gốc chứa mã nguồn hệ thống dạng .py
│   ├── __init__.py
│   ├── logger.py                  # Module thiết lập cấu hình ghi log tập trung
│   ├── pipeline_m1.py             # Script thực thi Pipeline Milestone 1 (Offline)
│   ├── pipeline_m2.py             # Script thực thi Pipeline Milestone 2 (Offline)
│   ├── pipeline_m3.py             # Script thực thi Pipeline Milestone 3 (Online: Retrieval)
│   ├── pipeline_m4.py             # Script thực thi Pipeline Milestone 4 (Online: RAG API)
│   │
│   ├── ingestion/                 # Module con phụ trách Milestone 1 (Tầng thu nạp)
│   │   ├── __init__.py
│   │   ├── document_loader.py     # Trích xuất văn bản thô đa định dạng (.pdf, .docx, .txt)
│   │   └── text_cleaner.py        # Làm sạch sâu văn bản tiếng Việt và loại bỏ PDF artifacts
│   │
│   ├── preprocessing/             # Module con phụ trách tiền xử lý văn bản
│   │   ├── __init__.py
│   │   └── chunker.py             # Chiến lược băm nhỏ Paragraph-Aware và xử lý Overlap
│   │
│   ├── embedding/                 # Module con phụ trách Milestone 2
│   │   ├── __init__.py
│   │   └── embedder.py            # Quản lý nạp mô hình BGE-M3 và tính toán vector hóa theo batch
│   │
│   ├── retrieval/                 # Module con phụ trách Milestone 3
│   │   ├── __init__.py
│   │   ├── searcher.py            # Tìm kiếm tương đồng trên FAISS Index
│   │   └── reranker.py            # Tái sắp xếp loại bỏ nhiễu ngữ cảnh bằng Cross-Encoder
│   │
│   ├── llm/                       # Module con phụ trách Milestone 4
│   │   ├── __init__.py
│   │   ├── model_manager.py       # Tải Local LLM (Qwen/Llama) áp dụng Quantization 4-bit/8-bit
│   │   └── prompter.py            # Cấu trúc Prompt Template và quản lý ngữ cảnh hệ thống
│   │
│   └── evaluation/                # Module con phụ trách Milestone 5 (Tầng đánh giá học thuật)
│       ├── __init__.py
│       └── evaluator.py           # Tính toán định lượng các chỉ số bằng framework RAGAS
│
└── tests/                         # Thư mục chứa mã nguồn kiểm thử tự động hệ thống
    ├── __init__.py
    ├── test_milestone1.py         # Unit test kiểm thử text_cleaner và chunker của Milestone 1
    └── eval_dataset.json          # Bộ dữ liệu testset gồm 30-50 câu hỏi-đáp chuẩn đối sánh