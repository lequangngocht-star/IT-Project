"""
config.py
---------
File cấu hình trung tâm cho toàn bộ hệ thống RAG Thư viện số.
Tất cả các module đều kế thừa hằng số từ tệp tin này.
"""

from pathlib import Path

# ==============================================================
# 1. ĐƯỜNG DẪN HỆ THỐNG VẬN HÀNH (PATHS)
# ==============================================================
BASE_DIR = Path(__file__).parent.resolve()

DATA_RAW_DIR       = BASE_DIR / "data" / "raw"        # Kho chứa PDF học thuật thô
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"  # Đầu ra Milestone 1 (chunks.json)
VECTOR_STORE_DIR   = BASE_DIR / "vector_store"        # Chỉ mục FAISS và Metadata
LOG_DIR            = BASE_DIR / "logs"

# Tự động khởi tạo thư mục hệ thống nếu chưa có sẵn
for folder in [DATA_RAW_DIR, DATA_PROCESSED_DIR, VECTOR_STORE_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ==============================================================
# 2. PHÂN HỆ THU THẬP & TIỀN XỬ LÝ (INGESTION & CHUNKING)
# ==============================================================
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

CHUNK_SIZE    = 512   # Kích thước ký tự tối đa của một đoạn văn băm nhỏ
CHUNK_OVERLAP = 64    # Số lượng ký tự gối đầu (overlap) để bảo toàn ngữ cảnh
MIN_CHUNK_LEN = 50    # Ngưỡng loại bỏ các phân đoạn nhiễu, quá ngắn

# ==============================================================
# 3. PHÂN HỆ EMBEDDING & VECTOR DATABASE (MILESTONE 2)
# ==============================================================
# BGE-M3 là mô hình nhúng đa ngữ nghĩa xuất sắc cho tiếng Việt học thuật
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE     = "cpu"   # Chuyển dịch sang "cuda" nếu máy bạn có GPU Nvidia
EMBEDDING_BATCH_SIZE = 16      # Số chunk đẩy vào xử lý song song trên RAM/GPU

FAISS_INDEX_FILE     = VECTOR_STORE_DIR / "faiss.index"
FAISS_META_FILE      = VECTOR_STORE_DIR / "faiss_meta.json"
TOP_K                = 5       # Số lượng phân đoạn tối đa trích xuất cho một câu hỏi

# ==============================================================
# 4. PHÂN HỆ MÔ HÌNH NGÔN NGỮ LỚN LOCAL LLM (MILESTONE 4)
# ==============================================================
# Qwen2.5-1.5B hoặc Qwen2.5-7B là lựa chọn tối ưu cho hạ tầng máy bộ vừa và nhỏ
LLM_MODEL_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_DEVICE         = "cpu"     # Chuyển dịch sang "cuda" nếu có cấu hình GPU
LLM_MAX_NEW_TOKENS = 512       # Giới hạn độ dài văn bản câu trả lời sinh ra
LLM_TEMPERATURE    = 0.2       # Giảm độ sáng tạo để LLM tập trung lập luận thực tế

# ==============================================================
# 5. PHÂN HỆ GHI LOG VÀ ĐÁNH GIÁ (LOGGING & EVALUATION)
# ==============================================================
LOG_LEVEL = "INFO"
LOG_FILE  = LOG_DIR / "rag.log"

EVAL_TOP_K_VALUES = [1, 3, 5]  # Cấu hình tính toán Recall@K cho tầng RAGAS