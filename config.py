"""
config.py
---------
File cấu hình trung tâm cho toàn bộ hệ thống RAG.
Tất cả các module đều import từ đây — không hardcode giá trị rải rác.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Đường dẫn project
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_RAW_DIR       = BASE_DIR / "data" / "raw"        # Tài liệu gốc (PDF, DOCX, TXT)
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"  # Chunks đã xử lý (JSON)
VECTOR_STORE_DIR   = BASE_DIR / "vector_store"        # Index FAISS / ChromaDB
LOG_DIR            = BASE_DIR / "logs"

# ──────────────────────────────────────────────
# Ingestion — đọc tài liệu
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# ──────────────────────────────────────────────
# Preprocessing — chunking
# ──────────────────────────────────────────────
CHUNK_SIZE    = 512   # Số ký tự tối đa mỗi chunk
CHUNK_OVERLAP = 64    # Số ký tự overlap giữa các chunk liên tiếp
MIN_CHUNK_LEN = 50    # Bỏ qua chunk ngắn hơn ngưỡng này (nhiễu)

# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────
# BGE-M3 hỗ trợ đa ngôn ngữ kể cả tiếng Việt, chạy local qua HuggingFace
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DEVICE     = "cpu"   # Đổi sang "cuda" nếu có GPU
EMBEDDING_BATCH_SIZE = 32      # Số chunk xử lý song song mỗi batch

# ──────────────────────────────────────────────
# Vector Database
# ──────────────────────────────────────────────
VECTOR_DB_BACKEND  = "faiss"           # "faiss" hoặc "chroma"
FAISS_INDEX_FILE   = VECTOR_STORE_DIR / "faiss.index"
FAISS_META_FILE    = VECTOR_STORE_DIR / "faiss_meta.json"
CHROMA_COLLECTION  = "rag_library"
TOP_K              = 5                  # Số chunk truy xuất mặc định

# ──────────────────────────────────────────────
# Retrieval — Milestone 3
# ──────────────────────────────────────────────
# Dense retrieval: số chunk lấy từ FAISS trước khi rerank
RETRIEVAL_TOP_K        = 10   # Lấy nhiều hơn TOP_K để reranker có đủ ứng viên

# Reranker: Cross-Encoder để chấm điểm lại chunk sau dense search
# Model nhẹ, chạy được trên CPU, hỗ trợ tiếng Việt qua multilingual training
RERANKER_MODEL_NAME    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_DEVICE        = "cpu"      # Đổi "cuda" nếu có GPU
RERANKER_TOP_N         = 5          # Số chunk giữ lại sau rerank (= TOP_K cuối cùng)
RERANKER_ENABLED       = True       # Tắt = chỉ dùng dense search thuần

# Score threshold: bỏ qua chunk có score quá thấp sau rerank
RERANKER_SCORE_THRESHOLD = -10.0    # Cross-Encoder ra logit, ngưỡng thực nghiệm

# ──────────────────────────────────────────────
# LLM (Milestone 4)
# ──────────────────────────────────────────────
LLM_MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"  # Thay bằng Gemma/Llama nếu cần
LLM_DEVICE       = "cpu"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE    = 0.2

# ──────────────────────────────────────────────
# Evaluation (Milestone 5)
# ──────────────────────────────────────────────
EVAL_TOP_K_VALUES = [1, 3, 5]   # Recall@k với các giá trị k này

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = LOG_DIR / "rag.log"