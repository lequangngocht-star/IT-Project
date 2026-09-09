import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Đường dẫn dữ liệu
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHUNKS_JSON_PATH = PROCESSED_DATA_DIR / "chunks.json"

# Đường dẫn Vector Store
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
FAISS_META_PATH = VECTOR_STORE_DIR / "faiss_meta.json"

# Cấu hình Embedding
# - "BAAI/bge-m3" (1024 dims, đa ngôn ngữ, chuẩn đề tài SĐH)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DEVICE = "cpu"  # Đổi thành "cuda" nếu chạy trên Colab/máy có GPU NVIDIA

RETRIEVAL_TOP_K = 5               
RERANKER_TOP_N = 2                    # Lọc lấy Top-5 sau khi Rerank
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE = "cpu"
RERANKER_SCORE_THRESHOLD = 0.35    

# LLM Generation Parameters
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_DEVICE = "cpu"
LLM_MAX_NEW_TOKENS = 256              # Đủ cho câu trả lời trọn vẹn, không bị cụt chữ
LLM_TEMPERATURE = 0.0                 # Nhiệt độ 0: Tất định, bám sát Grounding tuyệt đối