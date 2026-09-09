import time
import argparse
import torch
from pathlib import Path
from config import (
    CHUNKS_JSON_PATH,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE
)
from src.preprocessing.chunker import load_chunks
from src.embedding.embedder import TextEmbedder
from vector_store.faiss_store import FAISSVectorStore

def run_pipeline_m2(rebuild: bool = False):
    start_time = time.time()
    
    print("=" * 65)
    print("🏛  RAG LIBRARY — MILESTONE 2: EMBEDDING & FAISS VECTOR STORE")
    print("=" * 65)

    # 1. Kiểm tra tính tồn tại của index
    if not rebuild and FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists():
        print(f"[!] Tìm thấy chỉ mục FAISS tại: {FAISS_INDEX_PATH}")
        print("    Dùng cờ '--rebuild' nếu bạn muốn ghi đè và nhúng lại toàn bộ.")
        return

    # 2. Đọc chunks đã tạo ở Milestone 1
    if not CHUNKS_JSON_PATH.exists():
        print(f"[-] Không tìm thấy {CHUNKS_JSON_PATH}. Hãy chạy Milestone 1 trước!")
        return

    chunks = load_chunks(str(CHUNKS_JSON_PATH))
    total_chunks = len(chunks)
    print(f"[*] Đã nạp thành công {total_chunks} chunks từ Milestone 1.")

    if total_chunks == 0:
        print("[-] Dữ liệu chunks rỗng, dừng pipeline.")
        return

    # Tách riêng trường văn bản để nhúng và danh sách metadata
    texts = [c["text"] for c in chunks]
    metadata = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    # Lưu kèm văn bản gốc vào metadata để phục vụ trích dẫn và sinh câu trả lời
    for i, doc in enumerate(metadata):
        doc["text"] = texts[i]

    # 3. Khởi tạo Mô hình Embedding
    device = "cuda" if torch.cuda.is_available() and EMBEDDING_DEVICE == "cuda" else "cpu"
    embedder = TextEmbedder(model_name=EMBEDDING_MODEL_NAME, device=device)

    # 4. Sinh ma trận Vector theo Batch
    print(f"[*] Bắt đầu tính toán Vector nhúng (batch_size={EMBEDDING_BATCH_SIZE})...")
    embed_start = time.time()
    embeddings = embedder.embed_texts(texts, batch_size=EMBEDDING_BATCH_SIZE)
    embed_time = time.time() - embed_start
    print(f"[✓] Sinh xong ma trận: {embeddings.shape} trong {embed_time:.2f}s (~{total_chunks/max(embed_time,0.1):.1f} chunks/s)")

    # 5. Lưu trữ vào FAISS Index
    print(f"[*] Đang lưu trữ chỉ mục vào Vector Store...")
    vector_store = FAISSVectorStore(FAISS_INDEX_PATH, FAISS_META_PATH)
    vector_store.create_index(embedder.dimension)
    vector_store.add_embeddings(embeddings, metadata)
    vector_store.save()

    total_time = time.time() - start_time
    print("=" * 65)
    print(f"   - Model: {EMBEDDING_MODEL_NAME}")
    print(f"   - Số lượng vectors: {total_chunks}")
    print(f"   - Số chiều vector: {embedder.dimension}")
    print(f"   - Tổng thời gian thực thi: {total_time:.2f} giây")
    print(f"   - Tệp lưu trữ: {FAISS_INDEX_PATH.name} & {FAISS_META_PATH.name}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 2: Embedding & FAISS Builder")
    parser.add_argument("--rebuild", action="store_true", help="Ép buộc tạo lại Index dù file đã tồn tại")
    args = parser.parse_args()
    
    run_pipeline_m2(rebuild=args.rebuild)