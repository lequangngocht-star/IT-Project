import torch
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        print(f"[*] Đang tải mô hình Embedding: {model_name} trên {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_embedding_dimension()
        print(f"[✓] Nạp model thành công! Kích thước vector: {self.dimension} chiều.")

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Vector hóa danh sách chuỗi văn bản theo batch."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False  # Việc chuẩn hóa L2 sẽ do FAISS quản lý
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Vector hóa 1 câu hỏi người dùng phục vụ truy vấn."""
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=False
        )