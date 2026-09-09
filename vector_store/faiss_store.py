import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class FAISSVectorStore:
    def __init__(self, index_path: Path, meta_path: Path, dimension: Optional[int] = None):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []

    def create_index(self, dimension: int):
        """Khởi tạo chỉ mục FAISS tính Cosine Similarity (IndexFlatIP)."""
        self.dimension = dimension
        # Chuẩn hóa L2 trước khi dùng IndexFlatIP tương đương Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Thêm các ma trận vector và metadata tương ứng vào cơ sở dữ liệu."""
        if self.index is None:
            self.create_index(embeddings.shape[1])
            
        assert embeddings.shape[1] == self.dimension, (
            f"Lệch số chiều vector: index yêu cầu {self.dimension}, vector nhận {embeddings.shape[1]}"
        )
        
        # Chuẩn hóa vector đơn vị L2 để tích vô hướng chính là Cosine Similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)

    def save(self):
        """Lưu trữ cố định ra đĩa cứng."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"[✓] Đã lưu FAISS Index ({self.index.ntotal} vectors) vào {self.index_path}")
        print(f"[✓] Đã lưu Meta Mapping vào {self.meta_path}")

    def load(self):
        """Nạp chỉ mục và metadata từ đĩa cứng."""
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Không tìm thấy file index hoặc metadata. Hãy chạy pipeline_m2 trước!")
        
        self.index = faiss.read_index(str(self.index_path))
        self.dimension = self.index.d
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[✓] Đã nạp thành công FAISS Index: {self.index.ntotal} vectors | dim={self.dimension}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Tìm kiếm top_k văn bản tương đồng ngữ nghĩa nhất."""
        if self.index is None:
            self.load()

        vec = query_embedding.copy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        
        scores, indices = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
        return results