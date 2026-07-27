"""
src/llm/prompter.py
--------------------
Chịu trách nhiệm DUY NHẤT: xây dựng prompt từ câu hỏi + chunks đã truy xuất,
và parse câu trả lời từ LLM thành output chuẩn hóa.

Tại sao tách riêng prompter.py?
- model_manager.py không biết gì về RAG context hay format câu trả lời.
- prompter.py không biết gì về model hay tokenizer.
- Muốn thay đổi prompt template (thực nghiệm so sánh) chỉ sửa file này.
- Kiểm thử prompt logic không cần load model.

Thiết kế Prompt cho RAG tiếng Việt:
─────────────────────────────────────
Một RAG prompt tốt cần đảm bảo:
1. LLM CHỈ dựa vào context được cung cấp — không hallucinate từ training data.
2. LLM thừa nhận khi context không đủ thông tin.
3. Câu trả lời trích dẫn nguồn cụ thể (file_name, chunk_index).
4. Ngôn ngữ đầu ra là tiếng Việt.

Cấu trúc prompt (Chat Template format):
    [SYSTEM] Vai trò + hướng dẫn hành vi
    [USER]   Context (chunks) + Câu hỏi
    [ASSISTANT] → LLM sinh ra

Tại sao đặt context trong [USER] thay vì [SYSTEM]?
- Instruct models được fine-tune để xử lý thông tin trong [USER] turn.
- [SYSTEM] nên chứa persona và quy tắc hành vi, không phải dữ liệu.
- Thực nghiệm cho thấy context trong [USER] cho faithfulness cao hơn.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import LLM_MAX_CONTEXT_CHARS, LLM_MAX_CONTEXT_CHUNKS
from src.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Output chuẩn hóa của toàn bộ RAG pipeline (M4).

    Dùng dataclass thay vì dict để:
    - Type hints rõ ràng, IDE autocomplete.
    - Dễ serialize sang JSON cho evaluation (M5).
    - Bất biến hơn dict (tránh typo key).
    """
    query:        str                # Câu hỏi gốc
    answer:       str                # Câu trả lời từ LLM
    sources:      list[dict]         # Danh sách chunk nguồn đã dùng
    has_answer:   bool               # LLM có tìm được câu trả lời không
    context_used: int                # Số chunk thực sự đưa vào prompt
    latency_s:    float = 0.0        # Thời gian sinh câu trả lời (giây)

    def to_dict(self) -> dict:
        """Chuyển thành dict để JSON serialization."""
        return {
            "query":        self.query,
            "answer":       self.answer,
            "has_answer":   self.has_answer,
            "context_used": self.context_used,
            "latency_s":    round(self.latency_s, 3),
            "sources": [
                {
                    "rank":        s.get("rank", i + 1),
                    "file_name":   s.get("file_name", ""),
                    "chunk_index": s.get("chunk_index", -1),
                    "score":       round(s.get("score", 0.0), 4),
                    "rerank_score": round(s.get("rerank_score", 0.0), 4),
                }
                for i, s in enumerate(self.sources)
            ],
        }


# ─────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────

# System prompt định nghĩa vai trò và ràng buộc hành vi của LLM.
# Viết bằng tiếng Việt để instruct models hỗ trợ tiếng Việt hiểu tốt hơn.
_SYSTEM_PROMPT = """
Bạn là trợ lý học thuật của hệ thống RAG.

NHIỆM VỤ:

1. Chỉ sử dụng thông tin có trong phần "TÀI LIỆU THAM KHẢO".

2. Tuyệt đối KHÔNG sử dụng kiến thức đã học trước đó.

3. Không được suy luận hoặc tự bổ sung thông tin ngoài tài liệu.

4. Nếu tài liệu không đủ để trả lời thì chỉ trả lời:

"Tài liệu hiện có chưa đủ thông tin để trả lời câu hỏi này."

5. Trả lời đúng ngôn ngữ của câu hỏi:
- Câu hỏi tiếng Việt → trả lời tiếng Việt.
- Câu hỏi tiếng Anh → trả lời tiếng Anh.

6. Nếu có nhiều tài liệu, hãy tổng hợp chúng thành một câu trả lời thống nhất.

7. Không nhắc tới việc bạn là AI hay mô hình ngôn ngữ.
""".strip()

# Template cho phần user message — context + câu hỏi
_USER_PROMPT_TEMPLATE = """\
[TÀI LIỆU THAM KHẢO]
{context}

[CÂU HỎI]
{query}

Hãy trả lời câu hỏi dựa trên tài liệu tham khảo trên. Nếu tài liệu không đề cập đến vấn đề này, hãy cho biết rõ."""

# Từ khóa nhận diện "không có câu trả lời" trong output của LLM
_NO_ANSWER_SIGNALS = [

    # ===== Vietnamese =====

    "không có thông tin",
    "chưa có thông tin",
    "không đủ thông tin",
    "chưa đủ thông tin",
    "không tìm thấy thông tin",
    "không được đề cập",
    "tài liệu hiện có chưa đủ thông tin",
    "không thể trả lời",
    "không thể xác định",

    # ===== English =====

    "not enough information",
    "insufficient information",
    "no information",
    "not mentioned",
    "not provided",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "cannot determine",
    "cannot be determined",
    "not available",
]


# ─────────────────────────────────────────────────────────────
# Context Builder
# ─────────────────────────────────────────────────────────────

def build_context(
    chunks:         list[dict],
    max_chunks:     int = LLM_MAX_CONTEXT_CHUNKS,
    max_total_chars: int = LLM_MAX_CONTEXT_CHARS,
) -> tuple[str, list[dict]]:
    """
    Xây dựng phần context từ danh sách chunks đã retrieve+rerank.

    Tại sao cần giới hạn context?
    - LLM có context window cố định (thường 4096–8192 tokens cho model nhỏ).
    - Quá nhiều context → LLM "bị lạc" (lost in the middle phenomenon).
    - Quá ít context → thiếu thông tin để trả lời.
    - LLM_MAX_CONTEXT_CHUNKS=5 và LLM_MAX_CONTEXT_CHARS=3000 là trade-off thực nghiệm.

    Format mỗi chunk trong context:
        [Nguồn: tên_file | Đoạn số: N]
        Nội dung đoạn văn...
        ---

    Args:
        chunks:          List[dict] từ retrieval pipeline (đã có rank, score...).
        max_chunks:      Số chunk tối đa đưa vào context.
        max_total_chars: Tổng ký tự tối đa của context (không tính header).

    Returns:
        Tuple (context_str, used_chunks):
            context_str:  Chuỗi context đã format.
            used_chunks:  List chunk thực sự được dùng (có thể ít hơn input).
    """
    if not chunks:
        return "Không có tài liệu tham khảo.", []

    used_chunks: list[dict] = []
    context_parts: list[str] = []
    total_chars = 0

    # Chỉ giữ những chunk có chất lượng đủ tốt
    filtered_chunks = []

    for chunk in chunks:
        score = chunk.get("rerank_score")

        # Nếu có rerank score thì loại các chunk quá kém
        if score is not None and score < 0:
            continue

        filtered_chunks.append(chunk)

    # Nếu lọc hết thì quay lại dùng top-k ban đầu
    if not filtered_chunks:
        filtered_chunks = chunks[:max_chunks]

    for chunk in filtered_chunks[:max_chunks]:
        text        = chunk.get("text", "").strip()
        file_name   = chunk.get("file_name", "Không rõ nguồn")
        chunk_index = chunk.get("chunk_index", -1)
        rank        = chunk.get("rank", len(used_chunks) + 1)

        if not text:
            continue

        # Tạo header trích dẫn để LLM biết context đến từ đâu
        header = f"[Nguồn {rank}: {file_name} | Đoạn #{chunk_index}]"
        entry  = f"{header}\n{text}"

        # Kiểm tra giới hạn ký tự
        if total_chars + len(entry) > max_total_chars and used_chunks:
            # Đã có ít nhất 1 chunk — dừng để không vượt limit
            logger.debug(
                f"Dừng ở chunk {rank} — vượt max_total_chars={max_total_chars}"
            )
            break

        context_parts.append(entry)
        used_chunks.append(chunk)
        total_chars += len(entry)

    context_str = "\n---\n".join(context_parts)
    logger.info(
        f"Context: {len(used_chunks)}/{len(chunks)} chunks | "
        f"{total_chars:,} ký tự"
    )
    return context_str, used_chunks


# ─────────────────────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────────────────────

def build_messages(query: str, chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Xây dựng messages list cho Chat Template từ query + chunks.

    Đây là bước chuyển đổi từ dữ liệu structured (chunks)
    sang định dạng mà LLM instruct model hiểu được.

    Args:
        query:  Câu hỏi người dùng.
        chunks: List chunk từ retrieval pipeline.

    Returns:
        Tuple (messages, used_chunks):
            messages:    List[dict] với keys "role" và "content".
                         Sẵn sàng truyền vào LLMManager.generate_chat().
            used_chunks: Các chunk thực sự được đưa vào context.
    """
    # Bước 1: Build context từ chunks
    context_str, used_chunks = build_context(chunks)

    # Bước 2: Format user message
    user_content = _USER_PROMPT_TEMPLATE.format(
        context=context_str,
        query=query.strip(),
    )

    # Bước 3: Tạo messages theo Chat Template format
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    logger.debug(
        f"Messages built | system={len(_SYSTEM_PROMPT)} chars | "
        f"user={len(user_content)} chars"
    )
    return messages, used_chunks


# ─────────────────────────────────────────────────────────────
# Response Parser
# ─────────────────────────────────────────────────────────────

def parse_response(
    raw_answer:  str,
    query:       str,
    used_chunks: list[dict],
    latency_s:   float = 0.0,
) -> RAGResponse:
    """
    Parse raw text từ LLM thành RAGResponse chuẩn hóa.

    Kiểm tra xem LLM có thực sự trả lời được không hay
    thừa nhận thiếu thông tin.

    Args:
        raw_answer:  Chuỗi text trả về từ LLMManager.generate_chat().
        query:       Câu hỏi gốc.
        used_chunks: Các chunk đã đưa vào context.
        latency_s:   Thời gian sinh câu trả lời (giây).

    Returns:
        RAGResponse đã điền đầy đủ fields.
    """
    answer = raw_answer.strip()

    # Nhận diện "không có câu trả lời" bằng keyword matching
    answer_lower = answer.lower()
    has_answer = not any(
        signal in answer_lower
        for signal in _NO_ANSWER_SIGNALS
    )


    response = RAGResponse(
        query=query,
        answer=answer,
        sources=used_chunks,
        has_answer=has_answer,
        context_used=len(used_chunks),
        latency_s=latency_s,
    )

    logger.info(
        f"RAGResponse | has_answer={has_answer} | "
        f"answer_len={len(answer)} chars | latency={latency_s:.2f}s"
    )
    return response


# ─────────────────────────────────────────────────────────────
# Utility: hiển thị response đẹp trên console
# ─────────────────────────────────────────────────────────────

def format_response_for_display(response: RAGResponse) -> str:
    """
    Format RAGResponse thành chuỗi dễ đọc trên terminal.

    Args:
        response: RAGResponse từ parse_response().

    Returns:
        Chuỗi đã format với borders, emoji, sections.
    """
    lines = [
        f"\n{'═' * 65}",
        f"  ❓ CÂU HỎI: {response.query}",
        f"{'─' * 65}",
        f"  💬 TRẢ LỜI:\n",
    ]

    # Indent mỗi dòng câu trả lời 2 spaces
    for line in response.answer.split("\n"):
        lines.append(f"  {line}")

    lines.append(f"\n{'─' * 65}")

    if response.sources:
        lines.append(f"  📚 NGUỒN THAM KHẢO ({response.context_used} đoạn):")
        seen = set()
        for s in response.sources:
            fname = s.get("file_name", "N/A")
            cidx  = s.get("chunk_index", "?")
            score = s.get("rerank_score", s.get("score", 0.0))
            key   = f"{fname}_{cidx}"
            if key not in seen:
                lines.append(f"     • {fname} (đoạn #{cidx}, score={score:.3f})")
                seen.add(key)

    lines.append(f"  ⏱  Thời gian: {response.latency_s:.2f}s")
    lines.append(f"{'═' * 65}\n")

    return "\n".join(lines)
