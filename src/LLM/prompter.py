# src/llm/prompter.py
import re
from typing import List, Dict

class AcademicPrompter:
    @staticmethod
    def is_vietnamese(text: str) -> bool:
        """Kiểm tra câu hỏi có phải tiếng Việt hay không."""
        vn_chars = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
        return bool(re.search(f"[{vn_chars}]", text.lower()))

    @classmethod
    def build_prompt(cls, query: str, context_chunks: List[Dict]) -> str:
        is_vn = cls.is_vietnamese(query)

        # 1. Xử lý kịch bản Abstention (Không có tài liệu đạt ngưỡng)
        if not context_chunks:
            system_msg = (
                "Bạn là trợ lý học thuật thư viện số.\n"
                "Hiện tại kho tri thức không tìm thấy tài liệu nào đủ độ tin cậy để trả lời câu hỏi này."
            )
            user_msg = (
                f"Câu hỏi: {query}\n"
                "Hãy thông báo ngắn gọn và lịch sự cho người dùng rằng kho tri thức hiện tại chưa có tài liệu phù hợp để trả lời."
            )
            return (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        # 2. Xử lý kịch bản Grounding + Same-Language
        lang_instruction = (
            "- Ngôn ngữ bắt buộc: Câu hỏi được đặt bằng TIẾNG VIỆT, bạn PHẢI trả lời hoàn toàn bằng TIẾNG VIỆT (dù tài liệu tham khảo là tiếng Anh).\n"
            if is_vn else
            "- Mandatory Language: The query is in ENGLISH. You MUST answer entirely in ENGLISH.\n"
        )

        system_instruction = (
            "Bạn là trợ lý học thuật thông minh của thư viện số trường đại học.\n"
            "QUY TẮC RÀNG BUỘC NGHIÊM NGẶT (STRICT GROUNDING):\n"
            "1. CHỈ sử dụng thông tin và sự thật được nêu trực tiếp trong [Tài liệu tham khảo].\n"
            "2. TUYỆT ĐỐI KHÔNG tự suy diễn, không dùng kiến thức ngoài để bịa đặt thông tin (Anti-hallucination).\n"
            "3. Nếu tài liệu tham khảo không cung cấp đủ chi tiết để trả lời trọn vẹn câu hỏi, hãy chỉ nêu những gì có trong tài liệu và nói rõ phần thông tin còn thiếu.\n"
            f"{lang_instruction}"
            "- Đảm bảo văn phong học thuật, súc tích, giải thích đúng trọng tâm."
        )

        formatted_pieces = []
        for i, c in enumerate(context_chunks, 1):
            sub = c.get("subject", "N/A")
            title = c.get("title", "N/A")
            text = c.get("text", "").strip().replace("\n", " ")
            formatted_pieces.append(f"[Tài liệu {i}] (Môn: {sub} - Chủ đề: {title}):\n{text}")
        context_str = "\n\n".join(formatted_pieces)

        prompt = (
            f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Dưới đây là các tài liệu tham khảo trích xuất từ thư viện:\n"
            f"----------------------------------------\n"
            f"{context_str}\n"
            f"----------------------------------------\n\n"
            f"Câu hỏi: {query}\n"
            f"Hãy trả lời câu hỏi trên dựa hoàn toàn vào các tài liệu tham khảo đã cung cấp:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt