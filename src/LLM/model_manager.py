# src/llm/model_manager.py
import requests
import json
from config import LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE

class LocalLLMManager:
    def __init__(self, model_name: str = "qwen2.5:1.5b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"
        print(f"[*] Đang kết nối tới Ollama Engine tại: {self.api_url} (Model: {self.model_name})...")
        
        # Kiểm tra kết nối tới Ollama server
        try:
            res = requests.get(base_url, timeout=3)
            if res.status_code == 200:
                print(f"[✓] Kết nối Ollama thành công! Sẵn sàng sinh văn bản tốc độ cao.")
            else:
                print(f"[!] Cảnh báo: Ollama phản hồi với mã trạng thái {res.status_code}.")
        except requests.exceptions.RequestException:
            print("[X] LỖI: Không thể kết nối tới Ollama! Hãy chắc chắn bạn đã chạy Ollama trên máy.")

    def generate(self, prompt: str, max_new_tokens: int = LLM_MAX_NEW_TOKENS, temperature: float = LLM_TEMPERATURE) -> str:
        """Gửi prompt tới Ollama C++ Engine và nhận về câu trả lời trong vài giây."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "").strip()
                
                # Hậu xử lý chống đứt đoạn câu
                if text and not text.endswith(('.', '!', '?', ':', '"', '”', '```')):
                    last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'), text.rfind('\n'))
                    if last_punct > len(text) * 0.6:
                        text = text[:last_punct + 1]
                return text
            else:
                return f"[Lỗi Ollama]: Mã lỗi HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"[Lỗi kết nối]: {e}"