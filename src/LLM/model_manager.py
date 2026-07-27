"""
src/llm/model_manager.py
-------------------------
Chịu trách nhiệm DUY NHẤT: tải LLM local và sinh văn bản (generation).

Tại sao tách riêng model_manager.py?
- Tách biệt "load model" khỏi "xây dựng prompt" (prompter.py).
- Dễ swap model: đổi Qwen → Gemma → Llama chỉ bằng thay model_name trong config.
- Quantization logic tập trung một chỗ, không rải rác.

Mô hình được hỗ trợ (đều chạy local qua HuggingFace, không cần API):
    Qwen/Qwen2.5-1.5B-Instruct   (~3GB RAM)   ← mặc định, nhẹ nhất
    Qwen/Qwen2.5-3B-Instruct     (~6GB RAM)
    Qwen/Qwen2.5-7B-Instruct     (~14GB RAM)
    google/gemma-2-2b-it          (~5GB RAM)
    meta-llama/Llama-3.2-1B-Instruct (~2.5GB RAM)

Quantization (giảm RAM):
    4-bit (LLM_LOAD_IN_4BIT=True): ~75% ít RAM hơn, chất lượng giảm nhẹ
    8-bit (LLM_LOAD_IN_8BIT=True): ~50% ít RAM hơn, chất lượng gần như nguyên
    Cần cài: pip install bitsandbytes accelerate
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    LLM_MODEL_NAME,
    LLM_DEVICE,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_REPETITION_PENALTY,
    LLM_LOAD_IN_8BIT,
    LLM_LOAD_IN_4BIT,
)
from src.logger import get_logger

logger = get_logger(__name__)


class LLMManager:
    """
    Quản lý vòng đời của LLM local: load → generate → (optional) unload.

    Thiết kế lazy loading:
    - __init__ chỉ lưu config, chưa load model.
    - Model được load khi gọi load() hoặc lần đầu gọi generate().
    - Tiết kiệm RAM khi chạy pipeline offline (M1/M2) mà không cần LLM.

    Attributes:
        model_name:  HuggingFace model ID.
        device:      "auto", "cpu", "cuda", "cuda:0"...
        is_loaded:   True nếu model đã được load vào memory.
    """

    def __init__(
        self,
        model_name:         str   = LLM_MODEL_NAME,
        device:             str   = LLM_DEVICE,
        load_in_8bit:       bool  = LLM_LOAD_IN_8BIT,
        load_in_4bit:       bool  = LLM_LOAD_IN_4BIT,
        max_new_tokens:     int   = LLM_MAX_NEW_TOKENS,
        temperature:        float = LLM_TEMPERATURE,
        top_p:              float = LLM_TOP_P,
        repetition_penalty: float = LLM_REPETITION_PENALTY,
    ) -> None:
        """
        Khai báo config cho LLM — chưa load model vào RAM.

        Args:
            model_name:         HuggingFace model ID (local hoặc Hub).
            device:             Device để chạy model.
            load_in_8bit:       Bật 8-bit quantization (cần bitsandbytes).
            load_in_4bit:       Bật 4-bit quantization (cần bitsandbytes).
            max_new_tokens:     Số token tối đa sinh ra.
            temperature:        Độ ngẫu nhiên (0.0 = deterministic).
            top_p:              Nucleus sampling threshold.
            repetition_penalty: Tránh lặp lại token (1.0 = không phạt).
        """
        if load_in_4bit and load_in_8bit:
            raise ValueError("Chỉ được bật một trong load_in_4bit hoặc load_in_8bit")

        self.model_name         = model_name
        self.device             = device
        self._load_in_8bit      = load_in_8bit
        self._load_in_4bit      = load_in_4bit
        self._max_new_tokens    = max_new_tokens
        self._temperature       = temperature
        self._top_p             = top_p
        self._repetition_penalty = repetition_penalty

        # Chưa load — lazy initialization
        self._model     = None
        self._tokenizer = None

        quant = "4-bit" if load_in_4bit else ("8-bit" if load_in_8bit else "full precision")
        logger.info(
            f"LLMManager config | model={model_name} | "
            f"device={device} | quant={quant}"
        )

    # ─────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────

    def load(self) -> "LLMManager":
        """
        Load model và tokenizer vào memory.

        Tự động xử lý:
        - Quantization config nếu 4-bit/8-bit được bật.
        - device_map="auto" để HuggingFace phân bổ layers tối ưu.
        - torch_dtype=bfloat16 trên GPU để tiết kiệm VRAM.

        Returns:
            self — để có thể chain: manager = LLMManager().load()

        Raises:
            ImportError: Nếu transformers chưa cài.
            RuntimeError: Nếu model không tải được.
        """
        if self.is_loaded:
            logger.info("Model đã load rồi — bỏ qua")
            return self

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "Cài đặt: pip install transformers torch accelerate"
            )

        t0 = time.time()
        logger.info(f"Đang load model: {self.model_name}")
        logger.info("(Lần đầu sẽ download từ HuggingFace Hub...)")

        # ── Tokenizer ────────────────────────────────────────
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,   # Cần cho Qwen
        )
        # Đảm bảo pad_token tồn tại (một số model thiếu)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # ── Quantization config ───────────────────────────────
        quantization_config = None
        if self._load_in_4bit or self._load_in_8bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self._load_in_4bit,
                    load_in_8bit=self._load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16 if self._load_in_4bit else None,
                    bnb_4bit_quant_type="nf4" if self._load_in_4bit else None,
                )
                logger.info(
                    f"Quantization: {'4-bit NF4' if self._load_in_4bit else '8-bit'}"
                )
            except Exception as e:
                logger.warning(
                    f"Không tạo được BitsAndBytesConfig: {e}\n"
                    "  → Tiếp tục với full precision. Cài bitsandbytes nếu cần."
                )
                quantization_config = None

        # ── Model ─────────────────────────────────────────────
        # device_map="auto": HF tự phân bổ layers vào GPU/CPU
        # torch_dtype="auto": dùng dtype model đề xuất (bfloat16 với Qwen)
        model_kwargs = dict(
            trust_remote_code=True,
            device_map=self.device if self.device != "cpu" else None,
            torch_dtype="auto",
        )
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        # Với CPU, không dùng device_map
        if self.device == "cpu":
            model_kwargs.pop("device_map", None)
            model_kwargs["torch_dtype"] = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Nếu device là CPU và không có quantization, chuyển model sang CPU
        if self.device == "cpu" and quantization_config is None:
            self._model = self._model.to("cpu")

        self._model.eval()   # Tắt dropout — chỉ inference, không train

        elapsed = time.time() - t0
        logger.info(f"Model loaded | {elapsed:.1f}s | device={self.device}")
        return self

    # ─────────────────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens:     Optional[int]   = None,
        temperature:        Optional[float] = None,
        top_p:              Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Sinh văn bản từ prompt cho trước.

        Tự động load model nếu chưa load (lazy loading).

        Args:
            prompt:             Chuỗi prompt đầy đủ (đã bao gồm system message + context).
            max_new_tokens:     Override config mặc định.
            temperature:        Override config mặc định.
            top_p:              Override config mặc định.
            repetition_penalty: Override config mặc định.

        Returns:
            Chuỗi văn bản sinh ra (không bao gồm phần prompt đầu vào).

        Raises:
            RuntimeError: Nếu không load được model.
        """
        if not self.is_loaded:
            self.load()

        import torch

        # Merge params — override nếu được truyền vào
        gen_config = {
            "max_new_tokens":     max_new_tokens     or self._max_new_tokens,
            "temperature":        temperature         or self._temperature,
            "top_p":              top_p               or self._top_p,
            "repetition_penalty": repetition_penalty or self._repetition_penalty,
            "do_sample":          (temperature or self._temperature) > 0.0,
            "pad_token_id":       self._tokenizer.pad_token_id,
            "eos_token_id":       self._tokenizer.eos_token_id,
        }

        # Tokenize prompt
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,    # Tránh vượt context window model
        )

        # Chuyển inputs sang đúng device của model
        input_device = next(self._model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # Generate
        t0 = time.time()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                **gen_config,
            )

        # Cắt bỏ phần prompt — chỉ lấy token mới sinh ra
        new_token_ids = output_ids[0][input_len:]
        response = self._tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        elapsed = time.time() - t0
        logger.info(
            f"Generate | {len(new_token_ids)} tokens | {elapsed:.2f}s | "
            f"speed={len(new_token_ids)/elapsed:.1f} tok/s"
        )
        return response

    def generate_chat(
        self,
        messages: list[dict],
        **kwargs,
    ) -> str:
        """
        Sinh câu trả lời từ list messages theo format Chat Template.

        Chat Template là cách chuẩn để format conversation với instruct models.
        Mỗi model có template riêng (Qwen dùng ChatML, Llama dùng template khác).
        apply_chat_template() xử lý điều này tự động.

        Args:
            messages: List dict, mỗi dict có keys "role" và "content".
                      role: "system" | "user" | "assistant"
            **kwargs: Truyền thẳng vào generate() (max_new_tokens, temperature...).

        Returns:
            Chuỗi câu trả lời từ model.

        Example:
            messages = [
                {"role": "system", "content": "Bạn là trợ lý thư viện..."},
                {"role": "user", "content": "Machine Learning là gì?"},
            ]
        """
        if not self.is_loaded:
            self.load()

        # apply_chat_template format messages theo template của model cụ thể
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,   # Thêm dấu hiệu bắt đầu sinh
        )
        return self.generate(prompt, **kwargs)

    # ─────────────────────────────────────────────────────────
    # Properties & Utilities
    # ─────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True nếu model và tokenizer đã được load."""
        return self._model is not None and self._tokenizer is not None

    def unload(self) -> None:
        """
        Giải phóng model khỏi RAM/VRAM.
        Gọi khi không cần LLM nữa trong cùng process.
        """
        import gc
        self._model     = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("LLM unloaded — RAM đã được giải phóng")

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        quant  = "4bit" if self._load_in_4bit else ("8bit" if self._load_in_8bit else "fp")
        return (
            f"LLMManager("
            f"model='{self.model_name}', "
            f"quant={quant}, "
            f"status={status})"
        )

    def __enter__(self) -> "LLMManager":
        """Context manager: tự động load khi vào block."""
        return self.load()

    def __exit__(self, *_) -> None:
        """Context manager: tự động unload khi ra block."""
        self.unload()
