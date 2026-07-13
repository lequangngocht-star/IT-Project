"""
src/logger.py
-------------
Cấu hình hệ thống Logger đồng bộ ra cả màn hình Terminal và tệp tin logs/rag.log.
"""

import logging
import sys
from pathlib import Path

# Thêm root vào sys.path để import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LOG_FILE, LOG_LEVEL

def get_logger(name: str) -> logging.Logger:
    """Khởi tạo hoặc tái sử dụng bộ cấu hình Logger theo chuẩn định dạng."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        
        # Định dạng nội dung bản ghi log (Thời gian - Tên Module - Cấp độ - Tin nhắn)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler 1: Xuất luồng log ra file vật lý lưu trữ lâu dài
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Handler 2: Đẩy luồng log hiển thị trực tiếp trên Terminal để theo dõi nhanh
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger