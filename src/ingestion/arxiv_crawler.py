"""
src/ingestion/arxiv_crawler.py
------------------------------
Module tự động thu thập học liệu số quy mô lớn từ arXiv API.
Phân rã danh mục và tải chính xác số lượng tài liệu theo yêu cầu cấu hình.
"""

import sys
import time
from pathlib import Path
import urllib.request

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_RAW_DIR
from src.logger import get_logger

logger = get_logger(__name__)

try:
    import arxiv
    from tqdm import tqdm
except ImportError:
    raise ImportError("Vui lòng cài đặt đầy đủ thư viện phụ trợ: pip install arxiv tqdm")


# Cấu hình danh mục và số lượng chính xác theo yêu cầu hệ thống
TARGET_DATASET_CONFIG = [
    {"topic": "Artificial Intelligence", "query": "cat:cs.AI", "count": 150},
    {"topic": "Machine Learning",        "query": "cat:cs.LG", "count": 150},
    {"topic": "Deep Learning",           "query": 'cat:cs.LG AND "Deep Learning"', "count": 100},
    {"topic": "NLP",                     "query": "cat:cs.CL", "count": 100},
    {"topic": "Computer Vision",         "query": "cat:cs.CV", "count": 100},
    {"topic": "Database",                "query": "cat:cs.DB", "count": 100},
    {"topic": "Computer Networks",       "query": "cat:cs.NI", "count": 100},
    {"topic": "Operating Systems",       "query": "cat:cs.OS", "count": 100},
    {"topic": "Software Engineering",    "query": "cat:cs.SE", "count": 100},
]


def download_paper_with_retry(result, clean_topic: str, output_dir: Path, max_retries: int = 3) -> bool:
    """Tải tệp PDF từ arXiv sử dụng urllib với cơ chế tự động thử lại khi gặp sự cố mạng."""
    # 1. Làm sạch tên tệp tin, loại bỏ ký tự đặc biệt của hệ điều hành
    clean_title = "".join([c if c.isalnum() or c in " _-" else "_" for c in result.title])
    file_name = f"{clean_topic}_{result.get_short_id()}_{clean_title[:40]}.pdf"
    target_path = output_dir / file_name
    
    # 2. Lấy link tải PDF trực tiếp từ thuộc tính của bài báo
    pdf_url = result.pdf_url
    if not pdf_url:
        logger.error(f"  [-] Không tìm thấy liên kết PDF cho tệp: {result.get_short_id()}")
        return bool(False)

    # 3. Tiến hành tải bằng urllib kèm cơ chế Retry
    for attempt in range(max_retries):
        try:
            # Thiết lập Header User-Agent giả lập để tránh bị máy chủ chặn request
            req = urllib.request.Request(
                pdf_url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
                out_file.write(response.read())
            return bool(True)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"  [-] Thất bại khi tải {result.get_short_id()} qua URL {pdf_url}: {e}")
            else:
                logger.warning(f"  [-] Lần thử {attempt + 1} thất bại, đang thử lại sau 3 giây...")
                time.sleep(3)  # Giãn cách thời gian tải để tránh nghẽn
    return bool(False)


def run_bulk_arxiv_crawler(output_dir: Path = DATA_RAW_DIR):
    """Vận hành toàn mạch cào dữ liệu lớn tuần tự theo cấu hình."""
    output_dir.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    
    print("==================================================")
    print("🚀 KHỞI ĐỘNG HỆ THỐNG THU THẬP KHO HỌC LIỆU SỐ ARXIV")
    print("==================================================")
    
    total_downloaded = 0
    
    for config in TARGET_DATASET_CONFIG:
        topic = config["topic"]
        query = config["query"]
        target_count = config["count"]
        
        clean_topic = topic.replace(" ", "_")
        logger.info(f"[*] Bắt đầu xử lý chủ đề: {topic} | Mục tiêu: {target_count} PDFs")
        
        # Thiết lập tìm kiếm trên hệ thống API
        search = arxiv.Search(
            query=query,
            max_results=target_count,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        try:
            results = list(client.results(search))
            actual_results_count = len(results)
            logger.info(f"  -> Tìm thấy {actual_results_count} bài báo tương thích trên hệ thống.")
            
            if actual_results_count == 0:
                continue
                
            # Khởi tạo thanh tiến trình hiển thị trực quan trực tiếp trên Terminal
            success_count = 0
            with tqdm(total=min(target_count, actual_results_count), desc=f" Downloading {topic}") as pbar:
                for result in results:
                    # Kiểm tra xem file đã tồn tại hay chưa để tránh tải trùng tốn băng thông
                    clean_title = "".join([c if c.isalnum() or c in " _-" else "_" for c in result.title])
                    expected_name = f"{clean_topic}_{result.get_short_id()}_{clean_title[:40]}.pdf"
                    
                    if (output_dir / expected_name).exists():
                        success_count += 1
                        pbar.update(1)
                        continue
                        
                    if download_paper_with_retry(result, clean_topic, output_dir):
                        success_count += 1
                        pbar.update(1)
                        
            total_downloaded += success_count
            logger.info(f"  [+] Đã hoàn thành tải về: {success_count}/{target_count} PDFs cho nhóm {topic}\n")
            
        except Exception as e:
            logger.error(f"[-] Gặp sự cố nghiêm trọng khi quét chủ đề {topic}: {e}\n")
            
    print("==================================================")
    print(f"✅ HOÀN THÀNH TIẾN TRÌNH CRAWLER DATA ARXIV")
    print(f"[+] Tổng số lượng tài liệu hiện có tại kho data/raw/: {total_downloaded} PDFs")
    print("==================================================")


if __name__ == "__main__":
    run_bulk_arxiv_crawler()