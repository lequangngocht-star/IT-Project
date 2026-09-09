import time
from pathlib import Path
from src.preprocessing.chunker import process_file_to_chunks, save_chunks

def run_pipeline_m1(raw_dir: str = "data/raw", output_json: str = "data/processed/chunks.json"):
    start_time = time.time()
    raw_path = Path(raw_dir)
    txt_files = list(raw_path.glob("*.txt"))
    
    print("=" * 60)
    print(f"[*] BẮT ĐẦU OFFLINE PIPELINE - MILESTONE 1 (Ingestion & Chunking)")
    print(f"[*] Tìm thấy {len(txt_files)} tệp dữ liệu thô trong {raw_path.resolve()}")
    print("=" * 60)

    if not txt_files:
        print("[-] Không tìm thấy file txt nào trong data/raw/! Hãy kiểm tra lại bước crawl.")
        return

    all_chunks = []
    for idx, file_path in enumerate(txt_files, 1):
        file_chunks = process_file_to_chunks(file_path)
        all_chunks.extend(file_chunks)
        if idx % 50 == 0 or idx == len(txt_files):
            print(f" -> Đã xử lý [{idx}/{len(txt_files)}] file | Tổng số chunks hiện tại: {len(all_chunks)}")

    # Đóng gói và lưu dữ liệu kèm metadata
    save_chunks(all_chunks, output_json)
    
    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"    - Tổng tài liệu xử lý: {len(txt_files)} files")
    print(f"    - Tổng số chunks sinh ra: {len(all_chunks)} chunks")
    print(f"    - Thời gian thực thi: {elapsed:.2f} giây")
    print(f"    - File tri thức xuất xưởng: {output_json}")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline_m1()