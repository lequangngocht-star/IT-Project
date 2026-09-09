import os
import re
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List

PAGES_PER_SUBJECT = 50  

# Danh mục các Chuyên mục (Category) chính thức trên Wikipedia cho từng môn học
CATEGORY_MAP: Dict[str, str] = {
    # 1. Cơ sở ngành & Thuật toán
    "DSA": "Category:Data structures",
    "Algorithms": "Category:Algorithms",
    "DiscreteMath": "Category:Discrete mathematics",
    "OOP": "Category:Object-oriented programming",
    "FunctionalProg": "Category:Functional programming",
    "CompTheory": "Category:Theory of computation",

    # 2. Kiến trúc & Hệ thống
    "CompArch": "Category:Computer architecture",
    "OS": "Category:Operating systems",
    "Compilers": "Category:Compiler construction",
    "DistributedSys": "Category:Distributed computing",
    "CloudComputing": "Category:Cloud computing",

    # 3. Mạng & Bảo mật
    "Networks": "Category:Computer networking",
    "CyberSec": "Category:Computer security",
    "Crypto": "Category:Cryptography",
    "WebSec": "Category:Web security",

    # 4. Cơ sở dữ liệu & Dữ liệu lớn
    "DB": "Category:Database management systems",
    "NoSQL": "Category:NoSQL",
    "DataWarehouse": "Category:Data warehousing",
    "BigData": "Category:Big data",
    "DataMining": "Category:Data mining",

    # 5. Trí tuệ nhân tạo & Thị giác/Ngôn ngữ
    "AI": "Category:Artificial intelligence",
    "ML": "Category:Machine learning",
    "DL": "Category:Deep learning",
    "CV": "Category:Computer vision",
    "NLP": "Category:Natural language processing",

    # 6. Kỹ nghệ phần mềm & Vận hành
    "SE": "Category:Software engineering",
    "SoftwareTesting": "Category:Software testing",
    "DevOps": "Category:DevOps"
}

class AutoWikipediaCrawler:
    def __init__(self, output_dir: str = "data/raw", target_per_subject: int = PAGES_PER_SUBJECT):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_per_subject = target_per_subject
        self.headers = {"User-Agent": "AcademicLibraryBot/2.0 (University Research; contact@tdtu.edu.vn)"}

    def _get_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return {}

    def get_pages_from_category(self, category_name: str, limit: int) -> List[str]:
        """Tự động quét danh sách các bài viết thuộc về một Category trên Wikipedia."""
        titles = []
        cmcontinue = ""
        
        while len(titles) < limit:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category_name,
                "cmlimit": min(50, limit - len(titles)),
                "cmtype": "page",  # Chỉ lấy trang bài viết, bỏ sub-category
                "format": "json"
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
            data = self._get_json(url)
            
            members = data.get("query", {}).get("categorymembers", [])
            for m in members:
                titles.append(m["title"])
                if len(titles) >= limit:
                    break

            # Kiểm tra xem còn trang tiếp theo để phân trang không
            if "continue" in data:
                cmcontinue = data["continue"].get("cmcontinue", "")
            else:
                break
                
        return titles

    def fetch_page_content(self, title: str) -> str:
        """Tải nội dung text sạch của bài viết."""
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": True,
            "titles": title,
            "redirects": 1
        }
        url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
        data = self._get_json(url)
        pages = data.get("query", {}).get("pages", {})
        for _, page_data in pages.items():
            if "extract" in page_data:
                return page_data["extract"]
        return ""

    def clean_text(self, text: str) -> str:
        # Cắt bỏ phần tài liệu tham khảo và ghi chú ở cuối trang
        text = re.split(r"\n== (References|See also|Further reading|External links|Notes) ==", text)[0]
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def run(self):
        print(f"[*] Bắt đầu cào tự động: {self.target_per_subject} bài/môn...")
        total_saved = 0

        for subject_code, category in CATEGORY_MAP.items():
            print(f"\n=======================================================")
            print(f"[*] Môn: {subject_code} | Mục tiêu: {self.target_per_subject} bài")
            print(f"=======================================================")
            
            # Lấy danh sách tiêu đề tự động từ Category
            page_titles = self.get_pages_from_category(category, self.target_per_subject)
            saved_count = 0

            for idx, title in enumerate(page_titles, 1):
                raw_text = self.fetch_page_content(title)
                cleaned = self.clean_text(raw_text)

                # Bỏ qua các trang quá ngắn (dưới 80 từ)
                if len(cleaned.split()) < 80:
                    continue

                safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", title)
                file_name = self.output_dir / f"{subject_code}_{saved_count+1:03d}_{safe_title}.txt"

                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(f"Subject: {subject_code}\nTitle: {title}\nSource: Wikipedia {category}\n\n")
                    f.write(cleaned)

                saved_count += 1
                total_saved += 1
                print(f"  [{saved_count}/{len(page_titles)}] Đã lưu: {file_name.name}")
                time.sleep(0.05)

            print(f"[✓] Hoàn thành môn {subject_code}: {saved_count} bài.")

        print(f"\n[✓] TOÀN BỘ HOÀN TẤT! Tổng cộng {total_saved} tài liệu đã lưu tại {self.output_dir.resolve()}")

if __name__ == "__main__":
    crawler = AutoWikipediaCrawler()
    crawler.run()