# app.py
import time
import json
import os
from pathlib import Path
import requests
import streamlit as st

# 1. CẤU HÌNH GIAO DIỆN HỌC THUẬT
st.set_page_config(
    page_title="Thư Viện Số CNTT - Hệ Thống RAG & Tra Cứu Học Liệu",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy biến CSS nâng cao độ tương phản và thẩm mỹ
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 16px 22px;
        border-radius: 10px;
        color: white;
        margin-bottom: 15px;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.65rem; font-weight: 700; }
    .main-header p { color: #E0E7FF; margin: 4px 0 0 0; font-size: 0.88rem; }
    
    .perf-container {
        display: flex;
        gap: 10px;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .perf-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 6px 10px;
        flex: 1;
        text-align: center;
    }
    .perf-label { font-size: 0.7rem; color: #64748B; font-weight: 600; text-transform: uppercase; }
    .perf-value { font-size: 0.95rem; color: #0F172A; font-weight: 700; }
    
    .source-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
    }
    .source-tag {
        background: #EFF6FF;
        color: #1D4ED8;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .score-tag {
        background: #F0FDF4;
        color: #15803D;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .doc-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# 2. KHỞI TẠO PIPELINE TRUY XUẤT (CACHE TÀI NGUYÊN)
@st.cache_resource(show_spinner=False)
def load_retrieval_pipeline():
    from config import RETRIEVAL_TOP_K, RERANKER_TOP_N
    from src.retrieval.searcher import DenseRetriever
    from src.retrieval.reranker import CrossEncoderReranker
    
    retriever = DenseRetriever(top_k=RETRIEVAL_TOP_K)
    reranker = CrossEncoderReranker()
    return retriever, reranker

# 3. QUẢN LÝ KHO DỮ LIỆU GỐC 1,184 TÀI LIỆU
@st.cache_data(show_spinner=False)
def load_library_catalog():
    """Quét thư mục data/raw/ để lập danh mục tài liệu theo môn học."""
    raw_dir = Path("data/raw")
    catalog = {}
    
    if not raw_dir.exists():
        return catalog

    # Quét tất cả các file trong thư mục raw
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in [".txt", ".pdf", ".docx", ".json"]:
            # Nếu phân theo thư mục con (data/raw/DB/file.txt)
            if path.parent != raw_dir:
                subject = path.parent.name.upper()
            else:
                # Nếu đặt tên theo tiền tố (DB_file.txt hoặc DB - file.txt)
                name_parts = path.stem.replace("-", "_").split("_")
                subject = name_parts[0].upper() if len(name_parts) > 1 else "CHUNG"
                
            if subject not in catalog:
                catalog[subject] = []
                
            catalog[subject].append({
                "file_name": path.name,
                "file_path": str(path),
                "size_kb": round(path.stat().st_size / 1024, 1),
                "extension": path.suffix.lower()
            })
            
    return catalog

st.markdown("""
<div class="main-header">
    <h1>🎓 Thư Viện Số Chuyên Ngành CNTT - Hệ Thống RAG & Kho Học Liệu</h1>
    <p>Kiến trúc Advanced RAG: BAAI/bge-m3 • BGE-Reranker-v2-m3 • Ollama Qwen2.5 & Quản lý 1,184 tài liệu học thuật</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Đang kết nối kho vector FAISS và nạp mô hình Reranker..."):
    retriever, reranker = load_retrieval_pipeline()

library_catalog = load_library_catalog()
total_docs_found = sum(len(docs) for docs in library_catalog.values())

# 4. THANH ĐIỀU KHIỂN (SIDEBAR)
with st.sidebar:
    st.markdown("### ⚙️ Thông Số Hệ Thống")
    st.info(
        f"📚 **Tổng số tài liệu:** {total_docs_found:,} files\n\n"
        f"🏛 **Số phân môn:** {len(library_catalog)} chuyên đề\n\n"
        "🧠 **Embedding:** BAAI/bge-m3 (1024-dim)\n\n"
        "⚡ **Reranker:** BGE-Reranker-v2-m3\n\n"
        "🤖 **LLM Engine:** Ollama / Qwen2.5-1.5B"
    )
    
    st.divider()
    st.markdown("### 🎛 Tham Số RAG")
    top_k = st.slider("Ứng viên Dense Search (Top-K):", 4, 15, 6, 1)
    top_n = st.slider("Ngữ cảnh đưa vào LLM (Top-N):", 1, 4, 2, 1)
    
    st.divider()
    st.markdown("### 💡 Câu Hỏi Thực Nghiệm Mẫu")
    sample_queries = [
        "What are the four ACID properties in database management systems and what does each guarantee?",
        "Sự khác nhau giữa học có giám sát và học không giám sát trong machine learning?",
        "Thread và Process khác nhau như thế nào?",
        "What is a Binary Search Tree and what is the worst-case time complexity of its search operation?",
        "How does the A* search algorithm determine the optimal path using heuristic functions?"
    ]
    
    for idx, q in enumerate(sample_queries):
        btn_label = q if len(q) < 42 else q[:40] + "..."
        if st.button(f"📌 {btn_label}", key=f"sq_{idx}", use_container_width=True):
            st.session_state["active_query"] = q

    st.divider()
    if st.button("🗑 Làm mới phiên trò chuyện", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

# 5. PHÂN CHIA TAB: HỎI ĐÁP VÀ XEM TÀI LIỆU
tab_chat, tab_docs = st.tabs(["💬 Trợ Lý Hỏi Đáp (RAG Chatbot)", "📚 Tra Cứu Kho Tài Liệu (1,184 Files)"])

# =====================================================================
# TAB 1: GIAO DIỆN TRỢ LÝ HỎI ĐÁP THỜI GIAN THỰC (RAG CHAT)
# =====================================================================
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Xin chào! Tôi là trợ lý học thuật thư viện số. Bạn có thể tra cứu kiến thức học thuật từ 1,184 tài liệu chuyên ngành CNTT bằng cả tiếng Anh và tiếng Việt.",
                "sources": [],
                "perf": None
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant" and msg.get("perf"):
                p = msg["perf"]
                st.markdown(f"""
                <div class="perf-container">
                    <div class="perf-card">
                        <div class="perf-label">Dense Search</div>
                        <div class="perf-value">{p['dense_ms']:.1f} ms</div>
                    </div>
                    <div class="perf-card">
                        <div class="perf-label">Reranker</div>
                        <div class="perf-value">{p['rerank_ms']:.1f} ms</div>
                    </div>
                    <div class="perf-card">
                        <div class="perf-label">Ollama Sinh LLM</div>
                        <div class="perf-value" style="color: #16A34A;">{p['gen_s']:.2f} s</div>
                    </div>
                    <div class="perf-card">
                        <div class="perf-label">Tổng Độ Trễ</div>
                        <div class="perf-value" style="color: #2563EB;">{p['total_s']:.2f} s</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if msg.get("sources"):
                with st.expander(f"📚 Xem {len(msg['sources'])} tài liệu nguồn được trích dẫn làm căn cứ"):
                    for s_idx, src in enumerate(msg["sources"], 1):
                        sub = src.get("subject", "N/A")
                        title = src.get("title", "N/A")
                        score = src.get("rerank_score", 0.0)
                        txt = src.get("text", "")
                        st.markdown(f"""
                        <div class="source-box">
                            <span class="source-tag">Môn: {sub}</span> &nbsp;
                            <span class="source-tag" style="background:#F1F5F9; color:#475569;">Chủ đề: {title}</span> &nbsp;
                            <span class="score-tag">Độ liên quan: {score:.2f}</span>
                            <div style="font-size: 0.88rem; color: #334155; margin-top: 6px; line-height: 1.45;">
                                "{txt[:280]}..."
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    def stream_ollama_response(prompt: str, model_name: str = "qwen2.5:1.5b"):
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": 256, "temperature": 0.1, "top_p": 0.9}
        }
        try:
            response = requests.post(url, json=payload, stream=True, timeout=40)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    yield chunk.get("response", "")
        except Exception as e:
            yield f"\n\n[Lỗi kết nối Ollama]: {e}. Vui lòng khởi động Ollama trên máy."

    user_input = None
    if "active_query" in st.session_state and st.session_state["active_query"]:
        user_input = st.session_state.pop("active_query")
    else:
        user_input = st.chat_input("Nhập câu hỏi tra cứu học thuật tại đây...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            progress_text = st.empty()
            progress_text.caption("🔍 Đang quét cơ sở dữ liệu FAISS và tái chấm điểm qua BGE-Reranker...")
            
            t0 = time.time()
            candidates = retriever.retrieve(user_input, top_k=top_k)
            t_dense = (time.time() - t0) * 1000
            
            t1 = time.time()
            reranked_chunks = reranker.rerank(user_input, candidates, top_n=top_n)
            t_rerank = (time.time() - t1) * 1000
            
            progress_text.empty()

            from src.llm.prompter import AcademicPrompter
            prompt = AcademicPrompter.build_prompt(user_input, reranked_chunks)

            answer_placeholder = st.empty()
            full_response = ""
            t_gen_start = time.time()

            for chunk_token in stream_ollama_response(prompt):
                full_response += chunk_token
                answer_placeholder.markdown(full_response + "▌")
                
            answer_placeholder.markdown(full_response)
            t_gen = time.time() - t_gen_start
            t_total = (t_dense + t_rerank) / 1000 + t_gen

            perf_data = {
                "dense_ms": t_dense,
                "rerank_ms": t_rerank,
                "gen_s": t_gen,
                "total_s": t_total
            }
            
            st.markdown(f"""
            <div class="perf-container">
                <div class="perf-card">
                    <div class="perf-label">Dense Search</div>
                    <div class="perf-value">{t_dense:.1f} ms</div>
                </div>
                <div class="perf-card">
                    <div class="perf-label">Reranker</div>
                    <div class="perf-value">{t_rerank:.1f} ms</div>
                </div>
                <div class="perf-card">
                    <div class="perf-label">Ollama Sinh LLM</div>
                    <div class="perf-value" style="color: #16A34A;">{t_gen:.2f} s</div>
                </div>
                <div class="perf-card">
                    <div class="perf-label">Tổng Độ Trễ</div>
                    <div class="perf-value" style="color: #2563EB;">{t_total:.2f} s</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if reranked_chunks:
                with st.expander(f"📚 Xem {len(reranked_chunks)} tài liệu nguồn được trích dẫn làm căn cứ"):
                    for s_idx, src in enumerate(reranked_chunks, 1):
                        sub = src.get("subject", "N/A")
                        title = src.get("title", "N/A")
                        score = src.get("rerank_score", 0.0)
                        txt = src.get("text", "")
                        st.markdown(f"""
                        <div class="source-box">
                            <span class="source-tag">Môn: {sub}</span> &nbsp;
                            <span class="source-tag" style="background:#F1F5F9; color:#475569;">Chủ đề: {title}</span> &nbsp;
                            <span class="score-tag">Độ liên quan: {score:.2f}</span>
                            <div style="font-size: 0.88rem; color: #334155; margin-top: 6px; line-height: 1.45;">
                                "{txt[:280]}..."
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": reranked_chunks,
                "perf": perf_data
            })

# =====================================================================
# TAB 2: GIAO DIỆN KHÁM PHÁ & XEM HỌC LIỆU SỐ THEO CHỦ ĐỀ
# =====================================================================
with tab_docs:
    st.markdown("### 📖 Khám Phá Kho Học Liệu Số Theo Phân Môn")
    st.caption("Sinh viên có thể lọc theo chuyên đề, tìm kiếm theo tiêu đề bài học và đọc trực tiếp nội dung chi tiết.")

    if not library_catalog:
        st.warning("⚠️ Chưa tìm thấy tài liệu nào trong thư mục `data/raw/`.")
    else:
        col_sub, col_search = st.columns([1, 2])
        with col_sub:
            subject_list = ["TẤT CẢ"] + sorted(list(library_catalog.keys()))
            selected_subject = st.selectbox("📂 Chọn chuyên đề / môn học:", subject_list)
        with col_search:
            search_kw = st.text_input("🔍 Tìm kiếm tên tài liệu:", placeholder="Nhập từ khóa cần tìm (vd: database, thread, search...)")

        # Lọc danh sách tài liệu
        matched_docs = []
        for sub, docs in library_catalog.items():
            if selected_subject != "TẤT CẢ" and sub != selected_subject:
                continue
            for d in docs:
                if search_kw.strip().lower() in d["file_name"].lower():
                    item = d.copy()
                    item["subject"] = sub
                    matched_docs.append(item)

        st.markdown(f"**Tìm thấy `{len(matched_docs)}` tài liệu phù hợp:**")

        # Hiển thị tài liệu dạng thẻ kèm nút đọc nội dung
        for doc in matched_docs[:50]:  # Giới hạn hiển thị 50 file mỗi trang để giao diện mượt
            sub_name = doc["subject"]
            file_name = doc["file_name"]
            size_kb = doc["size_kb"]
            file_path = doc["file_path"]

            with st.expander(f"📄 [{sub_name}] {file_name} ({size_kb} KB)"):
                st.write(f"**Đường dẫn:** `{file_path}`")
                
                # Đọc nội dung nếu là file text
                if doc["extension"] == ".txt":
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            st.text_area("Nội dung trích xuất:", content[:3000] + ("\n... [Còn tiếp]" if len(content) > 3000 else ""), height=250)
                    except Exception as e:
                        st.error(f"Không thể mở file: {e}")
                else:
                    st.info("💡 Tài liệu định dạng nhị phân/PDF. Bạn có thể tải về để xem trọn vẹn:")
                    try:
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label=f"⬇️ Tải xuống {file_name}",
                                data=f,
                                file_name=file_name,
                                mime="application/octet-stream"
                            )
                    except Exception as e:
                        st.error(f"Lỗi tải file: {e}")

        if len(matched_docs) > 50:
            st.info(f"💡 Đang hiển thị 50/{len(matched_docs)} tài liệu. Vui lòng nhập từ khóa tìm kiếm để thu hẹp phạm vi.")