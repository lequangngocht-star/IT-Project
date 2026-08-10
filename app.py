"""
app.py
-------
FastAPI application — REST API cho hệ thống RAG thư viện số.

Đây là lớp giao tiếp giữa frontend / client và toàn bộ pipeline M1→M5.

Endpoints:
    POST /ask                  → Hỏi đáp RAG (query → answer + sources)
    GET  /health               → Kiểm tra trạng thái hệ thống
    GET  /stats                → Thống kê index (số chunks, model info)
    POST /upload               → Upload tài liệu mới vào thư viện
    POST /index/rebuild        → Rebuild FAISS index sau khi upload
    GET  /eval/run             → Chạy evaluation và trả về report
    GET  /history              → Lịch sử truy vấn trong session

Thiết kế:
- Startup: load tất cả model 1 lần, lưu vào app.state.
- Mỗi request dùng lại model đã load → không tốn thời gian reload.
- Async endpoint nhưng model inference vẫn blocking (chạy trong threadpool).
- Pydantic schemas validate input/output nghiêm ngặt.

Chạy:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

    # Production
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
    # workers=1 vì model LLM không thread-safe khi share state
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_RAW_DIR,
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    LLM_MODEL_NAME,
    RETRIEVAL_TOP_K,
    RERANKER_TOP_N,
    RERANKER_ENABLED,
    SUPPORTED_EXTENSIONS,
)
from src.logger import get_logger

logger = get_logger("app")

# ─────────────────────────────────────────────────────────────
# In-memory query history (session-scoped)
# ─────────────────────────────────────────────────────────────
_query_history: list[dict] = []
MAX_HISTORY = 50


# ─────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000,
                       description="Câu hỏi của người dùng")
    top_k: Optional[int] = Field(None, ge=1, le=20,
                                 description="Override số chunk retrieve")
    top_n: Optional[int] = Field(None, ge=1, le=10,
                                 description="Override số chunk vào context LLM")
    use_reranker: Optional[bool] = Field(None,
                                         description="Override bật/tắt reranker")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Điều kiện tiên quyết của môn Machine Learning là gì?",
                "top_k": 10,
                "top_n": 5,
            }
        }
    }


class SourceItem(BaseModel):
    rank:         int
    file_name:    str
    chunk_index:  int
    score:        float
    rerank_score: float


class AskResponse(BaseModel):
    query:        str
    answer:       str
    has_answer:   bool
    context_used: int
    latency_s:    float
    sources:      list[SourceItem]


class HealthResponse(BaseModel):
    status:       str           # "ok" | "degraded" | "error"
    index_loaded: bool
    llm_loaded:   bool
    index_size:   int
    model_name:   str
    uptime_s:     float


class StatsResponse(BaseModel):
    index_size:       int
    embedding_model:  str
    llm_model:        str
    reranker_enabled: bool
    top_k:            int
    top_n:            int
    total_queries:    int


class UploadResponse(BaseModel):
    filename:    str
    size_bytes:  int
    message:     str


class IndexRebuildResponse(BaseModel):
    status:      str
    chunks_count: int
    elapsed_s:   float
    message:     str


# ─────────────────────────────────────────────────────────────
# Startup / Shutdown — load models 1 lần
# ─────────────────────────────────────────────────────────────

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Chạy khi server start: load Retrieval Pipeline + LLM vào app.state.
    Chạy khi server stop: cleanup (nếu cần).

    Dùng asynccontextmanager thay vì on_event (deprecated từ FastAPI 0.93+).
    """
    logger.info("═" * 55)
    logger.info("  RAG LIBRARY API — Đang khởi động...")
    logger.info("═" * 55)

    # Kiểm tra prerequisite
    app.state.index_loaded = False
    app.state.llm_loaded   = False
    app.state.retriever    = None
    app.state.reranker     = None
    app.state.llm          = None

    if not FAISS_INDEX_FILE.exists():
        logger.warning(
            "FAISS index chưa tồn tại. Một số endpoint sẽ không hoạt động.\n"
            "  → Chạy: python src/pipeline_m1.py && python src/pipeline_m2.py"
        )
    else:
        try:
            from src.pipeline_m3 import build_retrieval_pipeline
            retriever, reranker = build_retrieval_pipeline(
                use_reranker=RERANKER_ENABLED,
                top_k=RETRIEVAL_TOP_K,
                top_n=RERANKER_TOP_N,
            )
            app.state.retriever    = retriever
            app.state.reranker     = reranker
            app.state.index_loaded = True
            logger.info("✅ Retrieval Pipeline sẵn sàng")
        except Exception as e:
            logger.error(f"Không load được Retrieval Pipeline: {e}")

        try:
            from src.LLM.model_manager import LLMManager
            llm = LLMManager()
            llm.load()
            app.state.llm        = llm
            app.state.llm_loaded = True
            logger.info(f"✅ LLM sẵn sàng: {LLM_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Không load được LLM: {e}")

    logger.info("  API sẵn sàng tại http://localhost:8000")
    logger.info("  Docs: http://localhost:8000/docs")
    logger.info("═" * 55)

    yield  # ← Server chạy ở đây

    # Shutdown cleanup
    logger.info("Server đang tắt...")


# ─────────────────────────────────────────────────────────────
# App instance
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Library API",
    description=(
        "Hệ thống hỏi đáp thông minh cho Thư viện số Trường Đại học ABC.\n\n"
        "Sử dụng kiến trúc RAG (Retrieval-Augmented Generation) với BGE-M3 embedding "
        "và LLM mã nguồn mở chạy local."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Production: thay bằng domain cụ thể
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _require_index(app_state) -> None:
    """Raise 503 nếu FAISS index chưa được load."""
    if not app_state.index_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "FAISS index chưa sẵn sàng. "
                "Chạy pipeline_m1.py và pipeline_m2.py trước, "
                "sau đó restart server."
            ),
        )


def _require_llm(app_state) -> None:
    """Raise 503 nếu LLM chưa được load."""
    if not app_state.llm_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM chưa sẵn sàng. "
                "Kiểm tra log server để biết lý do."
            ),
        )


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Kiểm tra trạng thái hệ thống.
    Dùng để monitoring, load balancer health check.
    """
    index_size = 0
    if app.state.index_loaded and app.state.retriever:
        try:
            index_size = app.state.retriever._vector_store.size
        except Exception:
            pass

    all_ok = app.state.index_loaded and app.state.llm_loaded
    return HealthResponse(
        status      = "ok" if all_ok else "degraded",
        index_loaded = app.state.index_loaded,
        llm_loaded   = app.state.llm_loaded,
        index_size   = index_size,
        model_name   = LLM_MODEL_NAME,
        uptime_s     = round(time.time() - _start_time, 1),
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Thống kê chi tiết về cấu hình và trạng thái hệ thống."""
    index_size = 0
    if app.state.index_loaded and app.state.retriever:
        try:
            index_size = app.state.retriever._vector_store.size
        except Exception:
            pass

    return StatsResponse(
        index_size       = index_size,
        embedding_model  = "BAAI/bge-m3",
        llm_model        = LLM_MODEL_NAME,
        reranker_enabled = RERANKER_ENABLED,
        top_k            = RETRIEVAL_TOP_K,
        top_n            = RERANKER_TOP_N,
        total_queries    = len(_query_history),
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(request: AskRequest):
    """
    **Endpoint chính** — Hỏi đáp RAG.

    Nhận câu hỏi tiếng Việt, trả về:
    - `answer`: Câu trả lời từ LLM dựa trên tài liệu thư viện.
    - `sources`: Danh sách tài liệu nguồn được trích dẫn.
    - `has_answer`: False nếu LLM không tìm thấy thông tin.
    - `latency_s`: Thời gian xử lý end-to-end.

    **Lưu ý:** Endpoint này blocking do LLM inference.
    Timeout khuyến nghị: 60 giây.
    """
    _require_index(app.state)
    _require_llm(app.state)

    query = request.query.strip()
    logger.info(f"POST /ask | query='{query[:60]}'")

    try:
        # Chạy trong threadpool để không block event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,  # default threadpool
            _run_rag_sync,
            query,
            request.top_n,
            app.state.retriever,
            app.state.reranker,
            app.state.llm,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Lỗi RAG query: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

    # Lưu vào history
    _query_history.append({
        "query":      query,
        "has_answer": response.has_answer,
        "latency_s":  response.latency_s,
        "timestamp":  time.time(),
    })
    if len(_query_history) > MAX_HISTORY:
        _query_history.pop(0)

    return AskResponse(
        query        = response.query,
        answer       = response.answer,
        has_answer   = response.has_answer,
        context_used = response.context_used,
        latency_s    = response.latency_s,
        sources      = [
            SourceItem(
                rank         = s.get("rank", i + 1),
                file_name    = s.get("file_name", ""),
                chunk_index  = s.get("chunk_index", -1),
                score        = round(s.get("score", 0.0), 4),
                rerank_score = round(s.get("rerank_score", 0.0), 4),
            )
            for i, s in enumerate(response.sources)
        ],
    )


def _run_rag_sync(query, top_n, retriever, reranker, llm):
    """Wrapper đồng bộ cho rag_query — chạy trong threadpool."""
    from src.pipeline_m3 import run_query as retrieve
    from src.pipeline_m4 import rag_query
    return rag_query(query, retriever, reranker, llm, top_n=top_n)


@app.post("/upload", response_model=UploadResponse, tags=["Library"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload tài liệu mới vào thư viện số.

    Hỗ trợ: PDF, DOCX, TXT.
    Sau khi upload, gọi `POST /index/rebuild` để cập nhật FAISS index.

    **Giới hạn:** 50MB mỗi file.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Định dạng '{ext}' không được hỗ trợ. Chấp nhận: {SUPPORTED_EXTENSIONS}",
        )

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_RAW_DIR / file.filename

    content = await file.read()

    # Giới hạn 50MB
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File vượt quá 50MB")

    dest.write_bytes(content)
    logger.info(f"Upload: {file.filename} ({len(content):,} bytes)")

    return UploadResponse(
        filename   = file.filename,
        size_bytes = len(content),
        message    = f"Đã lưu '{file.filename}'. Gọi POST /index/rebuild để cập nhật index.",
    )


@app.post("/index/rebuild", response_model=IndexRebuildResponse, tags=["Library"])
async def rebuild_index(background_tasks: BackgroundTasks):
    """
    Rebuild FAISS index từ toàn bộ tài liệu trong data/raw/.

    Quá trình: Ingestion → Chunking → Embedding → FAISS index.
    Thời gian: vài phút tùy số lượng tài liệu và hardware.

    **Lưu ý:** Endpoint này chạy đồng bộ và có thể mất vài phút.
    Trong production nên dùng background task queue (Celery/RQ).
    """
    logger.info("POST /index/rebuild — bắt đầu rebuild...")
    t0 = time.time()

    try:
        loop = asyncio.get_event_loop()
        chunks_count = await loop.run_in_executor(None, _rebuild_index_sync)
        elapsed = time.time() - t0

        # Reload retriever với index mới
        if FAISS_INDEX_FILE.exists():
            try:
                from src.pipeline_m3 import build_retrieval_pipeline
                retriever, reranker = build_retrieval_pipeline()
                app.state.retriever    = retriever
                app.state.reranker     = reranker
                app.state.index_loaded = True
                logger.info("Retrieval pipeline đã reload với index mới")
            except Exception as e:
                logger.error(f"Reload pipeline thất bại: {e}")

        return IndexRebuildResponse(
            status       = "success",
            chunks_count = chunks_count,
            elapsed_s    = round(elapsed, 2),
            message      = f"Rebuild hoàn tất: {chunks_count} chunks trong {elapsed:.1f}s",
        )

    except Exception as e:
        logger.error(f"Rebuild thất bại: {e}")
        raise HTTPException(status_code=500, detail=f"Rebuild thất bại: {str(e)}")


def _rebuild_index_sync() -> int:
    """Chạy M1 + M2 pipeline và trả về số chunks."""
    from src.pipeline_m1 import run_pipeline as run_m1
    from src.pipeline_m2 import run_pipeline_m2

    chunks = run_m1()
    run_pipeline_m2(rebuild=True)
    return len(chunks)


@app.get("/history", tags=["RAG"])
async def get_history(limit: int = 20):
    """
    Lịch sử các truy vấn trong session hiện tại.
    Tối đa 50 queries gần nhất.
    """
    recent = _query_history[-limit:][::-1]  # Mới nhất trước
    return {
        "total":   len(_query_history),
        "showing": len(recent),
        "history": recent,
    }


@app.post("/eval/run", tags=["Evaluation"])
async def run_evaluation(n_samples: int = 10):
    """
    Chạy evaluation trên tập test và trả về báo cáo metrics.

    Args:
        n_samples: Số câu hỏi chạy (default 10, max 30).

    Trả về:
        EvalReport với Faithfulness, Context Recall, Recall@k, v.v.
    """
    _require_index(app.state)

    n_samples = min(n_samples, 30)
    logger.info(f"POST /eval/run — {n_samples} samples")

    try:
        loop = asyncio.get_event_loop()
        report_dict = await loop.run_in_executor(
            None, _run_eval_sync, n_samples, app.state
        )
        return report_dict
    except Exception as e:
        logger.error(f"Evaluation lỗi: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _run_eval_sync(n_samples: int, state) -> dict:
    """Chạy evaluation đồng bộ trong threadpool."""
    from src.evaluation.evaluator import RAGEvaluator
    from src.pipeline_m4 import rag_query
    import json
    from config import EVAL_DATASET_PATH

    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"eval_dataset.json không tồn tại: {EVAL_DATASET_PATH}")

    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        all_samples = json.load(f)

    sample_ids = [s["id"] for s in all_samples[:n_samples]]

    def rag_fn(query: str):
        return rag_query(query, state.retriever, state.reranker, state.llm)

    evaluator = RAGEvaluator(rag_fn=rag_fn)
    report    = evaluator.run(sample_ids=sample_ids, save=True)
    return report.to_dict()


# ─────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "name":    "RAG Library API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "endpoints": {
            "ask":           "POST /ask",
            "upload":        "POST /upload",
            "rebuild_index": "POST /index/rebuild",
            "evaluation":    "POST /eval/run",
            "history":       "GET  /history",
            "stats":         "GET  /stats",
        },
    }
