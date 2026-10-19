"""Controller: FastAPI 엔드포인트"""
from dotenv import load_dotenv

load_dotenv()

import base64  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile  # noqa: E402
from PIL import Image  # noqa: E402

from .schemas import (AskRequest, AskResponse, CommentEvent, CommentRecord,  # noqa: E402
                      CommentStatus, MLRequest, WebhookAck)
from .services import llm, ml, rag, vision  # noqa: E402
from .storage import CommentStorage  # noqa: E402

app = FastAPI(title="AI·데이터 분석 통합 서비스 API")
db = CommentStorage()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
KST = timezone(timedelta(hours=9))


def now() -> datetime:
    return datetime.now(KST)


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / Path(name).name  # 경로 조작 방지
    if not path.exists():
        raise HTTPException(404, f"파일을 찾을 수 없습니다: {name}")
    for enc in ("utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise HTTPException(400, "CSV 인코딩을 읽을 수 없습니다.")


def read_image(file: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file.file.read()))
    except Exception:
        raise HTTPException(400, "이미지 파일을 읽을 수 없습니다.")


@app.get("/health")
def health():
    return {"status": "ok", "gemini_key_set": bool(os.getenv("GEMINI_API_KEY"))}


# ---------- 1) 웹훅 댓글 자동 답변 (Gemini) ----------
def process_comment(comment_id: str) -> None:
    rec = db.get(comment_id)
    if rec is None:
        return
    try:
        reply = llm.generate_reply(rec.author, rec.content)
        db.update(comment_id, status=CommentStatus.completed, ai_reply=reply, replied_at=now())
    except Exception as e:  # noqa: BLE001
        db.update(comment_id, status=CommentStatus.failed, error=str(e))


@app.post("/webhook/comment", status_code=202, response_model=WebhookAck)
def webhook_comment(event: CommentEvent, background: BackgroundTasks):
    record = CommentRecord(**event.model_dump(), received_at=now())
    if not db.add(record):
        raise HTTPException(409, f"이미 접수된 comment_id 입니다: {event.comment_id}")
    background.add_task(process_comment, event.comment_id)
    return WebhookAck(accepted=True, comment_id=event.comment_id, status=CommentStatus.pending,
                      message="댓글 이벤트가 접수되었습니다. (답변 생성 대기 중)")


@app.get("/comments", response_model=list[CommentRecord])
def list_comments():
    return db.list()


@app.get("/comments/{comment_id}", response_model=CommentRecord)
def get_comment(comment_id: str):
    rec = db.get(comment_id)
    if rec is None:
        raise HTTPException(404, "존재하지 않는 comment_id 입니다.")
    return rec


# ---------- 2) 데이터 분석 · 머신러닝 ----------
@app.get("/data/files")
def data_files():
    return {"files": sorted(p.name for p in DATA_DIR.glob("*.csv"))}


@app.post("/data/upload")
async def data_upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if not name.lower().endswith(".csv"):
        raise HTTPException(400, "CSV 파일만 업로드할 수 있습니다.")
    (DATA_DIR / name).write_bytes(await file.read())
    return {"saved": name}


@app.get("/data/eda")
def data_eda(file: str):
    return {"file": file, **ml.eda(load_csv(file))}


@app.post("/ml/run")
def ml_run(req: MLRequest):
    train = load_csv(req.train_file)
    test = load_csv(req.test_file) if req.test_file else None
    try:
        return ml.run_classification(train, req.target, test)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


# ---------- 3) CNN 분류 / 4) YOLO 탐지 ----------
@app.post("/vision/classify")
def vision_classify(file: UploadFile = File(...)):
    return vision.classify(read_image(file))


@app.post("/vision/detect")
def vision_detect(file: UploadFile = File(...), conf: float = 0.25):
    dets, annotated = vision.detect(read_image(file), conf)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    return {"detections": dets, "image_base64": base64.b64encode(buf.getvalue()).decode()}


# ---------- 5) RAG 문서 Q&A ----------
@app.post("/rag/upload")
def rag_upload(file: UploadFile = File(...)):
    try:
        n = rag.add_document(file.filename or "document", file.file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"filename": file.filename, "chunks_added": n, **rag.status()}


@app.get("/rag/status")
def rag_status():
    return rag.status()


@app.post("/rag/reset")
def rag_reset():
    rag.reset()
    return rag.status()


@app.post("/rag/ask", response_model=AskResponse)
def rag_ask(req: AskRequest):
    try:
        answer, sources = rag.ask(req.question, req.top_k)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return AskResponse(answer=answer, sources=sources)
