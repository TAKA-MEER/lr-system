import asyncio
import logging
import os
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.websocket import any_recording, ensure_stt_loaded, get_or_create_session, get_session, reset_session
from llm.client import OllamaClient
from llm.pipeline import build_minutes
from postprocess.formatter import generate_docx

logger = logging.getLogger(__name__)
router = APIRouter()
_generate_lock = asyncio.Lock()  # 議事録生成は同時に1つだけ(GPUを使うため)


# ------------------------------------------------------------------
# リクエスト/レスポンスモデル
# ------------------------------------------------------------------

class MetaRequest(BaseModel):
    session_id: str
    trial_name: str = ""
    location: str = ""
    client_attendees: list[str] = []
    our_attendees: list[str] = []


class GenerateRequest(BaseModel):
    session_id: str


class ThresholdRequest(BaseModel):
    session_id: str
    threshold: float


# ------------------------------------------------------------------
# エンドポイント
# ------------------------------------------------------------------

@router.post("/session/reset")
async def session_reset(session_id: str, request: Request):
    """新しい試験を始める前に文字起こしを消去する。
    録音中(内蔵マイクがチャンクを送ってきている間)は消去しない。BT画面の開始や操作順の違いで
    試験中の文字起こしが消えるのを防ぐため。試験情報(meta)は消さない。
    VRAM管理: 前回生成で保持されたままのOllamaモデルを解放し、STTモデルを再ロードする。"""
    session = get_session(session_id)
    if session is not None and session.is_recording:
        raise HTTPException(409, "録音中のため文字起こしを消去できません")
    if _generate_lock.locked():
        raise HTTPException(409, "議事録を生成中です。完了してから録音を開始してください")
    reset_session(session_id)
    await ensure_stt_loaded(request.app)
    return {"status": "ok", "session_id": session_id}


@router.post("/session/meta")
async def set_meta(req: MetaRequest):
    """試験名・参加者情報を登録する"""
    session = get_or_create_session(req.session_id)
    session.set_meta({
        "trial_name": req.trial_name,
        "location": req.location,
        "date": time.strftime("%Y年%m月%d日"),
        "attendees": {
            "client": req.client_attendees,
            "our_side": req.our_attendees,
        },
    })
    return {"status": "ok"}


@router.get("/transcript/{session_id}")
async def get_transcript(session_id: str):
    """現在の文字起こし内容を返す"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "セッションが存在しません")
    return {
        "segments": [
            {"speaker": s.speaker, "text": s.text, "timestamp": s.timestamp, "chunk_index": s.chunk_index}
            for s in session.transcript
        ]
    }


@router.post("/generate")
async def generate_minutes(req: GenerateRequest, request: Request):
    """
    議事録を生成してdocxファイルパスを返す。
    STTモデルを解放してからLLMをロードする（VRAM管理）。
    """
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(404, "セッションが存在しません")
    if not session.transcript:
        raise HTTPException(400, "文字起こしデータがありません")
    # 録音停止の直後は、最後のチャンクの文字起こし・話者判定が終わるまで数秒かかるので待つ
    for _ in range(24):
        if not any_recording():
            break
        await asyncio.sleep(0.5)
    else:
        # 生成のためにSTTモデルをVRAMから外すと、録音中の文字起こしが止まってしまう
        raise HTTPException(409, "録音中です。両方の画面で録音を停止してから議事録を生成してください")
    if _generate_lock.locked():
        raise HTTPException(409, "議事録を生成中です")

    async with _generate_lock:
        request.app.state.generating = True
        try:
            return await _generate(session, request)
        finally:
            request.app.state.generating = False


async def _generate(session, request: Request):
    # ---- VRAM管理: STT解放 ----
    # Ollamaは初回チャットリクエスト時に自動でモデルをロードするため、
    # 明示的なロード処理は不要（次のollama.chat()呼び出しがロードを兼ねる）。
    request.app.state.transcriber.unload()

    # ---- テキスト生成 ----
    full_text = "\n".join(
        f"[{s.speaker}] {s.text}" for s in session.transcript
    )

    cfg_path = Path("config/settings.yaml")
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ollama = OllamaClient(base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    try:
        minutes_json, info = await build_minutes(full_text, session.meta, cfg["llm"], ollama)
    except Exception as e:
        logger.error(f"議事録生成に失敗: {type(e).__name__}: {e}")
        raise HTTPException(502, f"議事録の生成に失敗しました(文字起こしは保存されています。再度お試しください): {type(e).__name__}")
    logger.info(f"議事録生成の処理情報: {info}")

    # ---- docx生成 ----
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"議事録_{time.strftime('%Y%m%d_%H%M%S')}.docx"
    output_path = output_dir / filename

    generate_docx(minutes_json, str(output_path))
    logger.info(f"議事録を生成しました: {output_path}")

    return {"status": "ok", "filename": filename, "parse_ok": info.get("stage2_parse_ok", True)}


@router.get("/download/{filename}")
async def download_minutes(filename: str):
    """生成済みdocxをダウンロードする"""
    import yaml
    with open("config/settings.yaml") as f:
        cfg = yaml.safe_load(f)
    # 出力フォルダ直下の .docx 以外は返さない(../ 等によるフォルダ外のファイル取得を防ぐ)
    if not re.fullmatch(r"[^/\\]+\.docx", filename) or filename.startswith("."):
        raise HTTPException(400, "不正なファイル名です")
    file_path = Path(cfg["output"]["dir"]) / filename
    if not file_path.exists():
        raise HTTPException(404, "ファイルが見つかりません")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.post("/speaker/threshold")
async def update_threshold(req: ThresholdRequest):
    """話者判定閾値をリアルタイムで変更する"""
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(404, "セッションが存在しません")
    session.speaker_detector.update_threshold(req.threshold)
    return {"status": "ok", "threshold": req.threshold}
