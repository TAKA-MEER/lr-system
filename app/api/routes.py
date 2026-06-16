import logging
import os
import time
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.websocket import get_session, reset_session
from llm.client import OllamaClient
from llm.prompts import build_stage1_prompt, build_stage2_prompt
from postprocess.chunker import split_transcript
from postprocess.formatter import generate_docx

logger = logging.getLogger(__name__)
router = APIRouter()


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
async def session_reset(session_id: str):
    """録音開始前にセッションをリセットする"""
    reset_session(session_id)
    return {"status": "ok", "session_id": session_id}


@router.post("/session/meta")
async def set_meta(req: MetaRequest):
    """試験名・参加者情報を登録する"""
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(404, "セッションが存在しません")
    session.meta = {
        "trial_name": req.trial_name,
        "location": req.location,
        "date": time.strftime("%Y年%m月%d日"),
        "attendees": {
            "client": req.client_attendees,
            "our_side": req.our_attendees,
        },
    }
    return {"status": "ok"}


@router.get("/transcript/{session_id}")
async def get_transcript(session_id: str):
    """現在の文字起こし内容を返す"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "セッションが存在しません")
    return {
        "segments": [
            {"speaker": s.speaker, "text": s.text, "timestamp": s.timestamp}
            for s in session.transcript
        ]
    }


@router.post("/generate")
async def generate_minutes(req: GenerateRequest, app=None):
    """
    議事録を生成してdocxファイルパスを返す。
    STTモデルを解放してからLLMをロードする（VRAM管理）。
    """
    from fastapi import Request
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(404, "セッションが存在しません")
    if not session.transcript:
        raise HTTPException(400, "文字起こしデータがありません")

    # ---- VRAM管理: STT解放 → LLMロード ----
    # transcriber は app.state 経由で取得する必要があるが、
    # ここでは設計上 app.state にアクセスしにくいため
    # generate エンドポイントは Depends(get_app) パターンを使う
    # → Phase2では手動でtorch.cuda.empty_cache() を呼ぶ
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- テキスト生成 ----
    full_text = "\n".join(
        f"[{s.speaker}] {s.text}" for s in session.transcript
    )

    cfg_path = Path("config/settings.yaml")
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ollama = OllamaClient(base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    model = cfg["llm"]["model"]
    chunk_chars = cfg["llm"]["chunk_chars"]

    # Stage 1: チャンクごとに要点JSON抽出
    chunks = split_transcript(full_text, max_chars=chunk_chars)
    logger.info(f"Stage1: {len(chunks)} チャンクを処理中...")
    summaries = []
    for i, chunk in enumerate(chunks):
        prompt = build_stage1_prompt(chunk)
        result = await ollama.chat(model=model, prompt=prompt)
        summaries.append(result)
        logger.info(f"  Stage1 [{i+1}/{len(chunks)}] 完了")

    # Stage 2: 要点JSONを統合して最終議事録JSON生成
    logger.info("Stage2: 最終議事録を生成中...")
    stage2_prompt = build_stage2_prompt(summaries, session.meta)
    minutes_json_str = await ollama.chat(model=model, prompt=stage2_prompt)

    import json
    try:
        minutes_json = json.loads(minutes_json_str)
    except json.JSONDecodeError:
        # JSON解析失敗時はフォールバック
        logger.warning("JSON解析失敗。フォールバック構造を使用")
        minutes_json = {
            "trial_name": session.meta.get("trial_name", ""),
            "date": session.meta.get("date", ""),
            "location": session.meta.get("location", ""),
            "attendees": session.meta.get("attendees", {"client": [], "our_side": []}),
            "discussions": [],
            "action_items": [],
            "raw": minutes_json_str,
        }

    # ---- docx生成 ----
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"議事録_{time.strftime('%Y%m%d_%H%M%S')}.docx"
    output_path = output_dir / filename

    generate_docx(minutes_json, str(output_path))
    logger.info(f"議事録を生成しました: {output_path}")

    return {"status": "ok", "filename": filename}


@router.get("/download/{filename}")
async def download_minutes(filename: str):
    """生成済みdocxをダウンロードする"""
    import yaml
    with open("config/settings.yaml") as f:
        cfg = yaml.safe_load(f)
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
