import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
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
async def session_reset(session_id: str, request: Request):
    """録音開始前にセッションをリセットする。
    VRAM管理: 前回生成で保持されたままのOllamaモデルを解放し、STTモデルを再ロードする
    （両方が同時にVRAMへ乗る瞬間を作らないよう、この順序で行う）。"""
    reset_session(session_id)

    cfg = request.app.state.config
    ollama = OllamaClient(base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    await ollama.unload_model(cfg["llm"]["model"])
    request.app.state.transcriber.load()

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
    model = cfg["llm"]["model"]
    chunk_chars = cfg["llm"]["chunk_chars"]
    temperature = cfg["llm"].get("temperature", 0.2)
    num_ctx = cfg["llm"].get("num_ctx", 32768)

    # Stage 1: チャンクごとに要点JSON抽出
    chunks = split_transcript(full_text, max_chars=chunk_chars)
    logger.info(f"Stage1: {len(chunks)} チャンクを処理中...")
    summaries = []
    for i, chunk in enumerate(chunks):
        prompt = build_stage1_prompt(chunk)
        result = await ollama.chat(model=model, prompt=prompt, temperature=temperature, num_ctx=num_ctx)
        summaries.append(result)
        logger.info(f"  Stage1 [{i+1}/{len(chunks)}] 完了")

    # Stage 2: 要点JSONを統合して最終議事録JSON生成
    logger.info("Stage2: 最終議事録を生成中...")
    stage2_prompt = build_stage2_prompt(summaries, session.meta)
    minutes_json_str = await ollama.chat(model=model, prompt=stage2_prompt, temperature=temperature, num_ctx=num_ctx)

    import json
    import re

    def _extract_json(raw: str) -> str:
        # ```json ... ``` のようなコードフェンスで囲われる場合があるため除去する
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            return m.group(1).strip()
        # 前後に説明文が付与された場合に備え、最初の{から最後の}までを抜き出す
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw[start : end + 1]
        return raw.strip()

    try:
        minutes_json = json.loads(_extract_json(minutes_json_str))
    except json.JSONDecodeError:
        # JSON解析失敗時はフォールバック
        logger.warning(f"JSON解析失敗。フォールバック構造を使用。raw={minutes_json_str[:2000]!r}")
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
