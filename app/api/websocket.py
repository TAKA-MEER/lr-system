import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from stt.speaker import SpeakerDetector

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------
# セッション管理
# ------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    timestamp: float
    speaker: str   # "our_side" | "client"
    text: str


@dataclass
class AudioSession:
    session_id: str
    transcript: List[TranscriptSegment] = field(default_factory=list)
    speaker_detector: SpeakerDetector = field(default_factory=SpeakerDetector)
    internal_ws: Optional[WebSocket] = None
    bt_ws: Optional[WebSocket] = None
    meta: dict = field(default_factory=dict)  # 日時・参加者など


# インメモリセッションストア
_sessions: Dict[str, AudioSession] = {}


def get_or_create_session(session_id: str) -> AudioSession:
    if session_id not in _sessions:
        _sessions[session_id] = AudioSession(session_id=session_id)
        logger.info(f"新規セッション: {session_id}")
    return _sessions[session_id]


def get_session(session_id: str) -> Optional[AudioSession]:
    return _sessions.get(session_id)


def reset_session(session_id: str) -> AudioSession:
    _sessions[session_id] = AudioSession(session_id=session_id)
    logger.info(f"セッションをリセット: {session_id}")
    return _sessions[session_id]


# ------------------------------------------------------------------
# BTマイク WebSocket  (/ws/audio/bt?session_id=xxx)
# ------------------------------------------------------------------

@router.websocket("/ws/audio/bt")
async def ws_bt(
    websocket: WebSocket,
    session_id: str = Query(..., description="セッションID"),
):
    await websocket.accept()
    session = get_or_create_session(session_id)
    session.bt_ws = websocket
    transcriber = websocket.app.state.transcriber
    cfg = websocket.app.state.config
    threshold = cfg["speaker"]["bt_rms_threshold"]
    session.speaker_detector.threshold = threshold

    logger.info(f"[BT] 接続: {session_id}")

    try:
        async for data in websocket.iter_bytes():
            if not data:
                continue
            # RMS計算してSpeakerDetectorに登録（ノンブロッキング）
            loop = asyncio.get_event_loop()
            rms = await loop.run_in_executor(None, transcriber.get_rms, data)
            session.speaker_detector.add_bt_rms(rms)
            logger.debug(f"[BT] RMS={rms:.0f}")

            # ブラウザにRMSを返す（音量バー表示用）
            await websocket.send_json({"type": "rms", "value": rms})

    except WebSocketDisconnect:
        logger.info(f"[BT] 切断: {session_id}")
        session.bt_ws = None
    except Exception as e:
        logger.error(f"[BT] 例外: {e}")
        session.bt_ws = None


# ------------------------------------------------------------------
# 内蔵マイク WebSocket  (/ws/audio/internal?session_id=xxx)
# ------------------------------------------------------------------

@router.websocket("/ws/audio/internal")
async def ws_internal(
    websocket: WebSocket,
    session_id: str = Query(..., description="セッションID"),
):
    await websocket.accept()
    session = get_or_create_session(session_id)
    session.internal_ws = websocket
    transcriber = websocket.app.state.transcriber
    cfg = websocket.app.state.config

    vad_params = {
        "min_silence_duration_ms": cfg["stt"]["vad_min_silence_ms"],
        "speech_pad_ms": cfg["stt"]["vad_speech_pad_ms"],
        "threshold": cfg["stt"]["vad_threshold"],
    }

    logger.info(f"[Internal] 接続: {session_id}")
    chunk_start = time.time()

    try:
        async for data in websocket.iter_bytes():
            if not data:
                continue

            chunk_end = time.time()

            # 話者判定（BTマイクのRMS履歴を参照）
            speaker = session.speaker_detector.get_speaker(chunk_start, chunk_end)

            # 文字起こし（スレッドプールで実行してイベントループをブロックしない）
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, transcriber.transcribe_bytes, data, vad_params
            )
            chunk_start = chunk_end

            if not text:
                continue

            # セッションに保存
            segment = TranscriptSegment(
                timestamp=chunk_end,
                speaker=speaker,
                text=text,
            )
            session.transcript.append(segment)
            logger.info(f"[{speaker}] {text[:60]}")

            # ブラウザに送信
            await websocket.send_json({
                "type": "transcript",
                "speaker": speaker,
                "text": text,
                "timestamp": chunk_end,
            })

    except WebSocketDisconnect:
        logger.info(f"[Internal] 切断: {session_id}")
        session.internal_ws = None
    except Exception as e:
        logger.error(f"[Internal] 例外: {e}")
        session.internal_ws = None
