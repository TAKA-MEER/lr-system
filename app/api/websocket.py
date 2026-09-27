import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from stt.speaker import SpeakerDetector
from stt.transcriber import SAMPLE_RATE, StreamDecoder, rms

logger = logging.getLogger(__name__)
router = APIRouter()

# 文字起こし・試験情報の保存先。コンテナが落ちても、再起動後に同じセッションIDで続きから使える
SESSION_DIR = Path("/output/sessions")
RECORDING_IDLE_SECONDS = 15.0  # 最後のチャンク受信からこの秒数を過ぎたら「録音中ではない」とみなす
BT_WAIT_SECONDS = 6.0          # 内蔵マイクの区間に対応するBT音声の到着を待つ上限


# ------------------------------------------------------------------
# セッション管理
# ------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    timestamp: float
    speaker: str   # "our_side" | "client"
    text: str
    chunk_index: int = 0  # 内蔵マイクWebSocket接続内での受信チャンク番号(1始まり)。遅延計測・試験の採点用


@dataclass
class AudioSession:
    session_id: str
    transcript: List[TranscriptSegment] = field(default_factory=list)
    speaker_detector: SpeakerDetector = field(default_factory=SpeakerDetector)
    internal_ws: Optional[WebSocket] = None
    bt_ws: Optional[WebSocket] = None
    meta: dict = field(default_factory=dict)  # 日時・参加者など
    last_internal_chunk_at: float = 0.0

    @property
    def is_recording(self) -> bool:
        return self.internal_ws is not None and time.time() - self.last_internal_chunk_at < RECORDING_IDLE_SECONDS

    # ---- 永続化(1セッション1ファイルの追記ログ) ----
    def _log(self, record: dict):
        try:
            SESSION_DIR.mkdir(parents=True, exist_ok=True)
            with open(SESSION_DIR / f"{self.session_id}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(f"セッションの保存に失敗: {e}")

    def add_segment(self, seg: TranscriptSegment):
        self.transcript.append(seg)
        self._log({"type": "segment", **asdict(seg)})

    def set_meta(self, meta: dict):
        self.meta = meta
        self._log({"type": "meta", "meta": meta})

    def clear(self):
        """新しい試験を始める。接続中のWebSocketはそのまま使い続けられるよう、オブジェクトは置き換えない"""
        self.transcript.clear()
        self.speaker_detector.clear()
        self._log({"type": "reset"})


def _new_detector(cfg: dict | None) -> SpeakerDetector:
    sp = (cfg or {}).get("speaker", {})
    return SpeakerDetector(
        threshold=sp.get("bt_rms_threshold", 800.0),
        method=sp.get("method", "adaptive"),
        history_seconds=sp.get("history_seconds", 60.0),
        margin_db=sp.get("margin_db", 16.0),
        segment_active_ratio=sp.get("segment_active_ratio", 0.3),
    )


def _load_from_disk(session_id: str, cfg: dict | None) -> Optional[AudioSession]:
    path = SESSION_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return None
    s = AudioSession(session_id=session_id, speaker_detector=_new_detector(cfg))
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r["type"] == "segment":
            r.pop("type")
            s.transcript.append(TranscriptSegment(**r))
        elif r["type"] == "meta":
            s.meta = r["meta"]
        elif r["type"] == "reset":
            s.transcript.clear()
    logger.info(f"セッションをファイルから復元: {session_id} segments={len(s.transcript)}")
    return s


# インメモリセッションストア(ファイルが正本、メモリはキャッシュ)
_sessions: Dict[str, AudioSession] = {}
_config: dict | None = None


def configure(cfg: dict):
    global _config
    _config = cfg


def get_or_create_session(session_id: str) -> AudioSession:
    s = get_session(session_id)
    if s is None:
        s = AudioSession(session_id=session_id, speaker_detector=_new_detector(_config))
        _sessions[session_id] = s
        logger.info(f"新規セッション: {session_id}")
    return s


def get_session(session_id: str) -> Optional[AudioSession]:
    if session_id not in _sessions:
        s = _load_from_disk(session_id, _config)
        if s is not None:
            _sessions[session_id] = s
    return _sessions.get(session_id)


def reset_session(session_id: str) -> AudioSession:
    s = get_or_create_session(session_id)
    s.clear()
    logger.info(f"セッションをリセット: {session_id}")
    return s


def any_recording() -> bool:
    return any(s.is_recording for s in _sessions.values())


async def ensure_stt_loaded(app):
    """STTモデルが外れていれば、先にLLMをVRAMから外してから読み込む(両方が同時に載らないように)"""
    transcriber = app.state.transcriber
    if transcriber.is_loaded:
        return
    import os
    from llm.client import OllamaClient
    ollama = OllamaClient(base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    try:
        await ollama.unload_model(app.state.config["llm"]["model"])
    except Exception as e:
        logger.warning(f"Ollamaモデルのアンロードに失敗(続行): {e}")
    await asyncio.get_running_loop().run_in_executor(None, transcriber.load)


class StreamClock:
    """1本のWebSocket接続の音声としての時刻。最初のチャンクの受信時刻からその長さを引いた時刻を
    録音開始時刻とし、以降はチャンクの長さを積み上げる"""

    def __init__(self):
        self.origin: float | None = None
        self.elapsed = 0.0

    def next(self, n_samples: int) -> tuple[float, float]:
        dur = n_samples / SAMPLE_RATE
        if self.origin is None:
            self.origin = time.time() - dur
        start = self.origin + self.elapsed
        self.elapsed += dur
        return start, start + dur


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
    decoder = StreamDecoder("BT")
    clock = StreamClock()
    loop = asyncio.get_running_loop()
    logger.info(f"[BT] 接続: {session_id}")

    try:
        async for data in websocket.iter_bytes():
            if not data:
                continue
            pcm = await loop.run_in_executor(None, decoder.decode, data)
            if pcm is None:
                continue
            start, _ = clock.next(len(pcm))
            level = rms(pcm)
            # セッションがリセットで作り直されていても、常に現在のセッションに登録する
            det = get_or_create_session(session_id).speaker_detector
            det.add_bt_audio(pcm, start)
            det.add_bt_rms(level, start + len(pcm) / SAMPLE_RATE)
            logger.info(f"[BT] RMS={level:.0f}")
            await websocket.send_json({"type": "rms", "value": level})
        logger.info(f"[BT] 切断: {session_id}")
    except WebSocketDisconnect as e:
        logger.info(f"[BT] 切断: {session_id} code={e.code}")
    except Exception as e:
        logger.error(f"[BT] 例外: {e}")
    finally:
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
    app = websocket.app
    transcriber = app.state.transcriber
    cfg = app.state.config

    vad_filter = cfg["stt"].get("vad_enabled", True)
    vad_params = {
        "min_silence_duration_ms": cfg["stt"]["vad_min_silence_ms"],
        "speech_pad_ms": cfg["stt"]["vad_speech_pad_ms"],
        "threshold": cfg["stt"]["vad_threshold"],
    }
    decoder = StreamDecoder("Internal")
    clock = StreamClock()
    loop = asyncio.get_running_loop()
    # 話者判定と送信は、BT音声の到着を待つことがあるため受信ループとは別のタスクで順番に行う
    queue: asyncio.Queue = asyncio.Queue()
    finalizer = asyncio.create_task(_finalize_loop(websocket, session_id, queue))

    logger.info(f"[Internal] 接続: {session_id} vad_filter={vad_filter}")
    chunk_count = 0

    try:
        async for data in websocket.iter_bytes():
            if not data:
                continue
            chunk_count += 1
            session.last_internal_chunk_at = time.time()
            logger.info(f"[Internal] チャンク#{chunk_count}受信 size={len(data)}")

            pcm = await loop.run_in_executor(None, decoder.decode, data)
            if pcm is None:
                continue
            chunk_start, chunk_end = clock.next(len(pcm))

            if not transcriber.is_loaded:
                if getattr(app.state, "generating", False):
                    logger.warning(f"[Internal] 議事録生成中のため文字起こしできません(チャンク#{chunk_count})")
                    await websocket.send_json({"type": "error", "message": "議事録生成中のため文字起こしを停止しています"})
                    continue
                await ensure_stt_loaded(app)

            prompt = (cfg["stt"].get("initial_prompt") or "").format(
                trial_name=get_or_create_session(session_id).meta.get("trial_name", ""))
            try:
                segments = await loop.run_in_executor(
                    None, transcriber.transcribe_pcm, pcm, vad_filter, vad_params, prompt)
            except Exception as e:  # 1チャンクの失敗で録音全体を止めない
                logger.error(f"[Internal] チャンク#{chunk_count} の文字起こしに失敗: {e}")
                continue
            if not segments:
                logger.info(f"[Internal] チャンク#{chunk_count}: 認識結果なし（空文字）")
                continue
            await queue.put((chunk_count, chunk_start, chunk_end, segments))
        # iter_bytes() はクライアントの切断時に例外を出さずに終わる(以前は例外処理の中でしか録音中の
        # 状態を解除しておらず、「切断を検知できない」原因になっていた)
        logger.info(f"[Internal] 切断: {session_id}")

    except WebSocketDisconnect as e:
        logger.info(f"[Internal] 切断: {session_id} code={e.code}")
    except Exception as e:
        logger.error(f"[Internal] 例外: {e}")
    finally:
        await queue.put(None)
        try:
            await asyncio.wait_for(finalizer, timeout=BT_WAIT_SECONDS + 5)
        except (asyncio.TimeoutError, Exception):
            finalizer.cancel()
        session.internal_ws = None


async def _finalize_loop(websocket: WebSocket, session_id: str, queue: asyncio.Queue):
    """文字起こし結果に話者を付けて保存・送信する(受信順)"""
    while True:
        item = await queue.get()
        if item is None:
            return
        chunk_index, chunk_start, chunk_end, segments = item
        session = get_or_create_session(session_id)
        det = session.speaker_detector
        # BTが接続されていれば、この区間を覆うBT音声が届くまで少し待つ
        deadline = time.time() + BT_WAIT_SECONDS
        while session.bt_ws is not None and det.bt_covered_until < chunk_end and time.time() < deadline:
            await asyncio.sleep(0.2)

        # セグメントごとに話者を判定し、同じ話者が続くセグメントは1行にまとめる
        groups: list[list] = []
        for s in segments:
            a = chunk_start + s.start
            b = min(chunk_start + max(s.end, s.start + 0.1), chunk_end)
            sp = det.label(a, b)
            if groups and groups[-1][0] == sp:
                groups[-1][1].append(s.text)
            else:
                groups.append([sp, [s.text]])
        for sp, texts in groups:
            text = " ".join(texts).strip()
            seg = TranscriptSegment(timestamp=chunk_end, speaker=sp, text=text, chunk_index=chunk_index)
            session.add_segment(seg)
            logger.info(f"[{sp}] {text[:60]}")
            try:
                await websocket.send_json({"type": "transcript", "speaker": sp, "text": text,
                                           "timestamp": chunk_end, "chunk_index": chunk_index})
            except Exception:
                pass  # ブラウザが閉じられていても文字起こしは保存済み
