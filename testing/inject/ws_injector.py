"""
internal_mix.wav / bt_mix.wav を単一WebM/Opusにエンコードし、Cluster境界で分割したチャンク列を
/ws/audio/internal と /ws/audio/bt に実時間ペースで同時送信する。
"""
import asyncio
import json
import logging
import random
from pathlib import Path

import websockets

from ebml_cluster_finder import split_into_mediarecorder_like_chunks
from webm_muxer import mux_to_single_webm

logger = logging.getLogger(__name__)


def prepare_chunks(wav_path: Path, work_dir: Path, name: str, sample_rate: int = 48000, bitrate: str = "32k",
                    cluster_time_limit_ms: int = 5000) -> list[bytes]:
    """WAVファイルをWebM/Opusにエンコードし、Cluster境界で分割したチャンク列を返す。"""
    webm_path = work_dir / f"{name}.webm"
    mux_to_single_webm(
        str(wav_path), str(webm_path),
        sample_rate=sample_rate, bitrate=bitrate, cluster_time_limit_ms=cluster_time_limit_ms,
    )
    data = webm_path.read_bytes()
    chunks = split_into_mediarecorder_like_chunks(data)
    logger.info(f"{name}: {len(chunks)} チャンクに分割 (webm size={len(data)} bytes)")
    return chunks


async def send_channel(ws_url: str, chunks: list[bytes], interval: float, jitter: float,
                        start_delay: float, label: str, on_message=None):
    await asyncio.sleep(start_delay)
    logger.info(f"[{label}] 接続開始: {ws_url}")
    async with websockets.connect(ws_url, max_size=None) as ws:

        async def receiver():
            try:
                async for msg in ws:
                    if on_message:
                        on_message(label, msg)
            except websockets.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())

        for i, chunk in enumerate(chunks):
            await ws.send(chunk)
            if i < len(chunks) - 1:
                await asyncio.sleep(max(0.0, interval + random.uniform(-jitter, jitter)))

        # 最後のチャンクのサーバー処理が終わるまで少し待ってから閉じる
        await asyncio.sleep(2.0)
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    logger.info(f"[{label}] 送信完了・切断: 総チャンク数={len(chunks)}")


def _log_message(label: str, msg):
    if isinstance(msg, (bytes, bytearray)):
        return
    try:
        data = json.loads(msg)
    except (ValueError, TypeError):
        return
    if data.get("type") == "transcript":
        logger.info(f"[{label}] transcript speaker={data.get('speaker')} text={data.get('text', '')[:60]}")
    elif data.get("type") == "rms":
        pass  # 頻度が高いのでログしない


async def run_injection_ws(
    internal_ws_url: str,
    bt_ws_url: str,
    internal_chunks: list[bytes],
    bt_chunks: list[bytes],
    interval: float,
    jitter: float,
    bt_start_offset: float,
):
    """internal/bt 両チャンネルへ同時にチャンクをペース送信する。"""
    await asyncio.gather(
        send_channel(bt_ws_url, bt_chunks, interval, jitter, start_delay=0.0, label="BT", on_message=_log_message),
        send_channel(internal_ws_url, internal_chunks, interval, jitter, start_delay=bt_start_offset,
                     label="Internal", on_message=_log_message),
    )
