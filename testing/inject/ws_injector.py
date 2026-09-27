"""
internal_mix.wav / bt_mix.wav を単一WebM/Opusにエンコードし、Cluster境界で分割したチャンク列を
/ws/audio/internal と /ws/audio/bt に実時間ペースで同時送信する。
"""
import asyncio
import json
import logging
import random
import time
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
                        start_delay: float, label: str, on_message=None, timing: dict | None = None):
    """timing を渡すと、チャンクごとの送信時刻(sent[i], iは1始まり)と、文字起こし結果の受信時刻
    (received[chunk_index])をホストの時計で記録する(遅延の計測用)。"""
    await asyncio.sleep(start_delay)
    logger.info(f"[{label}] 接続開始: {ws_url}")
    async with websockets.connect(ws_url, max_size=None) as ws:

        async def receiver():
            try:
                async for msg in ws:
                    if timing is not None and isinstance(msg, str):
                        try:
                            d = json.loads(msg)
                            if d.get("type") == "transcript" and "chunk_index" in d:
                                timing["received"][d["chunk_index"]] = time.time()
                        except ValueError:
                            pass
                    if on_message:
                        on_message(label, msg)
            except websockets.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())

        if timing is not None:
            timing.setdefault("sent", {})
            timing.setdefault("received", {})
        for i, chunk in enumerate(chunks):
            await ws.send(chunk)
            if timing is not None:
                timing["sent"][i + 1] = time.time()
            if i < len(chunks) - 1:
                await asyncio.sleep(max(0.0, interval + random.uniform(-jitter, jitter)))

        # 最後のチャンクのサーバー処理が終わるまで待ってから閉じる(処理が追いつかない条件では
        # 未処理のチャンクが残るため、受信が途絶えるまで最大 drain_timeout 秒待つ)
        await asyncio.sleep(2.0)
        if timing is not None:
            last = time.time()
            n_recv = -1
            while time.time() - last < 15.0 and time.time() - timing["sent"][len(chunks)] < timing.get("drain_timeout", 600):
                if len(timing["received"]) != n_recv:
                    n_recv = len(timing["received"])
                    last = time.time()
                await asyncio.sleep(1.0)
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
    timing: dict | None = None,
):
    """internal/bt 両チャンネルへ同時にチャンクをペース送信する。"""
    await asyncio.gather(
        send_channel(bt_ws_url, bt_chunks, interval, jitter, start_delay=0.0, label="BT", on_message=_log_message),
        send_channel(internal_ws_url, internal_chunks, interval, jitter, start_delay=bt_start_offset,
                     label="Internal", on_message=_log_message, timing=timing),
    )
