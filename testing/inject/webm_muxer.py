"""WAVファイルを1回のffmpeg呼び出しで単一の連続WebM/Opusストリームにエンコードする。"""
import subprocess
from pathlib import Path


def mux_to_single_webm(
    wav_path: str,
    out_webm_path: str,
    sample_rate: int = 48000,
    bitrate: str = "32k",
    cluster_time_limit_ms: int = 5000,
    timeout: float = 600.0,
) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-c:a", "libopus",
        "-b:a", bitrate,
        "-f", "webm",
        "-cluster_time_limit", str(cluster_time_limit_ms),
        "-live", "1",
        str(out_webm_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg失敗: {result.stderr.decode(errors='replace')[:2000]}")


def mux_wav_bytes_to_webm_bytes(
    wav_bytes: bytes,
    sample_rate: int = 48000,
    bitrate: str = "32k",
    cluster_time_limit_ms: int = 5000,
) -> bytes:
    cmd = [
        "ffmpeg", "-y",
        "-i", "pipe:0",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-c:a", "libopus",
        "-b:a", bitrate,
        "-f", "webm",
        "-cluster_time_limit", str(cluster_time_limit_ms),
        "-live", "1",
        "pipe:1",
    ]
    result = subprocess.run(cmd, input=wav_bytes, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg失敗: {result.stderr.decode(errors='replace')[:2000]}")
    return result.stdout
