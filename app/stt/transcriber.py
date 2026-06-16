import logging
import subprocess
import tempfile
import wave
import io
from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        logger.info(f"faster-whisper [{model_size}] をロード中...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._device = device
        logger.info("faster-whisper ロード完了")

    # ------------------------------------------------------------------
    # 公開メソッド
    # ------------------------------------------------------------------

    def transcribe_bytes(self, audio_bytes: bytes, vad_params: dict | None = None) -> str:
        """WebM/Opus バイト列を日本語テキストに変換して返す"""
        wav = self._to_wav(audio_bytes)
        if not wav:
            return ""

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(wav)
            tmp.close()

            vad_parameters = vad_params or {
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 300,
                "threshold": 0.3,
            }

            segments, _ = self.model.transcribe(
                tmp.name,
                language="ja",
                vad_filter=True,
                vad_parameters=vad_parameters,
                beam_size=5,
            )
            text = " ".join(s.text.strip() for s in segments)
            return text.strip()
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def get_rms(self, audio_bytes: bytes) -> float:
        """音量レベル（RMS）を返す。BTマイクの話者判定に使用する"""
        wav = self._to_wav(audio_bytes)
        if not wav:
            return 0.0
        try:
            with wave.open(io.BytesIO(wav), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            if len(audio) == 0:
                return 0.0
            return float(np.sqrt(np.mean(audio ** 2)))
        except Exception as e:
            logger.warning(f"RMS計算エラー: {e}")
            return 0.0

    def unload(self):
        """VRAMを解放する（LLM処理前に呼ぶ）"""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("STTモデルをVRAMから解放しました")

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _to_wav(self, audio_bytes: bytes) -> bytes | None:
        """ffmpeg で 16kHz モノラル WAV に変換"""
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", "pipe:0",
                    "-ar", "16000",
                    "-ac", "1",
                    "-f", "wav",
                    "pipe:1",
                ],
                input=audio_bytes,
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.debug(f"ffmpeg stderr: {result.stderr.decode()[:300]}")
                return None
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg タイムアウト")
            return None
        except Exception as e:
            logger.error(f"ffmpeg 例外: {e}")
            return None
