import logging
import subprocess
import tempfile
import wave
import io
from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel

from stt.ebml_utils import find_first_cluster_offset

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self.model: WhisperModel | None = None
        self._webm_header: bytes | None = None  # 最初のチャンクのWebMヘッダー
        self._load()

    # ------------------------------------------------------------------
    # 公開メソッド
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self):
        """STTモデルをVRAMにロードする（次回録音開始前に呼ぶ）。既にロード済みなら何もしない。"""
        if self.is_loaded:
            return
        self._load()

    def _load(self):
        logger.info(f"faster-whisper [{self._model_size}] をロード中...")
        self.model = WhisperModel(self._model_size, device=self._device, compute_type=self._compute_type)
        logger.info("faster-whisper ロード完了")

    def transcribe_bytes(self, audio_bytes: bytes, vad_filter: bool = True, vad_params: dict | None = None) -> str:
        """WebM/Opus バイト列を日本語テキストに変換して返す"""
        wav = self._to_wav(audio_bytes)
        if not wav:
            logger.warning("ffmpeg変換失敗: wav=None")
            return ""

        logger.info(f"transcribe_bytes: wav_size={len(wav)} vad_filter={vad_filter}")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(wav)
            tmp.close()

            vad_parameters = vad_params or {
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 300,
                "threshold": 0.3,
            }

            segments, info = self.model.transcribe(
                tmp.name,
                language="ja",
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                beam_size=5,
            )
            segments_list = list(segments)
            logger.info(f"transcribe_bytes: segments={len(segments_list)} duration={info.duration if info else '?'}")
            if segments_list:
                for seg in segments_list:
                    logger.info(f"  seg [{seg.start:.1f}-{seg.end:.1f}]: {seg.text.strip()[:80]}")
            text = " ".join(s.text.strip() for s in segments_list)
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
        """VRAMを解放する（LLM処理前に呼ぶ）。既に未ロードなら何もしない。"""
        if not self.is_loaded:
            return
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("STTモデルをVRAMから解放しました")

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _to_wav(self, audio_bytes: bytes) -> bytes | None:
        """ffmpeg で 16kHz モノラル WAV に変換"""
        # 2回目以降のチャンクはWebMヘッダーがないため、
        # 最初のチャンクの先頭部分をヘッダーとして付加する
        input_data = audio_bytes
        if self._webm_header:
            input_data = self._webm_header + audio_bytes

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
                input=input_data,
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(f"ffmpeg変換失敗: returncode={result.returncode} stderr={result.stderr.decode()[:500]}")
                return None

            # 最初の成功時にWebMヘッダーを保存
            # 注意: 単純に先頭Nバイトを切り出すと、最初のCluster(音声フレーム)の一部まで
            # ヘッダーとして保存してしまい、2個目以降の全チャンクにその音声が重複して
            # 前置される(=文字起こしに重複した断片が混入する)。EBML構造を辿って
            # 最初のCluster開始位置(=音声フレームを含まない真のヘッダー終端)を正確に求める。
            if not self._webm_header:
                cluster_offset = find_first_cluster_offset(audio_bytes)
                if cluster_offset is not None:
                    self._webm_header = audio_bytes[:cluster_offset]
                else:
                    logger.warning("EBML解析失敗。フォールバックとして先頭5120バイトを使用")
                    self._webm_header = audio_bytes[:5120]
                logger.info(f"WebMヘッダーを保存: {len(self._webm_header)} bytes")

            return result.stdout
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg タイムアウト")
            return None
        except Exception as e:
            logger.error(f"ffmpeg 例外: {e}")
            return None
