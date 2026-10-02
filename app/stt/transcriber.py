import logging
import subprocess
from dataclasses import dataclass

import numpy as np
import torch
from faster_whisper import WhisperModel

from stt.ebml_utils import find_first_cluster_offset

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
EBML_MAGIC = b"\x1a\x45\xdf\xa3"


@dataclass
class Segment:
    start: float  # チャンク先頭からの秒
    end: float
    text: str


class StreamDecoder:
    """1本のWebSocket接続(=1つのMediaRecorder)のWebM/Opusチャンク列を16kHzモノラルPCMに変換する。

    MediaRecorderは最初のチャンクにだけWebMヘッダーを付けるため、2個目以降はヘッダーを前置して
    ffmpegに渡す必要がある。ヘッダーは接続ごとに持つ(以前はTranscriber全体で1つだけ保存しており、
    BTと内蔵マイク・前回の試験・リロード後の新しい録音で同じヘッダーを使い回していた)。
    チャンクが新しいヘッダーで始まっていれば(録音のやり直し等)、そのヘッダーに置き換える。
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.header: bytes | None = None

    def decode(self, chunk: bytes) -> np.ndarray | None:
        if chunk.startswith(EBML_MAGIC):
            offset = find_first_cluster_offset(chunk)
            if offset is None:
                logger.warning(f"[{self.label}] EBML解析失敗。フォールバックとして先頭5120バイトを使用")
                offset = min(5120, len(chunk))
            if self.header is not None:
                logger.info(f"[{self.label}] 新しいWebMヘッダーを検出(録音の再開始)")
            self.header = chunk[:offset]
            data = chunk
        elif self.header is None:
            logger.warning(f"[{self.label}] ヘッダーのないチャンクを受信(先頭チャンクを取りこぼした可能性)")
            return None
        else:
            data = self.header + chunk
        return _ffmpeg_to_pcm(data, self.label)


def _ffmpeg_to_pcm(data: bytes, label: str) -> np.ndarray | None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0", "-ar", str(SAMPLE_RATE), "-ac", "1",
             "-f", "s16le", "pipe:1"],
            input=data, capture_output=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"[{label}] ffmpeg タイムアウト")
        return None
    if result.returncode != 0:
        logger.warning(f"ffmpeg変換失敗: [{label}] returncode={result.returncode} stderr={result.stderr.decode()[:300]}")
        return None
    return np.frombuffer(result.stdout, dtype=np.int16)


# Whisperが雑音・無音から作りやすい定型文(学習データの動画字幕に由来するもの)。
# 第2期試験で、工場騒音だけの区間から10分あたり600〜2400文字がこれらで埋まることを確認した
HALLUCINATION_PHRASES = ("ご視聴ありがとうございました", "チャンネル登録", "高評価", "おやすみなさい",
                         "字幕は", "お疲れ様でした。ご視聴")


def _is_hallucination(seg, text: str) -> bool:
    if any(p in text for p in HALLUCINATION_PHRASES) and len(text) <= 40:
        return True
    # 発話が無い可能性が高く、かつ認識の確信度も低いセグメント
    if seg.no_speech_prob > 0.6 and seg.avg_logprob < -0.8:
        return True
    # 同じ文字列の繰り返し(圧縮率が高い)
    if seg.compression_ratio > 2.4:
        return True
    return False


def rms(pcm: np.ndarray | None) -> float:
    """音量レベル(int16スケールのRMS)。BT画面の音量メーター表示用"""
    if pcm is None or len(pcm) == 0:
        return 0.0
    x = pcm.astype(np.float32)
    return float(np.sqrt(np.mean(x ** 2)))


class Transcriber:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self.model: WhisperModel | None = None
        self.filter_hallucination = True
        self._load()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self):
        """STTモデルをVRAMにロードする。既にロード済みなら何もしない。"""
        if self.is_loaded:
            return
        self._load()

    def _load(self):
        logger.info(f"faster-whisper [{self._model_size}] をロード中...")
        self.model = WhisperModel(self._model_size, device=self._device, compute_type=self._compute_type)
        logger.info("faster-whisper ロード完了")

    def unload(self):
        """VRAMを解放する(LLM処理前に呼ぶ)。既に未ロードなら何もしない。"""
        if not self.is_loaded:
            return
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("STTモデルをVRAMから解放しました")

    def transcribe_pcm(self, pcm: np.ndarray, vad_filter: bool = True, vad_params: dict | None = None,
                       initial_prompt: str | None = None) -> list[Segment]:
        """16kHz int16 PCM を文字起こしし、チャンク内の時刻付きセグメントを返す"""
        model = self.model
        if model is None:
            raise RuntimeError("STTモデルが読み込まれていません")
        audio = pcm.astype(np.float32) / 32768.0
        segments, info = model.transcribe(
            audio,
            language="ja",
            vad_filter=vad_filter,
            vad_parameters=vad_params or {"min_silence_duration_ms": 300, "speech_pad_ms": 300, "threshold": 0.3},
            beam_size=5,
            initial_prompt=initial_prompt or None,
            condition_on_previous_text=False,
        )
        out = []
        for s in segments:
            text = s.text.strip()
            if not text:
                continue
            if self.filter_hallucination and _is_hallucination(s, text):
                logger.info(f"  seg [{s.start:.1f}-{s.end:.1f}] 除外(ハルシネーションの疑い): {text[:40]} "
                            f"no_speech={s.no_speech_prob:.2f} logprob={s.avg_logprob:.2f}")
                continue
            out.append(Segment(s.start, s.end, text))
            logger.info(f"  seg [{s.start:.1f}-{s.end:.1f}]: {text[:80]}")
        return out

    # 旧インターフェース(試験スクリプト互換)。ヘッダーの扱いは呼び出し側の StreamDecoder に任せる
    _compat_decoder: StreamDecoder | None = None

    def transcribe_bytes(self, audio_bytes: bytes, vad_filter: bool = True, vad_params: dict | None = None) -> str:
        if self._compat_decoder is None:
            self._compat_decoder = StreamDecoder("compat")
        pcm = self._compat_decoder.decode(audio_bytes)
        if pcm is None:
            return ""
        return " ".join(s.text for s in self.transcribe_pcm(pcm, vad_filter, vad_params)).strip()

    def reset_compat_stream(self):
        self._compat_decoder = None
