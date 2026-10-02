"""BTイヤホン(自社側が装着)のマイク音声から話者を判定する。

判定方式(speaker.method):
- adaptive(既定): BTマイクの音声を50msのフレームに分け、各フレームで装着者が話しているかを判定する。
    閾値は直近 history_seconds 秒のフレーム音量から自動で決める(雑音の床の少し上、かつ
    装着者の発話の大きさから margin_db 以内)。内蔵マイクの文字起こしセグメント(時刻付き)ごとに、
    同じ時間帯のBTフレームの発話率が segment_active_ratio を超えれば our_side とする。
    手動の閾値調整は不要で、装着者の声の大きさ・マイク感度・BT側の雑音に左右されにくい。
- legacy: 旧方式。5秒チャンク全体のRMSが bt_rms_threshold を超えれば our_side。

第2期評価試験(testing/TEST_PLAN_v2.md)の話者判定模擬で、旧方式は発話時間の重み付き正解率0.59、
adaptive方式は0.80(BTを付けていない自社側話者の発話を除けば上限に相当)だった。

時刻はサーバー受信時刻ではなく「音声としての時刻」で扱う。各WebSocket接続で最初のチャンクの
受信時刻からその長さを引いた時刻を録音開始時刻とし、以降はチャンクの長さを積み上げる。
こうすると、文字起こしが遅れて受信処理が詰まっても、BTと内蔵マイクの時刻がずれない。
"""
import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

FRAME_SEC = 0.05
SAMPLE_RATE = 16000


@dataclass
class RMSRecord:
    timestamp: float
    rms: float


class SpeakerDetector:
    """セッションごとに1つ持つ(以前はプロセス全体で共有しており、別セッションや前回の試験の
    BT音量・閾値が混ざっていた)。"""

    def __init__(self, threshold: float = 800.0, window_seconds: float = 10.0, method: str = "adaptive",
                 history_seconds: float = 60.0, margin_db: float = 16.0, floor_margin_db: float = 6.0,
                 min_dynamic_db: float = 10.0, segment_active_ratio: float = 0.3):
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.method = method
        self.history_seconds = history_seconds
        self.margin_db = margin_db
        self.floor_margin_db = floor_margin_db
        self.min_dynamic_db = min_dynamic_db
        self.segment_active_ratio = segment_active_ratio
        self._history: deque[RMSRecord] = deque()
        # adaptive用: (フレーム開始時刻, dB) を時刻順に保持
        self._frame_t: deque[float] = deque()
        self._frame_db: deque[float] = deque()
        self.bt_connected = False

    # ------------------------------------------------------------------
    # BT側の入力
    # ------------------------------------------------------------------

    def add_bt_rms(self, rms: float, timestamp: float | None = None):
        now = timestamp or time.time()
        self._history.append(RMSRecord(timestamp=now, rms=rms))
        cutoff = now - max(self.window_seconds * 2, self.history_seconds * 2)
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()

    def add_bt_audio(self, pcm: np.ndarray, start_time: float):
        """BTマイクのPCM(16kHz int16)と、その音声の開始時刻(音声としての時刻)を登録する"""
        fl = int(SAMPLE_RATE * FRAME_SEC)
        n = len(pcm) // fl
        if n == 0:
            return
        x = pcm[: n * fl].astype(np.float32).reshape(n, fl)
        db = 10 * np.log10((x ** 2).mean(axis=1) + 1.0)
        for i in range(n):
            self._frame_t.append(start_time + i * FRAME_SEC)
            self._frame_db.append(float(db[i]))
        cutoff = start_time + n * FRAME_SEC - self.history_seconds * 2
        while self._frame_t and self._frame_t[0] < cutoff:
            self._frame_t.popleft()
            self._frame_db.popleft()

    @property
    def bt_covered_until(self) -> float:
        """BT音声が届いている最後の時刻"""
        return self._frame_t[-1] + FRAME_SEC if self._frame_t else 0.0

    # ------------------------------------------------------------------
    # 判定
    # ------------------------------------------------------------------

    def _active_ratio(self, a: float, b: float) -> float | None:
        """[a, b) の間でBT装着者が話していたフレームの割合。BTデータが無ければ None"""
        if not self._frame_t:
            return None
        t = np.fromiter(self._frame_t, dtype=np.float64)
        db = np.fromiter(self._frame_db, dtype=np.float64)
        hist = db[(t >= b - self.history_seconds) & (t < b)]
        sel = db[(t >= a) & (t < b)]
        if len(sel) == 0 or len(hist) < 20:
            return None
        floor, peak = np.percentile(hist, 20), np.percentile(hist, 98)
        if peak - floor < self.min_dynamic_db:  # 直近に大きな音(装着者の発話)が無い
            return 0.0
        thr = max(floor + self.floor_margin_db, peak - self.margin_db)
        return float((sel > thr).mean())

    def label(self, a: float, b: float) -> str:
        """[a, b) の区間の話者。BTデータが無い区間は client"""
        if self.method == "legacy":
            return self.get_speaker(a, b)
        r = self._active_ratio(a, b)
        return "our_side" if r is not None and r > self.segment_active_ratio else "client"

    def get_speaker(self, start_time: float, end_time: float) -> str:
        """旧方式: 指定時間範囲内のBT RMS記録の最大値で判定する"""
        records = [r for r in self._history if start_time <= r.timestamp <= end_time]
        if not records:
            if self._history:
                latest = self._history[-1]
                return "our_side" if latest.rms > self.threshold else "client"
            return "client"
        return "our_side" if max(r.rms for r in records) > self.threshold else "client"

    def update_threshold(self, new_threshold: float):
        """旧方式(legacy)の閾値を変更する。adaptive方式では使わない"""
        self.threshold = new_threshold
        logger.info(f"話者判定閾値を更新: {new_threshold}")

    def clear(self):
        self._history.clear()
        self._frame_t.clear()
        self._frame_db.clear()

    @property
    def latest_rms(self) -> float:
        return self._history[-1].rms if self._history else 0.0
