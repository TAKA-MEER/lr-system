import time
import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RMSRecord:
    timestamp: float
    rms: float


class SpeakerDetector:
    """
    BTマイク（our_side装着）のRMS履歴から話者を判定する。

    BTマイクに音声あり → our_side が話している
    BTマイクが無音     → client が話している
    """

    def __init__(self, threshold: float = 800.0, window_seconds: float = 10.0):
        self.threshold = threshold
        self.window_seconds = window_seconds
        self._history: deque[RMSRecord] = deque()

    def add_bt_rms(self, rms: float):
        """BTマイクのRMS値を記録する"""
        now = time.time()
        self._history.append(RMSRecord(timestamp=now, rms=rms))
        # 古いレコードを削除
        cutoff = now - self.window_seconds * 2
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()

    def get_speaker(self, start_time: float, end_time: float) -> str:
        """
        指定時間範囲内のBT RMS履歴を見て話者を返す。
        その範囲にRMS>閾値のレコードがあれば our_side、なければ client。
        """
        records = [r for r in self._history if start_time <= r.timestamp <= end_time]

        if not records:
            # 範囲内に記録がない場合は直近のレコードで判定
            if self._history:
                latest = self._history[-1]
                speaker = "our_side" if latest.rms > self.threshold else "client"
                logger.debug(f"履歴なし、直近RMS={latest.rms:.0f} → {speaker}")
                return speaker
            logger.debug("BTマイク履歴なし → client")
            return "client"

        max_rms = max(r.rms for r in records)
        speaker = "our_side" if max_rms > self.threshold else "client"
        logger.debug(f"maxRMS={max_rms:.0f} / threshold={self.threshold} → {speaker}")
        return speaker

    def update_threshold(self, new_threshold: float):
        """設定画面などから閾値を動的に変更する"""
        self.threshold = new_threshold
        logger.info(f"話者判定閾値を更新: {new_threshold}")

    @property
    def latest_rms(self) -> float:
        if self._history:
            return self._history[-1].rms
        return 0.0
