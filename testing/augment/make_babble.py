"""周囲の話し声(バブル雑音)の素材を作る。試験の会話とは関係のない雑談を、試験の話者とは
別の声で合成し、4系統を時間をずらして重ねる。augment/mix.py の noise: [babble] で使う。

    .venv/Scripts/python.exe augment/make_babble.py   → results/v2/audio/babble.wav
"""
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

TESTING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TESTING_DIR / "tts"))
from generate_audio_v2 import SR, synth  # noqa: E402
from voicevox_client import VoicevoxClient  # noqa: E402

VOICES = [8, 2, 42, 14]  # 春日部つむぎ / 四国めたん / ちび式じい / 冥鳴ひまり(s_e2e_mix 等の話者とは別の声)
TEXTS = [
    "午後の搬入はトラックが二台来る予定なので、フォークリフトを空けておいてください。",
    "来週の安全パトロールの当番表はもう回ってきましたか。",
    "倉庫の棚の整理、今日中に終わらせたいんですよね。",
    "この部品、在庫があと三つしかないので発注しておきます。",
    "昨日の夜勤の引き継ぎ書、どこに置きましたっけ。",
    "会議室の予約が取れなかったので、打ち合わせは食堂でやりましょう。",
    "クレーンの点検が終わるまで、こっちの作業は待ってもらえますか。",
    "新人研修の資料、印刷は何部必要でしょうか。",
    "今日は暑いので、休憩はこまめに取ってくださいね。",
    "梱包材が足りなくなりそうなので、総務に連絡しておきます。",
]


def main():
    rng = random.Random(0)
    client = VoicevoxClient()
    tracks = []
    for vid in VOICES:
        texts = TEXTS[:]
        rng.shuffle(texts)
        parts = []
        for t in texts:
            parts.append(synth(client, t, {"speaker_id": vid}, 1.0))
            parts.append(np.zeros(int(rng.uniform(0.3, 1.5) * SR), dtype="float32"))
        tracks.append(np.concatenate(parts))
    client.close()
    n = min(len(t) for t in tracks)
    babble = np.zeros(n, dtype="float32")
    for i, t in enumerate(tracks):
        babble += np.roll(t[:n], int(i * n / len(tracks)))
    out = TESTING_DIR / "results" / "v2" / "audio" / "babble.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, babble / np.abs(babble).max() * 0.5, SR)
    print(f"完了: {out} {n / SR:.0f}秒")


if __name__ == "__main__":
    main()
