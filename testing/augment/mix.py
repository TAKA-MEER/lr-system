"""
素材(wearer.wav / others.wav)に音響劣化を重ねて、注入用の internal_mix.wav と bt_mix.wav を作る。
すべて数式で合成し(外部の音源データは使わない)、乱数シードで再現できる。

条件(cond dict)の項目。省略時は劣化なし:
  内蔵マイク側
    int_gain_db:     話者の声の音量(マイクまでの距離)。0が基準
    rt60:            残響時間[秒]。0なら残響なし
    noise:           騒音の種類のリスト。factory(モーター/変圧器ハム+ファン+ピンク) / pink / impulse / alarm / babble
    snr_db:          騒音の強さ(発話部分の音声パワーとの比)。noise指定時に使用
    impulse_per_min: 衝撃音の回数/分(noiseにimpulseがある場合)
    impulse_db:      衝撃音のピークの強さ(発話のピーク基準、dB)
    babble_snr_db:   周囲の話し声の強さ(noiseにbabbleがある場合)
    clip_db:         入力を持ち上げてクリップさせる量(dB)。0なら何もしない
    ref_power:       SNRの基準にする発話パワー(騒音だけの素材で使う)
  BTマイク側
    bt_wearer_db:    装着者の声の音量(小声の模擬)
    bt_leak_db:      装着者以外の声がBTマイクに入る量(装着者の声基準、dB)。None なら漏れ込みなし
    bt_noise_snr_db: BTマイクに入る騒音(装着者の声基準)。None なら無し
    bt_band:         48k / 16k(HFP広帯域) / 8k(HFP狭帯域)
    bt_dropout:      音の途切れの割合(0〜1)
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 24000


def db(x: float) -> float:
    return 10 ** (x / 20)


def active_power(x: np.ndarray) -> float:
    """発話区間の平均パワー(無音部分を除いて計算。SNRの基準)"""
    frame = SR // 50
    n = len(x) // frame
    if n == 0:
        return float(np.mean(x ** 2) + 1e-12)
    p = (x[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    thr = p.max() * 1e-3
    act = p[p > thr]
    return float(act.mean()) if len(act) else 1e-12


def scale_to_snr(noise: np.ndarray, speech_power: float, snr_db: float) -> np.ndarray:
    npow = float(np.mean(noise ** 2)) + 1e-12
    return noise * np.sqrt(speech_power / npow / (10 ** (snr_db / 10)))


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    f = np.fft.rfft(white)
    k = np.arange(len(f))
    k[0] = 1
    return np.fft.irfft(f / np.sqrt(k), n).astype("float32")


def factory_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """モーター・変圧器のハム(50Hzとその高調波、100Hz系を強め)+ ファンの帯域雑音 + ピンク雑音"""
    t = np.arange(n) / SR
    hum = np.zeros(n)
    for h in range(1, 13):
        amp = (1.0 if h % 2 == 0 else 0.5) / h
        hum += amp * np.sin(2 * np.pi * 50 * h * t + rng.uniform(0, 2 * np.pi))
    hum *= 1 + 0.1 * np.sin(2 * np.pi * 0.3 * t)  # ゆっくりしたうなり
    b, a = signal.butter(2, [300 / (SR / 2), 2500 / (SR / 2)], btype="band")
    fan = signal.lfilter(b, a, rng.standard_normal(n))
    p = pink(n, rng)
    parts = [hum / np.std(hum), fan / np.std(fan), p / np.std(p)]
    return (0.5 * parts[0] + 0.6 * parts[1] + 0.6 * parts[2]).astype("float32")


def impulse_noise(n: int, rng: np.random.Generator, per_min: float, peak: float) -> np.ndarray:
    """打撃音・リレー・遮断器動作音(減衰する短いバースト)"""
    out = np.zeros(n, dtype="float32")
    count = int(per_min * n / SR / 60)
    for _ in range(count):
        pos = rng.integers(0, max(1, n - SR // 2))
        dur = int(rng.uniform(0.03, 0.15) * SR)
        env = np.exp(-np.arange(dur) / (dur / 5))
        burst = rng.standard_normal(dur) * env
        burst = burst / (np.abs(burst).max() + 1e-9) * peak * rng.uniform(0.5, 1.0)
        out[pos:pos + dur] += burst[: n - pos]
    return out


def alarm_noise(n: int) -> np.ndarray:
    """警報ブザー: 60秒ごとに10秒間、2kHzを0.5秒おきに断続"""
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 2000 * t) + 0.3 * np.sin(2 * np.pi * 4000 * t)
    gate = ((t % 60) < 10) & ((t % 1.0) < 0.5)
    return (tone * gate).astype("float32")


def reverb(x: np.ndarray, rt60: float, rng: np.random.Generator) -> np.ndarray:
    """指数減衰する雑音でインパルス応答を合成して畳み込む(直接音+残響)"""
    length = int(rt60 * SR)
    t = np.arange(length) / SR
    tail = rng.standard_normal(length) * np.exp(-6.9 * t / rt60)
    tail[: int(0.005 * SR)] = 0
    ir = tail * 0.3
    ir[0] = 1.0
    y = signal.fftconvolve(x, ir)[: len(x)]
    return (y * np.sqrt(np.mean(x ** 2) / (np.mean(y ** 2) + 1e-12))).astype("float32")


def band_limit(x: np.ndarray, band: str) -> np.ndarray:
    if band in (None, "48k", "24k"):
        return x
    target = 16000 if band == "16k" else 8000
    y = signal.resample_poly(x, target, SR)
    return signal.resample_poly(y, SR, target)[: len(x)].astype("float32")


def dropout(x: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    if not ratio:
        return x
    y = x.copy()
    total = int(ratio * len(x))
    cut = 0
    while cut < total:
        d = int(rng.uniform(0.02, 0.2) * SR)
        p = rng.integers(0, len(x) - d)
        y[p:p + d] = 0
        cut += d
    return y


def load_babble(n: int, path: Path) -> np.ndarray:
    b, sr = sf.read(path, dtype="float32")
    assert sr == SR
    reps = int(np.ceil(n / len(b)))
    return np.tile(b, reps)[:n]


def make_mix(stem_dir: Path, cond: dict, out_dir: Path, seed: int = 0, babble_path: Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    wearer, _ = sf.read(stem_dir / "wearer.wav", dtype="float32")
    others, _ = sf.read(stem_dir / "others.wav", dtype="float32")
    n = len(wearer)

    # ---- 内蔵マイク ----
    speech = wearer + others
    sp_pow = cond.get("ref_power") or active_power(speech)  # 発話のない素材では基準パワーを外から与える
    x = speech.copy()
    if cond.get("rt60"):
        x = reverb(x, cond["rt60"], rng)
    x = x * db(cond.get("int_gain_db", 0.0))
    noise = np.zeros(n, dtype="float32")
    kinds = cond.get("noise") or []
    ref_pow = sp_pow * db(cond.get("int_gain_db", 0.0)) ** 2
    if "factory" in kinds:
        noise += scale_to_snr(factory_noise(n, rng), ref_pow, cond["snr_db"])
    if "pink" in kinds:
        noise += scale_to_snr(pink(n, rng), ref_pow, cond["snr_db"])
    if "alarm" in kinds:
        noise += scale_to_snr(alarm_noise(n), ref_pow, cond.get("alarm_snr_db", cond.get("snr_db", 10)))
    if "babble" in kinds:
        noise += scale_to_snr(load_babble(n, babble_path), ref_pow, cond.get("babble_snr_db", 10))
    if "impulse" in kinds:
        peak = np.abs(x).max() * db(cond.get("impulse_db", 0.0))
        noise += impulse_noise(n, rng, cond.get("impulse_per_min", 6), peak)
    internal = x + noise
    if cond.get("clip_db"):
        internal = np.clip(internal * db(cond["clip_db"]), -1.0, 1.0)
    else:
        peak = np.abs(internal).max()
        if peak > 0.99:  # 基準の声より大きい条件でもクリップしないよう全体を下げる(音量比は保つ)
            internal = internal * 0.99 / peak

    # ---- BTマイク ----
    bt = wearer * db(cond.get("bt_wearer_db", 0.0))
    if cond.get("bt_leak_db") is not None:
        bt = bt + others * db(cond["bt_leak_db"])
    if cond.get("bt_noise_snr_db") is not None:
        bt = bt + scale_to_snr(factory_noise(n, rng), active_power(wearer), cond["bt_noise_snr_db"])
    bt = band_limit(bt, cond.get("bt_band"))
    bt = dropout(bt, cond.get("bt_dropout", 0.0), rng)
    bt = np.clip(bt, -1.0, 1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "internal_mix.wav", internal.astype("float32"), SR)
    sf.write(out_dir / "bt_mix.wav", bt.astype("float32"), SR)
    (out_dir / "cond.json").write_text(json.dumps(cond, ensure_ascii=False), encoding="utf-8")
    return {"internal": out_dir / "internal_mix.wav", "bt": out_dir / "bt_mix.wav"}
