"""L1 話者判定の限界試験(TEST_PLAN_v2.md B群)。GPU不要・ホストで実行。

本番の判定(app/stt/speaker.py + websocket.py)を時刻まで含めてオフラインで再現する:
- BTマイクの音声を本番と同じ WebM/Opus → 16kHz int16 に通し、5秒ごとのRMSを求める
- 実時間の対応: BT録音を時刻0に開始、内蔵マイクは offset 秒後に開始。どちらも同じ瞬間の音を拾う
- BTのRMS記録の時刻 = そのチャンクの音声の終わり(=ブラウザが送出する時刻)+ BT処理遅延
- 内蔵マイクのチャンク j の判定窓 = [offset + 5j, offset + 5(j+1)](本番は受信時刻の差で決まる)
- 窓内にRMS記録があればその最大値で、なければ直近の記録で閾値判定(get_speaker と同じ)

採点は「発話時間の重み付き正解率」: 各窓で実際に話している人の立場ごとの発話秒数を数え、
判定されたラベルと一致する秒数の割合。1つの窓に両者が話していれば、どう判定しても一部は誤りになる。

使い方:
    .venv/Scripts/python.exe l1/speaker_sim.py --tag base
結果: results/v2/l1_speaker/<tag>/{rows.jsonl, summary.md}
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTING_DIR = HERE.parent
sys.path.insert(0, str(TESTING_DIR / "augment"))
from mix import make_mix  # noqa: E402

AUDIO_DIR = TESTING_DIR / "results" / "v2" / "audio"
PY = sys.executable
WIN = 5.0
OFFSETS = [0.5, 1.5, 2.5, 3.5, 4.5]  # 内蔵マイク録音開始がBTより何秒遅いか(実運用ではばらつく)
THRESHOLDS = [200, 400, 800, 1200, 1600, 2400, 3200]


def ensure_stems(name: str, scen: str, extra: list[str]) -> Path:
    d = AUDIO_DIR / name
    if not (d / "wearer.wav").exists():
        subprocess.run([PY, str(TESTING_DIR / "tts" / "generate_audio_v2.py"),
                        str(TESTING_DIR / "scenario" / "generated" / scen), "--out", str(d), *extra], check=True)
    return d


FRAME = 0.05


def bt_pcm(bt_wav: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        webm = Path(td) / "bt.webm"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(bt_wav), "-ar", "48000", "-ac", "1",
                        "-c:a", "libopus", "-b:a", "32k", str(webm)], check=True)
        pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(webm), "-ar", "16000", "-ac", "1",
                              "-f", "s16le", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32)


def frame_activity(pcm: np.ndarray, history_sec: float = 60.0) -> np.ndarray:
    """改善案: 50msフレームごとに「装着者が話しているか」を判定する。閾値は直近 history_sec 秒の
    フレーム音量の分布(下位20%=雑音の床、上位5%=発話の大きさ)から自動で決める(手動の閾値調整が不要)。
    本番では5秒チャンクが届くたびに、それまでの履歴で閾値を更新する形になる。"""
    fl = int(16000 * FRAME)
    n = len(pcm) // fl
    db = 10 * np.log10((pcm[: n * fl].reshape(n, fl) ** 2).mean(axis=1) + 1.0)
    act = np.zeros(n, dtype=bool)
    per_chunk = int(WIN / FRAME)
    hist = int(history_sec / FRAME)
    for c0 in range(0, n, per_chunk):
        h = db[max(0, c0 + per_chunk - hist): c0 + per_chunk]
        floor, peak = np.percentile(h, 20), np.percentile(h, 95)
        seg = db[c0: c0 + per_chunk]
        if peak - floor < 10:  # 履歴内に発話らしい大きな音がない
            continue
        act[c0: c0 + per_chunk] = seg > floor + 0.5 * (peak - floor)
    return act


def simulate_v2(act: np.ndarray, turns: list[dict], total_sec: float, offset: float, bt_delay: float,
                per_segment: bool) -> float:
    """改善案の判定。BTの各フレームの実時刻 = BT開始からの時刻 + 到着遅れの見積もり誤差(bt_delay)。
    per_segment=False: 5秒窓ごとに1ラベル(窓と時間が重なるBTフレームの発話率で判定)
    per_segment=True : 窓を発話の切れ目(Whisperのセグメントに相当)で分け、区間ごとにラベル"""
    t_frame = np.arange(len(act)) * FRAME + bt_delay
    correct = total = 0.0
    j = 0
    while offset + (j + 1) * WIN <= total_sec:
        a, b = offset + j * WIN, offset + (j + 1) * WIN
        j += 1
        pieces = [(a, b)]
        if per_segment:
            cuts = sorted({a, b} | {x for t in turns for x in (t["start_sec"], t["end_sec"]) if a < x < b})
            pieces = list(zip(cuts[:-1], cuts[1:]))
        for pa, pb in pieces:
            rs = role_seconds(turns, pa, pb)
            speech = rs["client"] + rs["our_side"]
            if speech < 0.05:
                continue
            m = (t_frame >= pa) & (t_frame < pb)
            frac = act[m].mean() if m.any() else 0.0
            label = "our_side" if frac > (0.3 if per_segment else 0.15) else "client"
            correct += rs[label]
            total += speech
    return correct / total if total else None


def bt_rms_series(bt_wav: Path) -> np.ndarray:
    """本番と同じ経路(Opus 32kbps → 16kHz mono int16)を通して5秒ごとのRMSを返す"""
    with tempfile.TemporaryDirectory() as td:
        webm = Path(td) / "bt.webm"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(bt_wav), "-ar", "48000", "-ac", "1",
                        "-c:a", "libopus", "-b:a", "32k", str(webm)], check=True)
        pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(webm), "-ar", "16000", "-ac", "1",
                              "-f", "s16le", "pipe:1"], capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n = int(len(x) // (16000 * WIN))
    return np.sqrt((x[: n * 80000].reshape(n, 80000) ** 2).mean(axis=1))


def role_seconds(turns: list[dict], a: float, b: float) -> dict:
    out = {"client": 0.0, "our_side": 0.0, "our_side_nonbt": 0.0}
    for t in turns:
        s, e = max(a, t["start_sec"]), min(b, t["end_sec"])
        if e > s:
            out[t["role"]] += e - s
            if t["role"] == "our_side" and not t["is_bt_wearer"]:
                out["our_side_nonbt"] += e - s
    return out


def simulate(rms: np.ndarray, turns: list[dict], total_sec: float, threshold: float, offset: float,
             bt_delay: float) -> dict:
    rec_t = np.array([(k + 1) * WIN + bt_delay for k in range(len(rms))])
    correct = total = nonbt_total = nonbt_as_client = 0.0
    ideal_correct = 0.0
    j = 0
    while offset + (j + 1) * WIN <= total_sec:
        a, b = offset + j * WIN, offset + (j + 1) * WIN
        j += 1
        rs = role_seconds(turns, a, b)
        speech = rs["client"] + rs["our_side"]
        if speech < 0.3:  # 発話がほぼ無い窓は本番でも文字起こしが出ないので数えない
            continue
        in_win = (rec_t >= a) & (rec_t <= b)
        if in_win.any():
            val = rms[in_win].max()
        else:
            past = rec_t <= b  # 判定時点(内蔵チャンク受信時)までに届いている直近の記録
            val = rms[past][-1] if past.any() else 0.0
        label = "our_side" if val > threshold else "client"
        correct += rs[label]
        ideal_correct += max(rs["client"], rs["our_side"])
        total += speech
        nonbt_total += rs["our_side_nonbt"]
        if label == "client":
            nonbt_as_client += rs["our_side_nonbt"]
    return {"acc": correct / total if total else None, "ceiling": ideal_correct / total if total else None,
            "nonbt_share": nonbt_total / total if total else 0}


def run_condition(stem_dir: Path, cond: dict, work: Path, seed: int) -> dict:
    tl = json.loads((stem_dir / "timeline.json").read_text(encoding="utf-8"))
    tmp = work / "tmp"
    make_mix(stem_dir, cond, tmp, seed=seed, babble_path=AUDIO_DIR / "babble.wav")
    rms = bt_rms_series(tmp / "bt_mix.wav")
    turns, total = tl["turns"], tl["total_duration_sec"]
    # RMSの分布(装着者が話している窓 / 話していない窓)
    wearer_win, silent_win = [], []
    for k, v in enumerate(rms):
        rs = [t for t in turns if t["is_bt_wearer"] and t["start_sec"] < (k + 1) * WIN and t["end_sec"] > k * WIN]
        (wearer_win if rs else silent_win).append(float(v))
    act = frame_activity(bt_pcm(tmp / "bt_mix.wav"))
    delay = cond.get("_bt_delay", 0.2)
    v2_win = [simulate_v2(act, turns, total, off, delay, False) for off in OFFSETS]
    v2_seg = [simulate_v2(act, turns, total, off, delay, True) for off in OFFSETS]
    res_v2 = {"v2_window": float(np.mean(v2_win)), "v2_window_min": float(np.min(v2_win)),
              "v2_segment": float(np.mean(v2_seg)), "v2_segment_min": float(np.min(v2_seg))}
    res = res_v2 | {"rms_wearer_p10": float(np.percentile(wearer_win, 10)) if wearer_win else None,
           "rms_silent_p90": float(np.percentile(silent_win, 90)) if silent_win else None, "by_threshold": {}}
    for th in THRESHOLDS:
        accs = [simulate(rms, turns, total, th, off, cond.get("_bt_delay", 0.2)) for off in OFFSETS]
        res["by_threshold"][th] = {"mean": float(np.mean([a["acc"] for a in accs])),
                                   "min": float(np.min([a["acc"] for a in accs]))}
        if th == 800:
            res["ceiling"] = float(np.mean([a["ceiling"] for a in accs]))
            res["nonbt_share"] = accs[0]["nonbt_share"]
    return res


def conditions() -> list[tuple[str, str, dict]]:
    normal = "s_sg_normal"
    fast = "s_sg_fast"
    c = [("base", normal, {})]
    c += [(f"leak{d}dB", normal, {"bt_leak_db": d}) for d in [-40, -30, -20, -10]]
    c += [(f"wearer{d}dB", normal, {"bt_wearer_db": d}) for d in [-10, -20, -30]]
    c += [(f"btnoise_snr{s}dB", normal, {"bt_noise_snr_db": s}) for s in [30, 20, 10, 0]]
    c += [("fast_turns", fast, {})]
    c += [(f"btdelay{d}s", normal, {"_bt_delay": d}) for d in [1.0, 2.0, 3.0]]
    c += [("band16k", normal, {"bt_band": "16k"}), ("band8k", normal, {"bt_band": "8k"}),
          ("dropout5pct", normal, {"bt_dropout": 0.05})]
    c += [("realistic_leak-20+noise20", normal, {"bt_leak_db": -20, "bt_noise_snr_db": 20, "bt_band": "16k"})]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    work = TESTING_DIR / "results" / "v2" / "l1_speaker" / args.tag
    work.mkdir(parents=True, exist_ok=True)
    ensure_stems("s_sg_normal", "s_sg", [])
    ensure_stems("s_sg_fast", "s_sg", ["--gap", "0.1", "0.3", "--overlap-prob", "0.3", "--backchannel-prob", "0.4", "--seed", "1"])

    rows = []
    for i, (name, stems, cond) in enumerate(conditions()):
        r = run_condition(AUDIO_DIR / stems, cond, work, seed=i) | {"cond": name}
        rows.append(r)
        print(name, json.dumps(r["by_threshold"][800]), flush=True)
    with open(work / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    head = ("| 条件 | 上限(5秒窓) | " + " | ".join(f"閾値{t}" for t in THRESHOLDS)
            + " | 装着者発話窓RMS p10 | 無発話窓RMS p90 | 改善案:窓単位 | 改善案:区間単位 |")
    lines = [head, "|" + " --- |" * (len(THRESHOLDS) + 6)]
    for r in rows:
        cells = [f"{r['by_threshold'][t]['mean']:.3f} ({r['by_threshold'][t]['min']:.2f})" for t in THRESHOLDS]
        lines.append(f"| {r['cond']} | {r['ceiling']:.3f} | " + " | ".join(cells)
                     + f" | {r['rms_wearer_p10']:.0f} | {r['rms_silent_p90']:.0f} | {r['v2_window']:.3f} ({r['v2_window_min']:.2f})"
                     + f" | {r['v2_segment']:.3f} ({r['v2_segment_min']:.2f}) |")
    lines.append("")
    lines.append(f"数値は内蔵マイクの開始遅れ {OFFSETS} 秒の平均(括弧内は最小)。BT非装着の自社側発話の割合: {rows[0]['nonbt_share']:.3f}")
    md = "\n".join(lines)
    (work / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
