"""
台本(scenario/generated/<name>/script.yaml)から VOICEVOX で音声を合成し、劣化前の「素材」を作る。

旧 generate_audio.py との違い:
- 出力は完成音声ではなく2本の素材(stems)。騒音・残響・BTへの漏れ込み等は augment/mix.py が後から重ねる
    wearer.wav : BTイヤホン装着者の発話だけ
    others.wav : それ以外の全員の発話(相手方 + BT非装着の自社側)
- 話者ごとの声・話速・声の高さ・抑揚を voice_map_v2.yaml で指定。発話ごとに話速を揺らせる
- ターン間の間を指定範囲で揺らし、一定確率で前の発話に重ねる(同時発話・割り込み)
- 一定確率で相手側の相槌を発話の途中に重ねる
- 合成結果をキャッシュし、同じ文・同じ声の再合成を省く

使い方:
    .venv/Scripts/python.exe tts/generate_audio_v2.py scenario/generated/s_sg --out results/v2/audio/s_sg \
        [--gap 0.4 0.9] [--overlap-prob 0.0] [--overlap-sec 0.3 1.0] [--backchannel-prob 0.0] \
        [--speed-jitter 0.0] [--speed-scale 1.0] [--seed 0]
"""
import argparse
import hashlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voicevox_client import VoicevoxClient  # noqa: E402

TESTING_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = TESTING_DIR / "results" / "v2" / "tts_cache"
SR = 24000


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def synth(client: VoicevoxClient, text: str, voice: dict, speed: float) -> np.ndarray:
    key = hashlib.sha1(json.dumps([text, voice["speaker_id"], round(speed, 3), voice.get("pitch", 0.0),
                                   voice.get("intonation", 1.0)], ensure_ascii=False).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.wav"
    if not path.exists():
        wav = client.synthesize(text, speaker_id=voice["speaker_id"], speed_scale=speed,
                                pitch_scale=voice.get("pitch", 0.0), intonation_scale=voice.get("intonation", 1.0))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wav)
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert sr == SR, f"想定外のサンプルレート {sr}"
    return x * voice.get("gain", 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scen_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice-map", default=str(TESTING_DIR / "tts" / "voice_map_v2.yaml"))
    ap.add_argument("--gap", type=float, nargs=2, default=[0.4, 0.9])
    ap.add_argument("--overlap-prob", type=float, default=0.0)
    ap.add_argument("--overlap-sec", type=float, nargs=2, default=[0.3, 1.0])
    ap.add_argument("--backchannel-prob", type=float, default=0.0)
    ap.add_argument("--speed-jitter", type=float, default=0.0, help="発話ごとの話速の揺れ幅(±)")
    ap.add_argument("--speed-scale", type=float, default=1.0, help="全発話の話速倍率")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-voice", type=int, default=None, help="全話者を同じVOICEVOX声(style id)にする(声質の比較用)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    scen_dir = Path(args.scen_dir)
    script = load_yaml(scen_dir / "script.yaml")
    voices = load_yaml(Path(args.voice_map))["voices"]
    if args.all_voice is not None:
        voices = {n: {"speaker_id": args.all_voice, "speed": 1.0} for n in voices}
    common = load_yaml(TESTING_DIR / "scenario" / "bank" / "common.yaml")
    wearers = set(script["meta"].get("bt_wearers") or [script["meta"]["bt_wearer"]])
    roles = script["speakers"]
    client = VoicevoxClient()

    placed = []  # (start_sample, samples, turn情報)
    cursor = 0
    prev_end = 0
    for i, turn in enumerate(script["turns"]):
        v = voices[turn["speaker"]]
        speed = v.get("speed", 1.0) * args.speed_scale * (1 + rng.uniform(-args.speed_jitter, args.speed_jitter))
        x = synth(client, turn["text"], v, speed)
        start = cursor
        overlapped = False
        if i > 0 and args.overlap_prob and rng.random() < args.overlap_prob \
                and script["turns"][i - 1]["speaker"] != turn["speaker"]:
            start = max(0, prev_end - int(rng.uniform(*args.overlap_sec) * SR))
            overlapped = True
        placed.append((start, x, {"turn_id": turn["id"], "speaker": turn["speaker"], "role": roles[turn["speaker"]],
                                  "text": turn["text"], "backchannel": False, "overlapped": overlapped}))
        end = start + len(x)
        # 相槌: 発話の途中に、反対側の誰かが短く重ねる
        if args.backchannel_prob and len(x) > 2 * SR and rng.random() < args.backchannel_prob:
            other_role = "our_side" if roles[turn["speaker"]] == "client" else "client"
            cands = [n for n, r in roles.items() if r == other_role]
            who = rng.choice(cands)
            bc = synth(client, rng.choice(common["backchannels"]), voices[who], voices[who].get("speed", 1.0))
            bstart = start + int(rng.uniform(0.4, 0.8) * len(x))
            placed.append((bstart, bc, {"turn_id": turn["id"], "speaker": who, "role": other_role,
                                        "text": "", "backchannel": True, "overlapped": True}))
        gap = rng.uniform(*args.gap) + float(turn.get("wait_seconds", 0) or 0)
        prev_end = end
        cursor = max(cursor, end) + int(gap * SR)
    client.close()

    total = max(s + len(x) for s, x, _ in placed) + SR
    wearer = np.zeros(total, dtype="float32")
    others = np.zeros(total, dtype="float32")
    timeline = []
    for s, x, info in placed:
        is_bt = info["speaker"] in wearers
        (wearer if is_bt else others)[s:s + len(x)] += x
        timeline.append(info | {"is_bt_wearer": is_bt, "start_sec": round(s / SR, 3), "end_sec": round((s + len(x)) / SR, 3)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "wearer.wav", wearer, SR)
    sf.write(out / "others.wav", others, SR)
    (out / "timeline.json").write_text(json.dumps({
        "samplerate": SR, "total_duration_sec": round(total / SR, 3), "scen_dir": str(scen_dir),
        "params": vars(args), "turns": sorted(timeline, key=lambda t: t["start_sec"]),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完了: {out} 総尺 {total / SR / 60:.1f}分 ターン {len(script['turns'])} "
          f"重なり {sum(t['overlapped'] and not t['backchannel'] for t in timeline)} 相槌 {sum(t['backchannel'] for t in timeline)}")


if __name__ == "__main__":
    main()
