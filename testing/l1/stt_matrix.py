"""L1 文字起こし(STT)の音響限界試験(TEST_PLAN_v2.md A群)。

素材(tts/generate_audio_v2.py の出力)に augment/mix.py で劣化を重ね、本番と同じ
WebM/Opus 5秒チャンクに切ってから、使い捨てコンテナ内で本番の Transcriber に通す。

使い方:
    .venv/Scripts/python.exe l1/stt_matrix.py --sets all --tag base
    .venv/Scripts/python.exe l1/stt_matrix.py --score-only --tag base
結果: results/v2/l1_stt/<tag>/{conds.json, stt_out.jsonl, scores.jsonl, summary.md}
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

HERE = Path(__file__).resolve().parent
TESTING_DIR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TESTING_DIR / "augment"))
sys.path.insert(0, str(TESTING_DIR / "inject"))
sys.path.insert(0, str(TESTING_DIR / "evaluate"))
import docker_util  # noqa: E402
from ebml_cluster_finder import split_into_mediarecorder_like_chunks  # noqa: E402
from mix import SR, active_power, make_mix  # noqa: E402
from v2_metrics import cer, term_accuracy  # noqa: E402
from webm_muxer import mux_to_single_webm  # noqa: E402

AUDIO_DIR = TESTING_DIR / "results" / "v2" / "audio"
PROBE = "s_stt_probe"
PY = sys.executable
VOICE_IDS = {21: "剣崎雌雄", 9: "波音リツ", 52: "雀松朱司", 11: "玄野武宏", 53: "麒ヶ島宗麟", 14: "冥鳴ひまり",
             12: "白上虎太郎", 16: "九州そら", 8: "春日部つむぎ", 2: "四国めたん", 42: "ちび式じい", 19: "九州そら(ささやき)"}


def stems(variant: str = "") -> Path:
    """素材の場所。variant: '' / speed1.3 / voice21 など。無ければ合成する"""
    name = PROBE + (f"_{variant}" if variant else "")
    d = AUDIO_DIR / name
    if not (d / "wearer.wav").exists():
        cmd = [PY, str(TESTING_DIR / "tts" / "generate_audio_v2.py"), str(TESTING_DIR / "scenario" / "generated" / PROBE),
               "--out", str(d)]
        if variant.startswith("speed"):
            cmd += ["--speed-scale", variant[5:]]
        elif variant.startswith("voice"):
            cmd += ["--all-voice", variant[5:]]
        subprocess.run(cmd, check=True)
    return d


def silence_stems(seconds: int) -> Path:
    d = AUDIO_DIR / f"silence_{seconds}s"
    if not (d / "wearer.wav").exists():
        d.mkdir(parents=True, exist_ok=True)
        z = np.zeros(seconds * SR, dtype="float32")
        sf.write(d / "wearer.wav", z, SR)
        sf.write(d / "others.wav", z, SR)
    return d


def conditions(sets: list[str]) -> list[dict]:
    """各条件: {cond_id, set, stems, mix(cond dict), chunk_ms, vad_enabled?, vad_threshold?, ref}"""
    out = []

    def add(set_name, cid, mixc=None, variant="", chunk_ms=5000, **kw):
        out.append({"cond_id": f"{set_name}/{cid}", "set": set_name, "stems": str(stems(variant)),
                    "mix": mixc or {}, "chunk_ms": chunk_ms, "ref": "probe", **kw})

    if "snr" in sets:  # A-1
        add("snr", "clean")
        for s in [30, 20, 15, 10, 5, 0]:
            add("snr", f"factory{s:+d}dB", {"noise": ["factory"], "snr_db": s})
    if "noisetype" in sets:  # A-2
        for s in [20, 10, 5]:
            add("noisetype", f"pink{s:+d}dB", {"noise": ["pink"], "snr_db": s})
        for s in [15, 10, 5]:
            add("noisetype", f"babble{s:+d}dB", {"noise": ["babble"], "babble_snr_db": s})
        for s in [10, 0]:
            add("noisetype", f"alarm{s:+d}dB", {"noise": ["alarm"], "alarm_snr_db": s})
        for per_min, idb in [(6, 0), (30, 0), (30, 6)]:
            add("noisetype", f"impulse{per_min}pm{idb:+d}dB", {"noise": ["impulse"], "impulse_per_min": per_min, "impulse_db": idb})
        add("noisetype", "factory10+impulse30+alarm", {"noise": ["factory", "impulse", "alarm"], "snr_db": 10,
                                                        "impulse_per_min": 30, "impulse_db": 0, "alarm_snr_db": 10})
    if "reverb" in sets:  # A-3
        for rt in [0.3, 0.8, 1.5]:
            add("reverb", f"rt{rt}", {"rt60": rt})
            add("reverb", f"rt{rt}+factory15", {"rt60": rt, "noise": ["factory"], "snr_db": 15})
    if "level" in sets:  # A-4
        for g in [-20, -30, -40]:
            add("level", f"gain{g}dB", {"int_gain_db": g})
        for c in [10, 20]:
            add("level", f"clip+{c}dB", {"clip_db": c})
    if "speed" in sets:  # A-5
        for sp in ["0.8", "1.3", "1.6"]:
            add("speed", f"speed{sp}", variant=f"speed{sp}")
    if "voice" in sets:  # A-6
        for vid in VOICE_IDS:
            add("voice", f"{vid}_{VOICE_IDS[vid]}", variant=f"voice{vid}")
    if "chunk" in sets:  # A-8
        for ms in [10000, 20000]:
            add("chunk", f"chunk{ms // 1000}s", chunk_ms=ms)
            add("chunk", f"chunk{ms // 1000}s+factory10", {"noise": ["factory"], "snr_db": 10}, chunk_ms=ms)
        add("chunk", "single_file", chunk_ms=10_000_000)
    if "vad" in sets:  # A-9
        for mixname, mixc in [("clean", {}), ("factory10", {"noise": ["factory"], "snr_db": 10}),
                              ("factory5", {"noise": ["factory"], "snr_db": 5}), ("babble5", {"noise": ["babble"], "babble_snr_db": 5})]:
            add("vad", f"off/{mixname}", mixc, vad_enabled=False)
            for th in [0.3, 0.5]:
                add("vad", f"th{th}/{mixname}", mixc, vad_threshold=th)
            if mixname == "babble5":
                add("vad", f"th0.05/{mixname}", mixc)
    if "prompt" in sets:  # 改善案: Whisperに専門用語のヒントを与える(initial_prompt)
        vocab = ("工場立会試験の会話です。絶縁抵抗、メガオーム、耐電圧、遮断器、断路器、インターロック、変圧比、"
                 "無負荷損、短絡インピーダンス、温度上昇、ケルビン、デシベル、油中ガス分析、成極指数、始動電流、"
                 "入出力、二重化、タイムアウト、トレンド、重故障。")
        for mixname, mixc in [("clean", {}), ("factory5", {"noise": ["factory"], "snr_db": 5}),
                              ("rt0.8", {"rt60": 0.8})]:
            add("prompt", f"none/{mixname}", mixc, initial_prompt="")
            add("prompt", f"vocab/{mixname}", mixc, initial_prompt=vocab)
    if "noiseonly" in sets:  # A-10 騒音だけが10分続く
        ref_pow = active_power(sf.read(stems() / "wearer.wav", dtype="float32")[0] + sf.read(stems() / "others.wav", dtype="float32")[0])
        sil = str(silence_stems(600))
        for name, mixc in [("factory10", {"noise": ["factory"], "snr_db": 10}), ("factory0", {"noise": ["factory"], "snr_db": 0}),
                           ("babble10", {"noise": ["babble"], "babble_snr_db": 10}),
                           ("impulse30", {"noise": ["impulse"], "impulse_per_min": 30, "impulse_db": 0})]:
            for vad in [True, False]:
                out.append({"cond_id": f"noiseonly/{name}/vad_{'on' if vad else 'off'}", "set": "noiseonly", "stems": sil,
                            "mix": mixc | {"ref_power": ref_pow}, "chunk_ms": 5000, "ref": "none", "vad_enabled": vad})
    return out


def prepare(conds: list[dict], work: Path):
    babble = AUDIO_DIR / "babble.wav"
    for i, c in enumerate(conds):
        cdir = work / "chunks" / c["cond_id"].replace("/", "__")
        if cdir.exists() and any(cdir.glob("*.webm")):
            c["chunk_dir"] = str(cdir.relative_to(work)).replace("\\", "/")
            continue
        tmp = work / "tmp"
        make_mix(Path(c["stems"]), c["mix"], tmp, seed=i, babble_path=babble)
        mux_to_single_webm(str(tmp / "internal_mix.wav"), str(tmp / "internal.webm"), cluster_time_limit_ms=c["chunk_ms"])
        chunks = split_into_mediarecorder_like_chunks((tmp / "internal.webm").read_bytes())
        cdir.mkdir(parents=True, exist_ok=True)
        for j, ch in enumerate(chunks):
            (cdir / f"{j:05d}.webm").write_bytes(ch)
        c["chunk_dir"] = str(cdir.relative_to(work)).replace("\\", "/")
        c["audio_sec"] = round(sf.info(str(tmp / "internal_mix.wav")).duration, 1)
        print(f"prepared {c['cond_id']} chunks={len(chunks)}", flush=True)
    shutil.rmtree(work / "tmp", ignore_errors=True)


def score(work: Path):
    conds = {c["cond_id"]: c for c in json.load(open(work / "conds.json", encoding="utf-8"))}
    script = yaml.safe_load(open(TESTING_DIR / "scenario" / "generated" / PROBE / "script.yaml", encoding="utf-8"))
    expected = yaml.safe_load(open(TESTING_DIR / "scenario" / "generated" / PROBE / "expected.yaml", encoding="utf-8"))
    ref = "".join(t["text"] for t in script["turns"])
    rows = []
    for line in open(work / "stt_out.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        c = conds.get(rec["cond_id"])
        if not c:
            continue
        hyp = "".join(rec["texts"])
        audio_sec = c.get("audio_sec") or len(rec["texts"]) * c["chunk_ms"] / 1000
        row = {"cond_id": rec["cond_id"], "set": c["set"], "rtf": rec["total_seconds"] / audio_sec,
               "chunk_p95": float(np.percentile(rec["chunk_seconds"], 95)) if rec["chunk_seconds"] else None,
               "chunk_max": max(rec["chunk_seconds"]) if rec["chunk_seconds"] else None}
        if c["ref"] == "probe":
            cr = cer(ref, hyp)
            ta = term_accuracy(expected["terms"], hyp)
            row |= {"cer": cr["cer"], "hyp_ratio": cr["hyp_len"] / cr["ref_len"], "term_acc": ta["accuracy"],
                    "missed_terms": [d["alts"][0] for d in ta["details"] if not d["found"]]}
        else:
            row |= {"cer": None, "halluc_chars": len(hyp), "halluc_segments": sum(1 for t in rec["texts"] if t),
                    "halluc_sample": hyp[:80]}
        rows.append(row)
    with open(work / "scores.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def f(v, p=3):
        return "-" if v is None else f"{v:.{p}f}"

    lines = ["| 条件 | CER | 認識長/正解長 | 数値・用語 正解率 | RTF | チャンク処理 p95[s] | 最大[s] | 幻覚文字数 | 取りこぼした用語(先頭5) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['cond_id']} | {f(r.get('cer'))} | {f(r.get('hyp_ratio'), 2)} | {f(r.get('term_acc'), 2)} | "
                     f"{f(r['rtf'], 3)} | {f(r['chunk_p95'], 2)} | {f(r['chunk_max'], 2)} | {r.get('halluc_chars', '-')} | "
                     f"{', '.join(r.get('missed_terms', [])[:5]) or r.get('halluc_sample', '')} |")
    md = "\n".join(lines)
    (work / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="all")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--prepare-only", action="store_true", help="音声の準備だけ行う(GPUを使わない)")
    args = ap.parse_args()
    work = TESTING_DIR / "results" / "v2" / "l1_stt" / args.tag
    work.mkdir(parents=True, exist_ok=True)
    if not args.score_only:
        sets = ["snr", "noisetype", "reverb", "level", "speed", "voice", "chunk", "vad", "noiseonly", "prompt"] \
            if args.sets == "all" else args.sets.split(",")
        conds = conditions(sets)
        prepare(conds, work)
        path = work / "conds.json"
        if path.exists():
            old = {c["cond_id"]: c for c in json.load(open(path, encoding="utf-8"))}
            old |= {c["cond_id"]: c for c in conds}
            conds = list(old.values())
        path.write_text(json.dumps(conds, ensure_ascii=False, indent=1), encoding="utf-8")
        if args.prepare_only:
            return
        docker_util.stop_app()
        docker_util.unload_ollama()
        rc = docker_util.run_in_container("stt_runner.py", ["/work/conds.json", "/work/stt_out.jsonl"], work, work / "run.log")
        print(f"container exit={rc}")
    score(work)


if __name__ == "__main__":
    main()
