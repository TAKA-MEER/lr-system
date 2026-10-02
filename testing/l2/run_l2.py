"""L2 E2E試験(TEST_PLAN_v2.md 1章 L2、D群の一部)を順に実行する。

各条件: 素材(stems)に劣化を重ねた音声を作り、inject/run_injection.py で本番経路に実時間注入し、
evaluate/v2_eval.py で採点する。長時間かかるため、セッションから切り離して起動すること。

    .venv/Scripts/python.exe l2/run_l2.py --runs all --tag base
    .venv/Scripts/python.exe l2/run_l2.py --runs e2e_clean,e2e_factory --tag base
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTING_DIR = HERE.parent
sys.path.insert(0, str(TESTING_DIR / "augment"))
from mix import make_mix  # noqa: E402

PY = sys.executable
AUDIO = TESTING_DIR / "results" / "v2" / "audio"

FACTORY = {"noise": ["factory"], "snr_db": 10, "rt60": 0.5, "bt_leak_db": -20, "bt_noise_snr_db": 20, "bt_band": "16k"}
HARSH = {"noise": ["factory", "impulse", "alarm"], "snr_db": 5, "impulse_per_min": 20, "impulse_db": 0, "alarm_snr_db": 5,
         "rt60": 0.8, "bt_leak_db": -15, "bt_noise_snr_db": 10, "bt_band": "16k"}

# name: (台本, 素材名, 素材生成の追加引数, 劣化条件, run_injection の追加引数)
RUNS = {
    "v1_aligned": ("scenario/script_v1.yaml", None, None, None, []),
    "e2e_clean": ("scenario/generated/s_e2e_mix", "s_e2e_mix", [], {}, []),
    "e2e_factory": ("scenario/generated/s_e2e_mix", "s_e2e_mix", [], FACTORY, []),
    "e2e_harsh": ("scenario/generated/s_e2e_mix", "s_e2e_mix", [], HARSH, []),
    "multi_fast": ("scenario/generated/s_multi", "s_multi_fast",
                   ["--gap", "0.1", "0.4", "--overlap-prob", "0.25", "--backchannel-prob", "0.3", "--seed", "2"],
                   {"noise": ["factory"], "snr_db": 15, "bt_leak_db": -25, "bt_band": "16k"}, []),
    "speed1.5": ("scenario/generated/s_e2e_mix", "s_e2e_mix", [], {}, ["--speed", "1.5", "--no-generate"]),
    "speed2.0": ("scenario/generated/s_e2e_mix", "s_e2e_mix", [], {}, ["--speed", "2.0", "--no-generate"]),
    "speed3.0": ("scenario/generated/s_e2e_mix", "s_e2e_mix", [], {}, ["--speed", "3.0", "--no-generate"]),
    "speed2.0_factory": ("scenario/generated/s_e2e_mix", "s_e2e_mix", [], FACTORY, ["--speed", "2.0", "--no-generate"]),
}


def prepare(name: str, tag: str) -> tuple[str, Path]:
    scen, stem_name, stem_args, cond, _ = RUNS[name]
    if stem_name is None:  # 旧台本(script_v1)は旧音声をそのまま使う
        return scen, TESTING_DIR / "results" / "audio"
    stems = AUDIO / stem_name
    if not (stems / "wearer.wav").exists():
        subprocess.run([PY, str(TESTING_DIR / "tts" / "generate_audio_v2.py"), str(TESTING_DIR / scen), "--out", str(stems),
                        *stem_args], check=True)
    out = TESTING_DIR / "results" / "v2" / "audio_l2" / tag / name
    make_mix(stems, cond, out, seed=7, babble_path=AUDIO / "babble.wav")
    shutil.copy(stems / "timeline.json", out / "timeline.json")
    return scen, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="all")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    names = list(RUNS) if args.runs == "all" else args.runs.split(",")
    for name in names:
        scen, audio_dir = prepare(name, args.tag)
        if args.prepare_only:
            print(f"prepared {name}: {audio_dir}", flush=True)
            continue
        run_id = f"v2/l2/{args.tag}/{name}"
        script = scen if scen.endswith(".yaml") else f"{scen}/script.yaml"
        t0 = time.time()
        rc = subprocess.run([PY, str(TESTING_DIR / "inject" / "run_injection.py"), "--run-id", run_id,
                             "--script", script, "--audio-dir", str(audio_dir.relative_to(TESTING_DIR)),
                             "--monitor-interval", "15", *RUNS[name][4]], cwd=TESTING_DIR).returncode
        print(f"[{name}] run_injection exit={rc} {time.time() - t0:.0f}s", flush=True)
        if rc != 0 or scen.endswith(".yaml"):
            continue
        subprocess.run([PY, str(TESTING_DIR / "evaluate" / "v2_eval.py"), run_id, "--scen", scen,
                        "--audio-dir", str(audio_dir.relative_to(TESTING_DIR)), "--label", f"{args.tag}:{name}"],
                       cwd=TESTING_DIR)


if __name__ == "__main__":
    main()
