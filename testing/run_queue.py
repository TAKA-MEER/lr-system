"""GPUを使う試験を順番に流す(同時に流すとVRAMの使い方が本番と変わるため)。
セッションから切り離して起動する想定(Start-Process)。

    .venv/Scripts/python.exe run_queue.py --wait-pid 9096 --steps stt,l3,l2 --tag base
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
PY = sys.executable

STEPS = {
    "llm": lambda tag: [PY, "l1/llm_matrix.py", "--sets", "all", "--repeats", "3", "--tag", tag, "--num-gpu", "99"],
    "stt": lambda tag: [PY, "l1/stt_matrix.py", "--sets",
                        "all" if tag == "base" else "snr,reverb,noisetype,speed,noiseonly,prompt", "--tag", tag],
    "l3": lambda tag: [PY, "l3/fault_cases.py", "--cases", "all", "--tag", tag, "--ui", "v1" if tag == "base" else "v2"],
    "l2": lambda tag: [PY, "l2/run_l2.py", "--runs", "all", "--tag", tag],
}


def pid_alive(pid: int) -> bool:
    r = subprocess.run(["powershell", "-c", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"], capture_output=True)
    return bool(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=None)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    log = TESTING_DIR / "results" / "v2" / f"queue_{args.tag}.log"

    def note(msg):
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")

    if args.wait_pid:
        note(f"waiting for pid {args.wait_pid}")
        while pid_alive(args.wait_pid):
            time.sleep(30)
    for step in args.steps.split(","):
        note(f"start {step}")
        if step in ("l3", "l2"):  # 本番コンテナが必要
            subprocess.run(["docker", "start", "minutes-app"], capture_output=True)
            time.sleep(40)
        with open(TESTING_DIR / "results" / "v2" / f"{step}_{args.tag}.out", "a", encoding="utf-8") as out:
            rc = subprocess.run(STEPS[step](args.tag), cwd=TESTING_DIR, stdout=out, stderr=subprocess.STDOUT).returncode
        note(f"end {step} rc={rc}")
    subprocess.run(["docker", "start", "minutes-app"], capture_output=True)
    note("all done")


if __name__ == "__main__":
    main()
