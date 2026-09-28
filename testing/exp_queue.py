"""長時間試験の重複対策の比較実験: 各案(ブランチの作業ツリー)に同じLLM掃引を順に流す。
セッションから切り離して起動する想定(Start-Process)。

    .venv/Scripts/python.exe exp_queue.py base prog-merge topic-cluster stage2-thinking

base は improve/v2-evaluation(このリポジトリの app/)、それ以外は ../lr-system-wt/<名前>/app を使う。
結果: results/v2/l1_llm/exp_<名前>/、進行ログ: results/v2/exp_queue.log
"""
import os
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
REPO = TESTING_DIR.parent
WT = REPO.parent / "lr-system-wt"
ARGS = ["--sets", "long,topics,domain,status", "--repeats", "2", "--topics-n", "5,8,10", "--long-repeats", "2",
        "--num-gpu", "99"]


def pid_alive(pid: int) -> bool:
    r = subprocess.run(["powershell", "-c", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"], capture_output=True)
    return bool(r.stdout.strip())


def main():
    log = TESTING_DIR / "results" / "v2" / "exp_queue.log"
    names = sys.argv[1:]
    if names and names[0].startswith("--wait-pid="):  # 先行するキューの終了を待ってから始める
        pid = int(names.pop(0).split("=")[1])
        while pid_alive(pid):
            time.sleep(60)
    args = list(ARGS)
    if names and names[0].startswith("--sets="):  # 条件セットの指定(既存の結果に追加する)
        args[1] = names.pop(0).split("=", 1)[1]
    for name in names:
        app = REPO / "app" if name == "base" else WT / name / "app"
        env = dict(os.environ, L1_APP_DIR=str(app), PYTHONIOENCODING="utf-8")
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} start {name}\n")
        with open(TESTING_DIR / "results" / "v2" / f"exp_{name}.out", "a", encoding="utf-8") as out:
            rc = subprocess.run([sys.executable, "l1/llm_matrix.py", *args, "--tag", f"exp_{name}"], cwd=TESTING_DIR,
                                env=env, stdout=out, stderr=subprocess.STDOUT).returncode
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} end {name} rc={rc}\n")
    subprocess.run(["docker", "start", "minutes-app"], capture_output=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} all done\n")


if __name__ == "__main__":
    main()
