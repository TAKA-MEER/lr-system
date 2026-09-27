"""L1試験用: 本番と同じイメージ・同じ app/ を使った使い捨てコンテナで試験スクリプトを動かす。

本番コンテナ(minutes-app)はSTTモデルを常駐させているため、同じGPUで別プロセスから
LLMやSTTを動かすとVRAMの使い方が本番と変わってしまう。L1試験中は minutes-app を止め、
このモジュールで起動する使い捨てコンテナだけがGPUを使う状態にする。
"""
import os
import subprocess
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = TESTING_DIR.parent
IMAGE = "lr-system-minutes-app"
NETWORK = "lr-system_default"


def _win(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def stop_app():
    subprocess.run(["docker", "stop", "minutes-app"], capture_output=True, timeout=120)


def start_app():
    subprocess.run(["docker", "start", "minutes-app"], capture_output=True, timeout=120)


def unload_ollama():
    """コンテナ側Ollamaのモデルをアンロードする(ホスト側にも別のOllamaがいるためdocker exec経由)"""
    r = subprocess.run(["docker", "exec", "minutes-ollama", "ollama", "ps"], capture_output=True, timeout=30)
    for line in r.stdout.decode(errors="replace").splitlines()[1:]:
        name = line.split()[0] if line.strip() else None
        if name:
            subprocess.run(["docker", "exec", "minutes-ollama", "ollama", "stop", name], capture_output=True, timeout=60)


def run_in_container(script_rel: str, args: list[str], work_dir: Path, log_path: Path, gpu: bool = True,
                     extra_env: dict | None = None) -> int:
    """testing/l1/container/<script_rel> を work_dir を /work にマウントして実行する"""
    cmd = ["docker", "run", "--rm", "--network", NETWORK]
    if gpu:
        cmd += ["--gpus", "all"]
    env = {"OLLAMA_HOST": "http://minutes-ollama:11434", "HF_HOME": "/models", "HF_HUB_OFFLINE": "1"}
    env |= extra_env or {}
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [
        "-v", f"{_win(Path(os.environ.get('L1_APP_DIR', REPO_DIR / 'app')))}:/app",  # 改善版の比較用に差し替え可
        "-v", f"{_win(REPO_DIR / 'models')}:/models",
        "-v", f"{_win(TESTING_DIR / 'l1' / 'container')}:/l1",
        "-v", f"{_win(work_dir)}:/work",
        "-w", "/app", IMAGE, "python3", f"/l1/{script_rel}", *args,
    ]
    penv = dict(os.environ, MSYS_NO_PATHCONV="1")
    with open(log_path, "a", encoding="utf-8") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=penv).returncode
