"""
立会試験シミュレーションのメインオーケストレーター。

流れ:
  1. minutes-appコンテナをrestart(Transcriber/_webm_headerとSpeakerDetectorの
     プロセス共有状態をクリアするため必須)し、起動完了を待つ
  2. セッションreset・meta登録(REST)
  3. internal_mix.wav / bt_mix.wav をWebM/Opusエンコード→Cluster境界で分割
  4. /ws/audio/internal と /ws/audio/bt に実時間ペースで同時注入
  5. /api/transcript のポーリングで文字起こしの安定を待つ
  6. /api/generate → /api/download で議事録docxを取得
  7. results/<run_id>/ に成果物一式を保存

使い方:
    .venv/Scripts/python.exe inject/run_injection.py [--run-id NAME] [--skip-restart]
"""
import argparse
import asyncio
import json
import logging
import random
import string
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws_injector import prepare_chunks, run_injection_ws

TESTING_DIR = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_injection")


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def gen_session_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def unload_ollama_models_for_safety(ollama_base_url: str = "http://127.0.0.1:11434"):
    """restart前に、Ollamaに現在ロードされているモデルを明示的にアンロードしておく。
    モデル切り替え直後などに旧モデルがVRAMに残ったまま新しいwhisperロードと重なり
    OOMになるのを防ぐ(モデル比較検証時の安全策)。"""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{ollama_base_url}/api/ps")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            for m in models:
                name = m["name"]
                logger.info(f"Ollamaモデル {name} を事前アンロード中...")
                client.post(f"{ollama_base_url}/api/generate", json={"model": name, "keep_alive": 0}, timeout=30.0)
    except Exception as e:
        logger.warning(f"Ollama事前アンロードに失敗(続行): {e}")


def restart_app_container(container_name: str = "minutes-app"):
    logger.info(f"docker restart {container_name} を実行中...")
    result = subprocess.run(["docker", "restart", container_name], capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"docker restart失敗: {result.stderr.decode(errors='replace')}")
    logger.info("restartコマンド完了。起動待機中...")


def wait_for_ready(base_url: str, timeout_sec: float = 180.0):
    deadline = time.time() + timeout_sec
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                resp = client.get(base_url + "/")
                if resp.status_code == 200:
                    logger.info("アプリ起動確認OK")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(3)
    raise TimeoutError(f"{base_url} が {timeout_sec}秒以内に起動しませんでした")


def wait_for_quiescence(base_url: str, session_id: str, poll_sec: float, stable_rounds: int, timeout_sec: float):
    logger.info("文字起こしの安定化を待機中...")
    deadline = time.time() + timeout_sec
    last_count = -1
    stable_hits = 0
    with httpx.Client(timeout=10.0) as client:
        while time.time() < deadline:
            resp = client.get(f"{base_url}/api/transcript/{session_id}")
            resp.raise_for_status()
            segments = resp.json()["segments"]
            count = len(segments)
            if count == last_count:
                stable_hits += 1
                if stable_hits >= stable_rounds:
                    logger.info(f"安定化確認(セグメント数={count})")
                    return segments
            else:
                stable_hits = 0
            last_count = count
            time.sleep(poll_sec)
    logger.warning("安定化タイムアウト。現時点の文字起こしで続行します。")
    resp = httpx.get(f"{base_url}/api/transcript/{session_id}", timeout=10.0)
    return resp.json()["segments"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-restart", action="store_true", help="デバッグ用: コンテナ再起動をスキップ")
    args = parser.parse_args()

    cfg = load_yaml(TESTING_DIR / "config.yaml")
    script = load_yaml(TESTING_DIR / cfg["paths"]["script"])

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = TESTING_DIR / cfg["paths"]["results_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = TESTING_DIR / cfg["paths"]["audio_dir"]
    internal_wav = audio_dir / "internal_mix.wav"
    bt_wav = audio_dir / "bt_mix.wav"
    if not internal_wav.exists() or not bt_wav.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {internal_wav}, {bt_wav}。先にtts/generate_audio.pyを実行してください。")

    base_url = cfg["system"]["base_http_url"]
    ws_host = cfg["system"]["host_for_ws"]
    ws_scheme = cfg["system"]["ws_scheme"]

    restart_epoch = time.time()
    if not args.skip_restart:
        unload_ollama_models_for_safety()
        restart_app_container()
        wait_for_ready(base_url, timeout_sec=180.0)
    else:
        logger.info("--skip-restart のため再起動をスキップ")
        wait_for_ready(base_url, timeout_sec=30.0)

    session_id = gen_session_id()
    logger.info(f"run_id={run_id} session_id={session_id}")

    with httpx.Client(timeout=15.0) as client:
        resp = client.post(f"{base_url}/api/session/reset", params={"session_id": session_id})
        resp.raise_for_status()

        meta = script["meta"]
        resp = client.post(
            f"{base_url}/api/session/meta",
            json={
                "session_id": session_id,
                "trial_name": meta["trial_name"],
                "location": meta["location"],
                "client_attendees": meta["client_attendees"],
                "our_attendees": meta["our_attendees"],
            },
        )
        resp.raise_for_status()

    logger.info("音声をWebM/Opusにエンコードし、Cluster境界で分割中...")
    internal_chunks = prepare_chunks(
        internal_wav, run_dir, "internal",
        sample_rate=cfg["audio"]["sample_rate"], bitrate=cfg["audio"]["opus_bitrate"],
        cluster_time_limit_ms=int(cfg["injection"]["chunk_interval_seconds"] * 1000),
    )
    bt_chunks = prepare_chunks(
        bt_wav, run_dir, "bt",
        sample_rate=cfg["audio"]["sample_rate"], bitrate=cfg["audio"]["opus_bitrate"],
        cluster_time_limit_ms=int(cfg["injection"]["chunk_interval_seconds"] * 1000),
    )

    bt_offset_lo, bt_offset_hi = cfg["injection"]["bt_start_offset_seconds"]
    bt_start_offset = random.uniform(bt_offset_lo, bt_offset_hi)
    logger.info(f"BT窓を{bt_start_offset:.1f}秒先行して開始します(実運用の操作手順を模擬)")

    internal_ws_url = f"{ws_scheme}://{ws_host}/ws/audio/internal?session_id={session_id}"
    bt_ws_url = f"{ws_scheme}://{ws_host}/ws/audio/bt?session_id={session_id}"

    t0 = time.time()
    internal_start_wallclock = t0 + bt_start_offset
    asyncio.run(
        run_injection_ws(
            internal_ws_url=internal_ws_url,
            bt_ws_url=bt_ws_url,
            internal_chunks=internal_chunks,
            bt_chunks=bt_chunks,
            interval=cfg["injection"]["chunk_interval_seconds"],
            jitter=cfg["injection"]["chunk_jitter_seconds"],
            bt_start_offset=bt_start_offset,
        )
    )
    injection_duration = time.time() - t0
    logger.info(f"音声注入完了 ({injection_duration:.1f}秒)")

    segments = wait_for_quiescence(
        base_url, session_id,
        poll_sec=cfg["injection"]["quiescence_poll_seconds"],
        stable_rounds=cfg["injection"]["quiescence_stable_rounds"],
        timeout_sec=cfg["injection"]["quiescence_timeout_seconds"],
    )

    with open(run_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump({"session_id": session_id, "segments": segments}, f, ensure_ascii=False, indent=2)
    logger.info(f"文字起こし segments={len(segments)} を保存しました")

    logger.info("議事録生成を要求中(/api/generate)...")
    t1 = time.time()
    with httpx.Client(timeout=600.0) as client:
        resp = client.post(f"{base_url}/api/generate", json={"session_id": session_id})
        resp.raise_for_status()
        gen_result = resp.json()
        generate_duration = time.time() - t1
        logger.info(f"議事録生成完了 ({generate_duration:.1f}秒): {gen_result}")

        filename = gen_result["filename"]
        resp = client.get(f"{base_url}/api/download/{filename}")
        resp.raise_for_status()
        docx_path = run_dir / "minutes.docx"
        docx_path.write_bytes(resp.content)
        logger.info(f"議事録を保存しました: {docx_path}")

    run_info = {
        "run_id": run_id,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "injection_duration_sec": round(injection_duration, 1),
        "generate_duration_sec": round(generate_duration, 1),
        "bt_start_offset_sec": round(bt_start_offset, 2),
        "internal_start_wallclock": internal_start_wallclock,
        "restart_epoch": restart_epoch,
        "num_segments": len(segments),
        "filename": filename,
    }
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完了: {run_dir} ===")
    print(json.dumps(run_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
