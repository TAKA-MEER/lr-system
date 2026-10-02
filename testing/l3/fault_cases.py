"""L3 異常系・運用試験(TEST_PLAN_v2.md E群)。本番コンテナ(minutes-app)に対して実行する。

    .venv/Scripts/python.exe l3/fault_cases.py --cases all
    .venv/Scripts/python.exe l3/fault_cases.py --cases E1,E5
結果: results/v2/l3/<日時>/results.json, summary.md

各ケースは約60秒の短い音声(配電盤台本の冒頭)を使う。音声は時刻をそろえて(実運用と同じく
両マイクが同じ瞬間の音を拾う形で)注入する。
"""
import argparse
import asyncio
import json
import random
import string
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets

HERE = Path(__file__).resolve().parent
TESTING_DIR = HERE.parent
sys.path.insert(0, str(TESTING_DIR / "inject"))
sys.path.insert(0, str(TESTING_DIR / "augment"))
sys.path.insert(0, str(TESTING_DIR / "evaluate"))
from ebml_cluster_finder import split_into_mediarecorder_like_chunks  # noqa: E402
from mix import make_mix  # noqa: E402
from v2_metrics import cer  # noqa: E402

BASE = "http://127.0.0.1:8000"
WS = "ws://127.0.0.1:8000"
STEMS = TESTING_DIR / "results" / "v2" / "audio" / "s_sg_normal"
CLIP_SEC = 60
UI = "v1"  # v1: 現行UIの操作順(録音開始のたびにreset、試験情報→resetの順) / v2: 改善後UIの操作順


def sid() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def encode(wav: Path, out: Path, channels: int = 1, sr: int = 48000, bitrate: str = "32k") -> list[bytes]:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-ar", str(sr), "-ac", str(channels),
                    "-c:a", "libopus", "-b:a", bitrate, "-f", "webm", "-cluster_time_limit", "5000", "-live", "1",
                    str(out)], check=True)
    return split_into_mediarecorder_like_chunks(out.read_bytes())


class Clip:
    """60秒の検証用音声(内蔵・BT)と正解文"""

    def __init__(self, work: Path):
        tl = json.loads((STEMS / "timeline.json").read_text(encoding="utf-8"))
        make_mix(STEMS, {}, work / "full")
        sr = 24000
        for name in ["internal_mix", "bt_mix"]:
            x, _ = sf.read(work / "full" / f"{name}.wav", dtype="float32")
            sf.write(work / f"{name}.wav", x[: CLIP_SEC * sr], sr)
        self.ref = "".join(t["text"] for t in tl["turns"] if t["end_sec"] <= CLIP_SEC and t["text"])
        self.internal = encode(work / "internal_mix.wav", work / "internal.webm")
        self.bt = encode(work / "bt_mix.wav", work / "bt.webm")
        self.internal_stereo = encode(work / "internal_mix.wav", work / "internal_st.webm", channels=2)
        self.work = work


async def send(url: str, chunks: list[bytes], interval: float = 5.0, delay: float = 0.0, close: bool = True,
               received: list | None = None):
    await asyncio.sleep(delay)
    ws = await websockets.connect(url, max_size=None)

    async def rx():
        try:
            async for m in ws:
                if received is not None and isinstance(m, str):
                    received.append(json.loads(m))
        except websockets.ConnectionClosed:
            pass

    task = asyncio.create_task(rx())
    for i, ch in enumerate(chunks):
        await ws.send(ch)
        if i < len(chunks) - 1:
            await asyncio.sleep(interval)
    await asyncio.sleep(3.0)
    if close:
        await ws.close()
        task.cancel()
    return ws


async def record(session: str, internal: list[bytes], bt: list[bytes] | None, bt_lead: float = 0.0):
    """BTを bt_lead 秒先に開始する場合は、BT側に同じ秒数の無音を前置きした音声を渡す必要がある。
    ここでは簡単のためBTとinternalを同時に開始する(bt_lead=0)。"""
    tasks = [send(f"{WS}/ws/audio/internal?session_id={session}", internal, delay=bt_lead)]
    if bt is not None:
        tasks.append(send(f"{WS}/ws/audio/bt?session_id={session}", bt))
    await asyncio.gather(*tasks)


def http(method: str, path: str, **kw):
    with httpx.Client(timeout=kw.pop("timeout", 30.0)) as c:
        return c.request(method, BASE + path, **kw)


def new_session(meta: bool = True) -> str:
    s = sid()
    r = http("POST", "/api/session/reset", params={"session_id": s}, timeout=120)
    r.raise_for_status()
    if meta:
        http("POST", "/api/session/meta", json={"session_id": s, "trial_name": "L3試験", "location": "試験室",
                                                "client_attendees": ["山田 隆"], "our_attendees": ["田中 誠"]})
    return s


def transcript(s: str) -> list[dict]:
    r = http("GET", f"/api/transcript/{s}")
    return r.json().get("segments", []) if r.status_code == 200 else []


def text_cer(clip: Clip, segs: list[dict]) -> float | None:
    return cer(clip.ref, "".join(x["text"] for x in segs))["cer"]


def logs_since(t0: float) -> str:
    r = subprocess.run(["docker", "logs", "--since", str(int(t0)), "minutes-app"], capture_output=True, timeout=60)
    return r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")


def restart_app():
    subprocess.run(["docker", "restart", "minutes-app"], capture_output=True, timeout=120)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            if http("GET", "/", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise TimeoutError("minutes-app が起動しない")


# ------------------------------------------------------------------
# ケース
# ------------------------------------------------------------------

async def e1_second_session_without_restart(clip: Clip) -> dict:
    """E-1: コンテナを再起動せずに2回目の試験を行う"""
    restart_app()
    s1 = new_session()
    await record(s1, clip.internal, clip.bt)
    c1 = text_cer(clip, transcript(s1))
    s2 = new_session()
    t0 = time.time()
    await record(s2, clip.internal, clip.bt)
    c2 = text_cer(clip, transcript(s2))
    lg = logs_since(t0)
    return {"cer_1st": c1, "cer_2nd": c2, "ffmpeg_fail_2nd": lg.count("ffmpeg変換失敗"),
            "judge": "NG" if (c2 or 1) > (c1 or 0) + 0.1 else "OK"}


async def e2_reconnect_midway(clip: Clip) -> dict:
    """E-2: 録音中にブラウザをリロード(内蔵マイクのWebSocketが新しいMediaRecorderで再接続)"""
    restart_app()
    s = new_session()
    half = len(clip.internal) // 2
    # 後半は「新しい録音」なので、後半の音声を改めて先頭からエンコードし直したチャンク(ヘッダー付き)を送る
    x, sr = sf.read(clip.work / "internal_mix.wav", dtype="float32")
    sf.write(clip.work / "second_half.wav", x[half * 5 * sr:], sr)
    second = encode(clip.work / "second_half.wav", clip.work / "second_half.webm")
    t0 = time.time()
    await send(f"{WS}/ws/audio/internal?session_id={s}", clip.internal[:half])
    n_first = len(transcript(s))
    await send(f"{WS}/ws/audio/internal?session_id={s}", second)
    segs = transcript(s)
    lg = logs_since(t0)
    second_text = "".join(x["text"] for x in segs[n_first:])
    return {"segments_first_half": n_first, "segments_second_half": len(segs) - n_first,
            "second_half_text_sample": second_text[:120], "ffmpeg_fail": lg.count("ffmpeg変換失敗"),
            "cer_whole": text_cer(clip, segs)}


async def e3_bt_stereo_mismatch(clip: Clip) -> dict:
    """E-3: BTが先に接続(モノラル)、内蔵マイクはステレオ録音。ヘッダーがBT由来で共有される"""
    restart_app()
    s = new_session()
    t0 = time.time()
    await asyncio.gather(send(f"{WS}/ws/audio/bt?session_id={s}", clip.bt),
                         send(f"{WS}/ws/audio/internal?session_id={s}", clip.internal_stereo, delay=1.0))
    segs = transcript(s)
    lg = logs_since(t0)
    # 対照: 再起動して内蔵ステレオだけを先に接続
    restart_app()
    s2 = new_session()
    await send(f"{WS}/ws/audio/internal?session_id={s2}", clip.internal_stereo)
    segs2 = transcript(s2)
    return {"cer_bt_first": text_cer(clip, segs), "cer_internal_only": text_cer(clip, segs2),
            "ffmpeg_fail": lg.count("ffmpeg変換失敗")}


async def e4_concurrent_sessions(clip: Clip) -> dict:
    """E-4: 2つのセッションを同時に使う。セッションXはBTで自社が話し続け、セッションYはBTなし"""
    restart_app()
    x, y = new_session(), new_session()
    await asyncio.gather(
        send(f"{WS}/ws/audio/bt?session_id={x}", clip.bt),
        send(f"{WS}/ws/audio/internal?session_id={x}", clip.internal),
        send(f"{WS}/ws/audio/internal?session_id={y}", clip.internal, delay=0.5),
    )
    sy = transcript(y)
    return {"y_segments": len(sy), "y_labeled_our_side": sum(1 for z in sy if z["speaker"] == "our_side"),
            "note": "YにはBTが無いので本来すべてclient。our_sideがあればXのBT音量が混入している"}


async def e4b_threshold_persists(clip: Clip) -> dict:
    """E-4b: あるセッションで変えた閾値が、別セッション・リセット後にも残るか"""
    restart_app()
    a = new_session()
    http("POST", "/api/speaker/threshold", json={"session_id": a, "threshold": 99999})
    b = new_session()
    await record(b, clip.internal, clip.bt)
    sb = transcript(b)
    return {"b_segments": len(sb), "b_labeled_our_side": sum(1 for z in sb if z["speaker"] == "our_side"),
            "note": "閾値99999が残っていれば全部clientになる"}


async def e5_generate_during_recording(clip: Clip) -> dict:
    """E-5: 録音中に「議事録を生成」を押す"""
    restart_app()
    s = new_session()
    t0 = time.time()
    rec = asyncio.create_task(record(s, clip.internal, clip.bt))
    await asyncio.sleep(25)
    n_before = len(transcript(s))
    gen = await asyncio.to_thread(http, "POST", "/api/generate", json={"session_id": s}, timeout=900)
    await rec
    await asyncio.sleep(5)
    n_after = len(transcript(s))
    lg = logs_since(t0)
    return {"segments_before_generate": n_before, "segments_after_recording": n_after,
            "generate_status": gen.status_code, "exceptions_in_log": lg.count("例外"),
            "sample_exception": next((l for l in lg.splitlines() if "例外" in l), "")[:200],
            "judge": "NG" if n_after <= n_before + 1 else "OK"}


async def e6_double_generate(clip: Clip) -> dict:
    """E-6: 生成ボタンの二重押下"""
    restart_app()
    s = new_session()
    await record(s, clip.internal, clip.bt)
    t0 = time.time()
    r = await asyncio.gather(asyncio.to_thread(http, "POST", "/api/generate", json={"session_id": s}, timeout=1800),
                             asyncio.to_thread(http, "POST", "/api/generate", json={"session_id": s}, timeout=1800))
    return {"status": [x.status_code for x in r], "files": [x.json().get("filename") if x.status_code == 200 else x.text[:100] for x in r],
            "seconds": round(time.time() - t0, 1), "note": "2回とも処理が走ればLLM処理時間が倍になる"}


async def e7_ollama_down(clip: Clip) -> dict:
    """E-7: 生成中にOllamaが停止 → 復旧後に再生成できるか"""
    restart_app()
    s = new_session()
    await record(s, clip.internal, clip.bt)
    gen = asyncio.create_task(asyncio.to_thread(http, "POST", "/api/generate", json={"session_id": s}, timeout=900))
    await asyncio.sleep(8)
    subprocess.run(["docker", "stop", "minutes-ollama"], capture_output=True, timeout=120)
    r1 = await gen
    subprocess.run(["docker", "start", "minutes-ollama"], capture_output=True, timeout=120)
    await asyncio.sleep(10)
    r2 = await asyncio.to_thread(http, "POST", "/api/generate", json={"session_id": s}, timeout=900)
    return {"first_status": r1.status_code, "first_body": r1.text[:150], "retry_status": r2.status_code,
            "transcript_kept": len(transcript(s)) > 0}


async def e8_container_crash(clip: Clip) -> dict:
    """E-8: 試験中にコンテナが落ちる(再起動)"""
    restart_app()
    s = new_session()
    rec = asyncio.create_task(record(s, clip.internal, clip.bt))
    await asyncio.sleep(30)
    n_before = len(transcript(s))
    subprocess.run(["docker", "restart", "minutes-app"], capture_output=True, timeout=120)
    try:
        await rec
    except Exception as e:  # 送信側は切断で例外になる
        err = type(e).__name__
    else:
        err = None
    restart_app()
    r = http("GET", f"/api/transcript/{s}")
    return {"segments_before_crash": n_before, "transcript_status_after": r.status_code,
            "segments_after": len(r.json().get("segments", [])) if r.status_code == 200 else 0, "sender_error": err}


async def e9_bt_disconnect_loud(clip: Clip) -> dict:
    """E-9: BTが装着者の発話中に切れた後、内蔵マイクだけで録音が続く"""
    restart_app()
    s = new_session()
    # BTの最後のチャンクが「装着者が話している」チャンクになる位置で打ち切る
    tl = json.loads((STEMS / "timeline.json").read_text(encoding="utf-8"))
    loud = [int(t["end_sec"] // 5) for t in tl["turns"] if t["is_bt_wearer"] and t["end_sec"] < CLIP_SEC / 2]
    cut = (loud[-1] + 1) if loud else 3
    await asyncio.gather(send(f"{WS}/ws/audio/bt?session_id={s}", clip.bt[:cut]),
                         send(f"{WS}/ws/audio/internal?session_id={s}", clip.internal))
    segs = transcript(s)
    after = [z for z in segs[cut:]]
    return {"bt_chunks_sent": cut, "segments_after_bt_lost": len(after),
            "after_labeled_our_side": sum(1 for z in after if z["speaker"] == "our_side"),
            "note": "BT切断後は直近のRMS記録(装着者の声)で判定され続けるため、全部our_sideになる恐れ"}


async def e10_garbage(clip: Clip) -> dict:
    """E-10: 壊れたデータ・空データ・巨大データの混入"""
    restart_app()
    s = new_session()
    rnd = bytes(np.random.default_rng(0).integers(0, 256, 20000, dtype=np.uint8))
    chunks = clip.internal[:4] + [rnd, b"\x00" * 10, rnd * 500] + clip.internal[4:8]
    t0 = time.time()
    await send(f"{WS}/ws/audio/internal?session_id={s}", chunks)
    segs = transcript(s)
    lg = logs_since(t0)
    return {"segments": len(segs), "ffmpeg_fail": lg.count("ffmpeg変換失敗"),
            "recovered_after_garbage": len(segs) >= 6, "exceptions": lg.count("例外")}


async def e11_disconnect_detect(clip: Clip) -> dict:
    """E-11: クライアント切断をサーバーが検知するか(ログ)"""
    restart_app()
    s = new_session()
    t0 = time.time()
    await send(f"{WS}/ws/audio/internal?session_id={s}", clip.internal[:3])
    await asyncio.sleep(3)
    lg = logs_since(t0)
    return {"disconnect_logged": "[Internal] 切断" in lg}


async def e12_path_traversal(clip: Clip) -> dict:
    """E-12: ダウンロードAPIのパス検証"""
    results = {}
    for p in ["..%2Fapp%2Fconfig%2Fsettings.yaml", "..%2F..%2Fetc%2Fpasswd", "%2Fetc%2Fpasswd", "....//....//etc/passwd"]:
        r = http("GET", f"/api/download/{p}")
        results[p] = {"status": r.status_code, "body_head": r.text[:60]}
    return results


async def e13_bt_started_after_main(clip: Clip) -> dict:
    """E-13: 操作順の逆転(メイン画面で録音開始した後に、BT画面の録音開始を押す)。ブラウザは開始時に毎回resetを呼ぶ"""
    restart_app()
    s = new_session()
    rec = asyncio.create_task(send(f"{WS}/ws/audio/internal?session_id={s}", clip.internal))
    await asyncio.sleep(30)
    n_before = len(transcript(s))
    http("POST", "/api/session/reset", params={"session_id": s}, timeout=120)  # BT画面の「録音開始」
    bt = asyncio.create_task(send(f"{WS}/ws/audio/bt?session_id={s}", clip.bt[6:]))
    await rec
    await bt
    segs = transcript(s)
    return {"segments_before_bt_start": n_before, "segments_visible_at_end": len(segs),
            "judge": "NG" if len(segs) < n_before else "OK",
            "note": "resetで新しいセッションに置き換わると、それまでの文字起こしが議事録生成の対象から消える"}


async def e14_resume_after_break(clip: Clip) -> dict:
    """E-14: 休憩で録音を停止し、同じ画面で録音を再開する(開始時にresetが呼ばれる)"""
    restart_app()
    s = new_session()
    half = len(clip.internal) // 2
    await send(f"{WS}/ws/audio/internal?session_id={s}", clip.internal[:half])
    n_first = len(transcript(s))
    if UI == "v1":
        http("POST", "/api/session/reset", params={"session_id": s}, timeout=120)  # 再開時の「録音開始」
    x, sr = sf.read(clip.work / "internal_mix.wav", dtype="float32")
    sf.write(clip.work / "resume.wav", x[half * 5 * sr:], sr)
    await send(f"{WS}/ws/audio/internal?session_id={s}", encode(clip.work / "resume.wav", clip.work / "resume.webm"))
    segs = transcript(s)
    return {"segments_before_break": n_first, "segments_at_end": len(segs),
            "first_half_kept": len(segs) > 0 and n_first > 0 and len(segs) > n_first,
            "note": "前半の文字起こしが残っていなければ、休憩をはさむと前半の議事録が失われる"}


async def e15_meta_browser_order(clip: Clip) -> dict:
    """E-15: ブラウザと同じ順序(試験情報の登録 → reset → 録音)で試験情報が議事録に残るか"""
    restart_app()
    s = sid()
    meta = {"session_id": s, "trial_name": "E15確認用の試験名XYZ", "location": "E15試験室",
            "client_attendees": ["山田 隆"], "our_attendees": ["田中 誠"]}
    if UI == "v1":
        r_meta = http("POST", "/api/session/meta", json=meta)
        http("POST", "/api/session/reset", params={"session_id": s}, timeout=120)
    else:
        http("POST", "/api/session/reset", params={"session_id": s}, timeout=120)
        r_meta = http("POST", "/api/session/meta", json=meta)
    await record(s, clip.internal[:6], clip.bt[:6])
    g = http("POST", "/api/generate", json={"session_id": s}, timeout=1800)
    trial = None
    if g.status_code == 200:
        from docx import Document
        path = clip.work / "e15.docx"
        path.write_bytes(http("GET", f"/api/download/{g.json()['filename']}").content)
        doc = Document(str(path))
        trial = next((r.cells[1].text for t in doc.tables for r in t.rows if r.cells[0].text == "試験名"), None)
    return {"meta_status": r_meta.status_code, "generate_status": g.status_code, "trial_name_in_minutes": trial,
            "judge": "OK" if trial and "XYZ" in trial else "NG"}


CASES = {"E1": e1_second_session_without_restart, "E2": e2_reconnect_midway, "E3": e3_bt_stereo_mismatch,
         "E4": e4_concurrent_sessions, "E4b": e4b_threshold_persists, "E5": e5_generate_during_recording,
         "E6": e6_double_generate, "E7": e7_ollama_down, "E8": e8_container_crash, "E9": e9_bt_disconnect_loud,
         "E10": e10_garbage, "E11": e11_disconnect_detect, "E12": e12_path_traversal,
         "E13": e13_bt_started_after_main, "E14": e14_resume_after_break, "E15": e15_meta_browser_order}


async def main_async(cases: list[str], work: Path):
    clip = Clip(work)
    results = {}
    for c in cases:
        print(f"=== {c}: {CASES[c].__doc__.strip().splitlines()[0]}", flush=True)
        t0 = time.time()
        try:
            results[c] = await CASES[c](clip)
        except Exception as e:
            results[c] = {"harness_error": f"{type(e).__name__}: {e}"}
        results[c]["_seconds"] = round(time.time() - t0, 1)
        print(json.dumps(results[c], ensure_ascii=False), flush=True)
        (work / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="all")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ui", default="v1", choices=["v1", "v2"])
    args = ap.parse_args()
    global UI
    UI = args.ui
    cases = list(CASES) if args.cases == "all" else args.cases.split(",")
    work = TESTING_DIR / "results" / "v2" / "l3" / (args.tag or datetime.now().strftime("%Y%m%d_%H%M%S"))
    work.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(main_async(cases, work))
    lines = ["| ケース | 内容 | 結果 |", "| --- | --- | --- |"]
    for c, r in results.items():
        lines.append(f"| {c} | {CASES[c].__doc__.strip().splitlines()[0]} | `{json.dumps(r, ensure_ascii=False)[:400]}` |")
    (work / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
