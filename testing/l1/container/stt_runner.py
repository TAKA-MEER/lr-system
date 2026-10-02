"""コンテナ内で本番の Transcriber(app/stt/transcriber.py)に5秒WebMチャンク列を順に渡す。

    python /l1/stt_runner.py /work/conds.json /work/stt_out.jsonl

conds.json: [{"cond_id", "chunk_dir"(/work相対), "vad_enabled"(任意), "vad_threshold"(任意)}]
条件ごとに Transcriber のWebMヘッダー状態をリセットする(本番では録音のたびにコンテナを再起動していない
点は L3 試験で別途確認する)。
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")
import yaml  # noqa: E402

from stt.transcriber import Transcriber  # noqa: E402


def main(conds_path: str, out_path: str):
    with open("/app/config/settings.yaml") as f:
        cfg = yaml.safe_load(f)
    conds = json.load(open(conds_path, encoding="utf-8"))
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["cond_id"] for l in open(out_path, encoding="utf-8") if l.strip()}
    st = cfg["stt"]
    tr = Transcriber(model_size=st["model_size"], device=st["device"], compute_type=st["compute_type"])
    for ci, c in enumerate(conds):
        if c["cond_id"] in done:
            continue
        vad = c.get("vad_enabled", st.get("vad_enabled", True))
        vad_params = {
            "min_silence_duration_ms": st["vad_min_silence_ms"],
            "speech_pad_ms": st["vad_speech_pad_ms"],
            "threshold": c.get("vad_threshold", st["vad_threshold"]),
        }
        chunks = sorted(Path("/work", c["chunk_dir"]).glob("*.webm"))
        texts, secs = [], []
        t_all = time.time()
        if hasattr(tr, "transcribe_pcm"):  # 改善版(接続ごとのデコーダ・initial_prompt対応)
            from stt.transcriber import StreamDecoder
            dec = StreamDecoder("l1")
            prompt = c.get("initial_prompt", st.get("initial_prompt", ""))
            for p in chunks:
                t0 = time.time()
                pcm = dec.decode(p.read_bytes())
                segs = tr.transcribe_pcm(pcm, vad, vad_params, prompt) if pcm is not None else []
                texts.append(" ".join(x.text for x in segs))
                secs.append(round(time.time() - t0, 3))
        else:  # 現行版
            tr._webm_header = None
            for p in chunks:
                t0 = time.time()
                texts.append(tr.transcribe_bytes(p.read_bytes(), vad, vad_params))
                secs.append(round(time.time() - t0, 3))
        rec = {"cond_id": c["cond_id"], "texts": texts, "chunk_seconds": secs, "total_seconds": round(time.time() - t_all, 1)}
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{ci+1}/{len(conds)}] {c['cond_id']} chunks={len(chunks)} {rec['total_seconds']}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
