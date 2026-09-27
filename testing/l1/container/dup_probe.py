"""長時間試験で協議事項が重複する原因の切り分け(コンテナ内で実行)。

    python3 /l1/dup_probe.py /work/jobs.json /work/dup_out.jsonl

各ジョブについて Stage1 を1回だけ実行して要点JSONを保存し、同じ要点JSONから
  A: 本番どおり1回で統合(Stage2)       — 入力・出力トークン数と停止理由を記録
  B: 6個ずつ段階的に統合してから最終統合 — 1回に渡す量を減らす
を作る。A でトークン数が num_ctx に達していないのに重複し、B で重複が減るなら、
コンテキストの上限ではなく「多数の要点を一度にまとめる能力」(モデル性能)の限界と判断できる。
"""
import asyncio
import json
import sys

sys.path.insert(0, "/app")
import httpx  # noqa: E402
import yaml  # noqa: E402

from llm.pipeline import STAGE1_SCHEMA, STAGE2_SCHEMA, _try_parse  # noqa: E402
from llm.prompts import build_stage1_prompt, build_stage2_prompt  # noqa: E402
from postprocess.chunker import split_transcript  # noqa: E402

BASE = "http://minutes-ollama:11434"


async def gen(client, cfg, prompt, schema, max_tokens):
    r = await client.post(f"{BASE}/api/generate", json={
        "model": cfg["model"], "prompt": prompt, "stream": False, "think": False, "format": schema,
        "options": {"num_predict": max_tokens, "temperature": cfg.get("temperature", 0.2),
                    "num_ctx": cfg.get("num_ctx", 32768), "num_gpu": cfg.get("num_gpu", 99)}})
    d = r.json()
    return d.get("response", ""), {"prompt_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count"),
                                   "done_reason": d.get("done_reason"), "num_ctx": cfg.get("num_ctx", 32768)}


async def main(jobs_path, out_path):
    cfg = yaml.safe_load(open("/app/config/settings.yaml"))["llm"]
    s2max = cfg.get("stage2_max_tokens", 8192)
    async with httpx.AsyncClient(timeout=1200) as client:
        for job in json.load(open(jobs_path, encoding="utf-8")):
            chunks = split_transcript(job["full_text"], max_chars=cfg["chunk_chars"])
            s1 = []
            s1_tokens = []
            for ch in chunks:
                raw, st = await gen(client, cfg, build_stage1_prompt(ch), STAGE1_SCHEMA, cfg.get("max_tokens", 4096))
                p = _try_parse(raw)
                if p:
                    s1.append(p)
                s1_tokens.append(st)
            s1_raw = [json.dumps(x, ensure_ascii=False) for x in s1]
            n_s1_topics = sum(len(x.get("discussions", [])) for x in s1)
            # A: 1回で統合
            rawA, stA = await gen(client, cfg, build_stage2_prompt(s1_raw, job["meta"]), STAGE2_SCHEMA, s2max)
            # B: 6個ずつ段階的に統合
            level = s1_raw
            stB = []
            while len(level) > 6:
                nxt = []
                for i in range(0, len(level), 6):
                    raw, st = await gen(client, cfg, build_stage2_prompt(level[i:i + 6], job["meta"]), STAGE2_SCHEMA, s2max)
                    stB.append(st)
                    p = _try_parse(raw)
                    nxt.append(json.dumps({"discussions": p.get("discussions", []), "action_items": p.get("action_items", [])},
                                          ensure_ascii=False) if p else raw)
                level = nxt
            rawB, st = await gen(client, cfg, build_stage2_prompt(level, job["meta"]), STAGE2_SCHEMA, s2max)
            stB.append(st)
            rec = {"job_id": job["job_id"], "stage1_chunks": len(chunks), "stage1_topics_total": n_s1_topics,
                   "stage1_max_prompt_tokens": max(t["prompt_tokens"] or 0 for t in s1_tokens),
                   "A": {"stats": stA, "minutes": _try_parse(rawA)}, "B": {"stats": stB, "minutes": _try_parse(rawB)}}
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(job["job_id"], json.dumps(stA), flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
