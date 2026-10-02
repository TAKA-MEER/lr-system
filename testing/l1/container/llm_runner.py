"""コンテナ内で本番の議事録生成処理(app/llm/pipeline.py)を直接呼ぶ。

    python /l1/llm_runner.py /work/jobs.json /work/out.jsonl

jobs.json: [{"job_id", "full_text", "meta", "llm_overrides"(任意)}]
out.jsonl: 1行1ジョブ {"job_id", "minutes", "info", "error"}。既に out.jsonl にあるジョブは飛ばす(再開可能)。
"""
import asyncio
import json
import os
import sys
import time
import traceback

sys.path.insert(0, "/app")
import yaml  # noqa: E402

from llm.client import OllamaClient  # noqa: E402
from llm.pipeline import build_minutes  # noqa: E402


async def main(jobs_path: str, out_path: str):
    with open("/app/config/settings.yaml") as f:
        cfg = yaml.safe_load(f)
    jobs = json.load(open(jobs_path, encoding="utf-8"))
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["job_id"])
            except Exception:
                pass
    ollama = OllamaClient(base_url=os.environ.get("OLLAMA_HOST", "http://minutes-ollama:11434"))
    for i, job in enumerate(jobs):
        if job["job_id"] in done:
            continue
        llm_cfg = dict(cfg["llm"]) | job.get("llm_overrides", {})
        t0 = time.time()
        rec = {"job_id": job["job_id"], "minutes": None, "info": None, "error": None}
        try:
            minutes, info = await build_minutes(job["full_text"], job["meta"], llm_cfg, ollama)
            rec["minutes"], rec["info"] = minutes, info
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        rec["seconds"] = round(time.time() - t0, 1)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i+1}/{len(jobs)}] {job['job_id']} {rec['seconds']}s error={bool(rec['error'])}", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
