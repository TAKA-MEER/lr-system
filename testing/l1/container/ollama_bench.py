"""Ollamaの速度・GPU配置を条件別に測る(D群)。 python /l1/ollama_bench.py /work/bench.json
bench.json: {"prompt_file": "...", "cases": [{"name", "options": {...}}]}"""
import json, sys, time
import httpx
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
prompt = open(cfg["prompt_file"], encoding="utf-8").read()
base = "http://minutes-ollama:11434"
for case in cfg["cases"]:
    httpx.post(f"{base}/api/generate", json={"model": "qwen3.5:9b", "keep_alive": 0}, timeout=60)
    time.sleep(3)
    t0 = time.time()
    r = httpx.post(f"{base}/api/generate", json={"model": "qwen3.5:9b", "prompt": prompt, "stream": False, "think": False,
                   "options": case["options"]}, timeout=1200).json()
    ps = httpx.get(f"{base}/api/ps").json()["models"]
    m = ps[0] if ps else {}
    print(json.dumps({"name": case["name"], "wall": round(time.time() - t0, 1), "load": round(r.get("load_duration", 0) / 1e9, 1),
          "prompt_tok": r.get("prompt_eval_count"), "prompt_tok_s": round(r.get("prompt_eval_count", 0) / max(1e-9, r.get("prompt_eval_duration", 1) / 1e9), 1),
          "eval_tok": r.get("eval_count"), "eval_tok_s": round(r.get("eval_count", 0) / max(1e-9, r.get("eval_duration", 1) / 1e9), 1),
          "size_vram_mib": round(m.get("size_vram", 0) / 2**20), "size_mib": round(m.get("size", 0) / 2**20),
          "error": r.get("error")}, ensure_ascii=False), flush=True)
